"""Copy the robot's paper trades to a real account, sized for that account, with a trailing stop.

Every paper trade on the demo account (robot signals, and optionally the trades from analyses you ask
for in the chat) is copied when it opens - pending limit/stop orders are copied when they fill:

  Same idea, real size  - same direction, same stop-loss and take-profit PRICE LEVELS as the paper
                          trade, but the position is sized from the real account: the loss at the
                          stop is the chosen % of the real balance (1% by default), whatever the paper
                          trade risked. On Deriv the size is a Multipliers stake (Deriv has no lots).
  Only fresh trades     - a copy is placed only within 5 minutes of the paper trade opening and while
                          price is still within 0.3R of its entry; otherwise it is logged as skipped
                          (chasing a trade that already moved changes its risk/reward).
  Trailing stop         - once the copy is 1R in profit its stop moves to breakeven (plus costs); after
                          that it trails 1R behind the best price reached (e.g. best +1.8R -> stop +0.8R).
                          Checked every 10 seconds. The original stop loss and take profit also sit on
                          the broker's side, so they still work if Sigma is offline.
  In sync               - when the paper trade closes (take profit, stop loss, time limit) the copy is
                          closed too, if the trailing stop hasn't closed it already.
  Account guard         - no new copies after the real account loses the daily limit (3%) today, and at
                          most N copies open at once.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from . import config, db
from .autobot import brokers
from .autobot.sizing import size_position
from .data import market
from .symbols import resolve

log = logging.getLogger("sigma.copy")

DEFAULTS = {
    "running": False,
    "account": "paper",          # paper = the copier's own test account | deriv = real Deriv account
    "risk_pct": 1.0,             # loss at the stop, % of the real balance
    "max_open": 3,
    "daily_loss_pct": 3.0,
    "copy": "all",               # all = everything on the demo account | signals = robot signals only
    "trail_start_r": 1.0,        # start trailing once this far in profit (in R)
    "trail_distance_r": 1.0,     # trail this far behind the best price
    "paper_start_balance": 100.0,
    "live_confirmed": False,
    "started_at": None,
}
MAX_LATE_SEC = 300
MAX_DRIFT_R = 0.3
BE_LOCK_R = 0.1                  # breakeven stop sits slightly in profit to cover costs
LOOP_EVERY = 10

_thread: threading.Thread | None = None
_lock = threading.RLock()
_log: list[dict] = []


def say(msg: str, level: str = "info") -> None:
    _log.append({"t": time.time(), "msg": msg, "level": level})
    del _log[:-150]
    getattr(log, "warning" if level == "warn" else "info")(msg)


# --------------------------------------------------------------------------------- settings ---
def settings() -> dict:
    return {**DEFAULTS, **(db.kv_get("copy_settings") or {})}


def update_settings(**kw) -> dict:
    s = settings()
    limits = {"risk_pct": (0.1, 3), "max_open": (1, 10), "daily_loss_pct": (0.5, 10),
              "trail_start_r": (0.3, 3), "trail_distance_r": (0.3, 3), "paper_start_balance": (10, 1e7)}
    for k, v in kw.items():
        if k not in DEFAULTS or v is None:
            continue
        if k in limits:
            v = float(v)
            lo, hi = limits[k]
            if not lo <= v <= hi:
                raise ValueError(f"{k.replace('_', ' ')} must be between {lo:g} and {hi:g}")
            if k == "max_open":
                v = int(v)
        if k == "account" and v not in ("paper", "deriv"):
            raise ValueError("account must be paper or deriv")
        if k == "copy" and v not in ("all", "signals"):
            raise ValueError("copy must be all or signals")
        s[k] = v
    if s["account"] != "paper" and not s.get("live_confirmed"):
        s["running"] = False
    db.kv_set("copy_settings", {k: s[k] for k in DEFAULTS})
    return s


def broker(s: dict | None = None):
    s = s or settings()
    if s["account"] == "deriv":
        return brokers.DerivBroker()
    return brokers.PaperBroker(float(s["paper_start_balance"]), "USD", key="copy_paper")


def open_copies(account: str) -> list[dict]:
    return db.q("SELECT * FROM copy_trades WHERE status='open' AND account=? ORDER BY opened_at", (account,))


def _day_start() -> float:
    tz = ZoneInfo(config.DISPLAY_TZ)
    return datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def realized_today(account: str) -> float:
    r = db.q1("SELECT COALESCE(SUM(pnl),0) v FROM copy_trades WHERE status='closed' AND account=? AND closed_at>=?",
              (account, _day_start()))
    return float(r["v"] or 0)


# ------------------------------------------------------------------------------- trailing ---
def trail_stop_r(best_r: float, s: dict) -> float | None:
    """The trailing stop (in R from entry) for the best profit reached so far; None = not trailing yet."""
    if best_r < s["trail_start_r"]:
        return None
    return max(BE_LOCK_R, best_r - s["trail_distance_r"])


def r_of(c: dict, price: float) -> float:
    return c["direction"] * (price - c["entry"]) / c["risk_dist"]


# ---------------------------------------------------------------------------------- copying ---
def candidates(s: dict) -> list[dict]:
    sql = ("SELECT t.*, a.source a_source, a.is_signal a_signal FROM trades t LEFT JOIN analyses a ON a.id=t.analysis_id "
           "WHERE t.demo=1 AND t.status='open' AND t.opened_at>=? "
           "AND NOT EXISTS (SELECT 1 FROM copy_trades c WHERE c.trade_id=t.id AND c.account=?) ORDER BY t.opened_at, t.id")
    rows = db.q(sql, (s.get("started_at") or time.time(), s["account"]))
    if s["copy"] == "signals":
        rows = [r for r in rows if r.get("a_source") != "chat" and r.get("origin") != "user"]
    return rows


def _skip(t: dict, s: dict, why: str) -> None:
    db.insert("copy_trades", {"trade_id": t["id"], "account": s["account"], "symbol": t["symbol"],
                              "direction": t["direction"], "entry": t["entry"], "opened_at": time.time(),
                              "status": "skipped", "note": why})
    say(f"Skipped {'BUY' if t['direction'] > 0 else 'SELL'} {t['symbol']}: {why}")


def copy_new(b, s: dict, now: float | None = None) -> int:
    now = now or time.time()
    done = 0
    for t in candidates(s):
        if now - t["opened_at"] > MAX_LATE_SEC:
            _skip(t, s, f"seen {int((now - t['opened_at']) / 60)} min after it opened - too late to copy")
            continue
        if len(open_copies(s["account"])) >= s["max_open"]:
            _skip(t, s, f"{s['max_open']} copies already open")
            continue
        bal = b.balance()
        day_loss = realized_today(s["account"])
        if day_loss <= -(bal - day_loss) * s["daily_loss_pct"] / 100:
            _skip(t, s, f"daily loss limit ({s['daily_loss_pct']}%) reached on the real account")
            continue
        d = t["direction"]
        try:
            px = b.price(t["symbol"])
        except Exception as e:
            say(f"{t['symbol']}: no price ({e}) - will retry", "warn")
            continue
        risk0 = abs(t["entry"] - t["sl"])
        drift = d * (px - t["entry"]) / risk0 if risk0 else 0
        if abs(drift) > MAX_DRIFT_R:
            _skip(t, s, f"price already moved {drift:+.2f}R from the paper entry")
            continue
        sl_d, tp_d = abs(px - t["sl"]), abs(t["tp"] - px)
        if (t["sl"] - px) * d >= 0 or (t["tp"] - px) * d <= 0:
            _skip(t, s, "price is beyond the stop or target")
            continue
        risk_usd = bal * s["risk_pct"] / 100
        size = size_position(px, tp_d, sl_d, risk_usd * tp_d / sl_d, risk_usd, b.multipliers(t["symbol"]),
                             b.min_stake(t["symbol"]), b.cost_fraction(t["symbol"]))
        if size is None:
            _skip(t, s, "too small for the broker's minimum stake at this risk, or costs too high")
            continue
        try:
            o = b.open(t["symbol"], d, size, t["tp"], t["sl"])
        except Exception as e:
            _skip(t, s, f"broker refused: {e}")
            continue
        entry = o["entry"]
        db.insert("copy_trades", {
            "trade_id": t["id"], "account": s["account"], "symbol": t["symbol"], "direction": d,
            "entry": entry, "sl_price": t["sl"], "tp_price": t["tp"], "risk_dist": abs(entry - t["sl"]) or sl_d,
            "stake": size.stake, "multiplier": size.multiplier, "exposure": size.exposure, "cost": size.cost_usd,
            "tp_usd": size.tp_usd, "sl_usd": size.sl_usd, "opened_at": time.time(), "status": "open",
            "best_r": 0.0, "broker_ref": o["ref"]})
        say(f"Copied {'BUY' if d > 0 else 'SELL'} {t['symbol']} at {entry:.5g}: risk {size.sl_usd:.2f}, "
            f"target {size.tp_usd:.2f} (stake {size.stake:.2f} x{size.multiplier})")
        done += 1
    return done


def _close(b, c: dict, reason: str) -> None:
    try:
        r = b.close(c["broker_ref"], c)
    except Exception as e:
        say(f"close failed for {c['symbol']}: {e}", "warn")
        return
    db.update("copy_trades", c["id"], {"status": "closed", "closed_at": time.time(), "exit_price": r["exit"],
                                       "pnl": round(r["pnl"], 4), "close_reason": reason})
    say(f"Closed {c['symbol']} copy: {r['pnl']:+.2f} ({reason})")


def manage(b, s: dict, close_all: str | None = None) -> None:
    for c in open_copies(s["account"]):
        try:
            m = b.mark(c)
        except Exception as e:
            say(f"could not price {c['symbol']}: {e}", "warn")
            continue
        if m.get("closed"):
            p = m["pnl"]
            db.update("copy_trades", c["id"], {"status": "closed", "closed_at": time.time(),
                                               "exit_price": m.get("exit") or m["price"], "pnl": round(p, 4),
                                               "close_reason": "take profit (on broker)" if p > 0 else "stop loss (on broker)"})
            say(f"{c['symbol']} copy closed by the broker: {p:+.2f}")
            continue
        if close_all:
            _close(b, c, close_all)
            continue
        r = r_of(c, m["price"])
        best = max(c.get("best_r") or 0.0, r)
        stop = trail_stop_r(best, s)
        if best > (c.get("best_r") or 0) or stop != c.get("trail_r"):
            db.update("copy_trades", c["id"], {"best_r": round(best, 3), "trail_r": stop})
            if stop is not None and c.get("trail_r") is None:
                say(f"{c['symbol']} copy is +{best:.1f}R: stop moved to {'breakeven' if stop <= BE_LOCK_R else f'+{stop:.1f}R'}")
        tp_r = abs(c["tp_price"] - c["entry"]) / c["risk_dist"]
        paper = db.q1("SELECT status FROM trades WHERE id=?", (c["trade_id"],))
        if r >= tp_r:
            _close(b, c, "take profit")
        elif r <= -1:
            _close(b, c, "stop loss")
        elif stop is not None and r <= stop:
            _close(b, c, f"trailing stop (locked {stop:+.1f}R, best {best:+.1f}R)")
        elif paper and paper["status"] not in ("open", "pending"):
            _close(b, c, f"paper trade closed ({paper['status']})")


# ------------------------------------------------------------------------------------ loop ---
def step(b=None, now: float | None = None) -> None:
    s = settings()
    b = b or broker(s)
    manage(b, s)
    if s["running"]:
        copy_new(b, s, now)


def _loop() -> None:
    b = None
    say("Copy trading started.")
    time.sleep(2)
    while settings()["running"]:
        s = settings()
        try:
            if b is None or b.name != s["account"]:
                b = broker(s)
            step(b)
        except Exception as e:
            say(f"copy cycle error: {e}", "warn")
            b = None
        time.sleep(LOOP_EVERY)
    say("Copy trading stopped (open copies keep their trailing stop only while it runs; "
        "their stop loss and take profit stay on the broker).")


def start() -> dict:
    global _thread
    s = settings()
    if s["account"] != "paper" and not s.get("live_confirmed"):
        raise ValueError("Copying to a real account needs your confirmation first.")
    update_settings(running=True, started_at=s.get("started_at") if s["running"] else time.time())
    with _lock:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_loop, daemon=True, name="copytrade")
            _thread.start()
    return status(light=True)


def stop(close_all: bool = False) -> dict:
    s = update_settings(running=False)
    if close_all:
        threading.Thread(target=lambda: manage(broker(s), s, close_all="closed by you"), daemon=True).start()
    return status(light=True)


def resume_if_running() -> None:
    if settings()["running"]:
        try:
            start()
        except Exception as e:
            say(f"could not resume: {e}", "warn")


def reset_paper(start_balance: float | None = None) -> dict:
    s = settings()
    if s["running"]:
        raise ValueError("Stop copy trading before resetting its test account.")
    if start_balance:
        s = update_settings(paper_start_balance=float(start_balance))
    brokers.PaperBroker(key="copy_paper").reset(float(s["paper_start_balance"]))
    db.execute("DELETE FROM copy_trades WHERE account='paper'")
    say("Copy test account reset.")
    return status(light=True)


def status(light: bool = False) -> dict:
    s = settings()
    out = {"settings": s, "log": list(_log[-60:]), "thread_alive": bool(_thread and _thread.is_alive())}
    if light:
        return out
    try:
        b = broker(s)
        bal = b.balance()
        pos = []
        for c in open_copies(s["account"]):
            try:
                m = b.mark(c)
                c["pnl_now"], c["price_now"], c["r_now"] = round(m["pnl"], 4), m["price"], round(r_of(c, m["price"]), 2)
            except Exception:
                c["pnl_now"] = None
            pos.append(c)
        rows = db.q("SELECT * FROM copy_trades WHERE account=? AND status IN ('closed','skipped') "
                    "ORDER BY COALESCE(closed_at, opened_at) DESC LIMIT 60", (s["account"],))
        closed = [r for r in rows if r["status"] == "closed"]
        allc = db.q("SELECT pnl, close_reason FROM copy_trades WHERE account=? AND status='closed'", (s["account"],))
        wins = [r["pnl"] for r in allc if r["pnl"] > 0]
        out.update(balance=round(bal, 2), currency=getattr(b, "currency", "USD"), positions=pos,
                   history=rows, realized_today=round(realized_today(s["account"]), 2),
                   stats={"trades": len(allc), "win_rate": len(wins) / len(allc) if allc else None,
                          "net": round(sum(r["pnl"] for r in allc), 2),
                          "trailing_exits": sum(1 for r in allc if (r["close_reason"] or "").startswith("trailing"))},
                   recent_closed=len(closed))
    except Exception as e:
        out["error"] = str(e)
    return out
