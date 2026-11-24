"""The auto-trading bot.

Every 10 seconds it manages open positions; every 60 seconds it looks for new trades on your
bot markets. It treats the account like a prop-firm account: a daily loss limit, a maximum
drawdown, a cap on total open risk, a cooldown after losing streaks, and no trading into
high-impact news. Reaching the daily target never makes it take more risk - after half the
target it trades smaller and pickier, and at the full target it stops for the day.

Two modes, chosen per market from the facts:
  SHORT  - a confirmed 15-minute signal in a trending market: at most 2 positions, one closing at
           1R and one at 2R; when the first closes in profit, the second is protected at breakeven.
  SCALP  - a strong 1-minute setup that agrees with the 5-minute trend: several small positions,
           each closing at a dollar profit (50c / $1 on small accounts), more positions on bigger
           accounts, only in strong setups.

Exits (whichever comes first): take profit; stop loss; 'protect profit' when a position that
reached 60% of its target gives most of it back (the most common reason good trades turned into
losers); time limits (scalps 15 min if in profit / 30 min max); breakeven on the second short trade.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from .. import config, db, engine
from ..analysis import sentiment
from ..data import market
from ..plan import tp_probability
from ..symbols import resolve
from . import brokers, review
from .sizing import auto_take_profit, max_scalp_positions, size_position

log = logging.getLogger("sigma.bot")

DEFAULTS = {
    "running": False, "account": "paper",               # paper | deriv
    "symbols": ["R_75", "R_100", "1HZ100V", "XAUUSD", "EURUSD", "BTCUSD"],
    "daily_target": 2.0,            # account currency
    "daily_loss_pct": 3.0,          # stop for the day after losing this % of the day's starting equity
    "max_drawdown_pct": 8.0,        # stop completely (until you reset) at this drop from the peak
    "risk_scalp_pct": 1.0,          # max loss per scalp position, % of equity (costs included)
    "risk_short_pct": 1.0,          # max loss per short trade (both positions together), % of equity
    "max_open_risk_pct": 3.0,       # all open stop losses together
    "scalp_tp": None,               # None = automatic (50c / $1 / 0.1% of equity)
    "max_short_positions": 2,
    "max_per_symbol": 3,
    "currency": "USD", "paper_start_balance": 100.0,
    "live_confirmed": False,
}
SCALP_TP_ATR = 0.8          # scalp target distance, in 1-minute ATRs
SCALP_SL_ATR = 1.2          # scalp stop distance
SCALP_MIN_P = 0.62          # minimum chance of reaching the scalp target first
GIVEBACK_TRIGGER = 0.6      # a position that reached 60% of its target...
GIVEBACK_FLOOR = 0.15       # ...and falls back to 15% of it is closed to protect the profit
SCALP_MAX_MIN_IN_PROFIT = 15
SCALP_MAX_MIN = 30
SHORT_MAX_HOURS = 6
NO_ENTRY_BEFORE_CLOSE_MIN = 15   # no new trades this close to a market's close
FLAT_BEFORE_CLOSE_MIN = 5        # close open positions before the market shuts (no weekend/holiday gaps)
MANAGE_EVERY = 10
ENTRY_EVERY = 60

_lock = threading.RLock()
_thread: threading.Thread | None = None
_log: list[dict] = []


def say(msg: str, level: str = "info") -> None:
    _log.append({"t": time.time(), "msg": msg, "level": level})
    del _log[:-200]
    getattr(log, "warning" if level == "warn" else "info")(msg)


def recent_log() -> list[dict]:
    return list(_log[-80:])


# ------------------------------------------------------------------------- settings & state ---
def settings() -> dict:
    return {**DEFAULTS, **(db.kv_get("bot_settings") or {})}


def update_settings(**kw) -> dict:
    s = settings()
    for k, v in kw.items():
        if k not in DEFAULTS or v is None:
            continue
        if k in ("daily_loss_pct", "max_drawdown_pct", "risk_scalp_pct", "risk_short_pct", "max_open_risk_pct"):
            v = float(v)
            limits = {"daily_loss_pct": (0.5, 10), "max_drawdown_pct": (2, 30), "risk_scalp_pct": (0.05, 2),
                      "risk_short_pct": (0.1, 3), "max_open_risk_pct": (0.5, 10)}[k]
            if not limits[0] <= v <= limits[1]:
                raise ValueError(f"{k} must be between {limits[0]} and {limits[1]}")
        if k == "daily_target":
            v = float(v)
            if v <= 0:
                raise ValueError("daily target must be positive")
        if k == "symbols":
            v = [resolve(x).name for x in v if str(x).strip()][:12]
        if k == "account" and v not in ("paper", "deriv"):
            raise ValueError("account must be paper or deriv")
        s[k] = v
    if s["account"] != "paper" and not s.get("live_confirmed"):
        s["running"] = False
    db.kv_set("bot_settings", {k: s[k] for k in DEFAULTS})
    return s


def state() -> dict:
    return db.kv_get("bot_state") or {}


def save_state(st: dict) -> None:
    db.kv_set("bot_state", st)


def today() -> str:
    return datetime.now(ZoneInfo(config.DISPLAY_TZ)).strftime("%Y-%m-%d")


def open_positions(account: str | None = None) -> list[dict]:
    sql = "SELECT * FROM bot_trades WHERE status='open'"
    params: list = []
    if account:
        sql += " AND account=?"
        params.append(account)
    return db.q(sql + " ORDER BY opened_at", params)


def realized_today(account: str) -> float:
    tz = ZoneInfo(config.DISPLAY_TZ)
    d0 = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    r = db.q1("SELECT COALESCE(SUM(pnl),0) v FROM bot_trades WHERE status='closed' AND account=? AND closed_at>=?",
              (account, d0))
    return float(r["v"] or 0)


# ------------------------------------------------------------------------------- risk rules ---
def risk_check(s: dict, broker, equity: float, unrealized: float) -> dict:
    """Updates the day/peak bookkeeping and returns what the bot may do right now."""
    st = state()
    day = today()
    if st.get("day") != day:                              # a new trading day
        if st.get("day") and st.get("day_start_equity"):
            _daily_review(s, st, equity)
        st.update(day=day, day_start_equity=equity, consec_losses=0, cooldown_until=0)
        if (st.get("halted") or {}).get("kind") == "daily":
            st["halted"] = None
    st.setdefault("peak_equity", equity)
    st.setdefault("start_equity", equity)
    st["peak_equity"] = max(st["peak_equity"], equity)
    start = st.get("day_start_equity") or equity
    day_pnl = equity - start
    realized = realized_today(s["account"])
    out = {"can_open": True, "risk_factor": 1.0, "min_p_boost": 0.0, "reason": None, "close_all": False,
           "day_pnl": day_pnl, "realized_today": realized, "day_start_equity": start,
           "peak_equity": st["peak_equity"]}
    halted = st.get("halted")
    if not halted and day_pnl <= -start * s["daily_loss_pct"] / 100:
        halted = st["halted"] = {"kind": "daily", "reason": f"daily loss limit ({s['daily_loss_pct']}%) reached",
                                 "at": time.time()}
        out["close_all"] = True
        say(f"Daily loss limit reached ({day_pnl:+.2f}). Closing everything and stopping until tomorrow.", "warn")
    if not halted and equity <= st["peak_equity"] * (1 - s["max_drawdown_pct"] / 100):
        halted = st["halted"] = {"kind": "drawdown", "reason": f"maximum drawdown ({s['max_drawdown_pct']}%) reached",
                                 "at": time.time()}
        out["close_all"] = True
        say("Maximum drawdown reached. Closing everything; the bot stays stopped until you reset it.", "warn")
    if halted:
        out.update(can_open=False, reason=halted["reason"])
    elif realized >= s["daily_target"]:
        out.update(can_open=False, reason="daily target reached - done for today")
    elif time.time() < (st.get("cooldown_until") or 0):
        out.update(can_open=False, reason="cooling down after a losing streak")
    elif realized >= s["daily_target"] / 2:
        out.update(risk_factor=0.5, min_p_boost=0.05, reason="half the target is banked: smaller, pickier trades")
    save_state(st)
    return out


def _daily_review(s: dict, st: dict, equity: float) -> None:
    """Posts yesterday's self-review (as a notification and in the bot log)."""
    try:
        since = time.time() - 36 * 3600
        x = review.stats(s["account"], since)["overall"]
        if not x.get("trades"):
            return
        start = st["day_start_equity"]
        hit = "target met" if equity - start >= s["daily_target"] else (
            "half the target met" if equity - start >= s["daily_target"] / 2 else "target missed")
        msg = (f"{st['day']}: {equity - start:+.2f} ({hit}) from {x['trades']} trades, "
               f"{x['win_rate']:.0%} wins, profit factor {x['profit_factor'] or '-'}; "
               f"{x['gave_back_after_profit']} trades gave back profit.")
        say("Daily review - " + msg)
        from .. import push
        push.notify("Sigma bot: daily review", msg, url="/#bot", tag="sigma-bot-daily")
    except Exception as e:
        log.debug("daily review: %s", e)


def record_close(trade: dict, pnl_value: float, exit_price: float, reason: str) -> None:
    db.update("bot_trades", trade["id"], {"status": "closed", "closed_at": time.time(), "exit_price": exit_price,
                                          "pnl": round(pnl_value, 4), "close_reason": reason})
    st = state()
    st["consec_losses"] = (st.get("consec_losses", 0) + 1) if pnl_value < 0 else 0
    if st["consec_losses"] >= 3:
        st["cooldown_until"] = time.time() + 30 * 60
        st["consec_losses"] = 0
        say("Three losses in a row: pausing new trades for 30 minutes.", "warn")
    save_state(st)
    say(f"Closed {trade['mode']} {'BUY' if trade['direction'] > 0 else 'SELL'} {trade['symbol']}: "
        f"{pnl_value:+.2f} ({reason})")


# --------------------------------------------------------------------------------- managing ---
def exit_reason(t: dict, p: float, now: float) -> str | None:
    """Smart exit rules for one open position, given its current profit p (in money)."""
    tp, sl = t["tp_usd"], t["sl_usd"]
    best = max(t.get("max_profit") or 0.0, p)
    age_min = (now - t["opened_at"]) / 60
    if p >= tp - max(1e-4, 0.002 * tp):          # rounding tolerance only
        return "take profit"
    if p <= -sl + 1e-6:
        return "stop loss"
    if best >= GIVEBACK_TRIGGER * tp and p <= GIVEBACK_FLOOR * tp:
        return "protect profit (gave back most of the move)"
    if t["mode"] == "scalp":
        if age_min >= SCALP_MAX_MIN_IN_PROFIT and p > 0:
            return "time exit in profit"
        if age_min >= SCALP_MAX_MIN:
            return "time exit"
    else:
        if t.get("breakeven") and p <= 0.02 * tp:
            return "breakeven stop (first target was hit)"
        if age_min >= SHORT_MAX_HOURS * 60:
            return "time exit"
    return None


def _closing_soon(sym: str, minutes: float) -> str | None:
    try:
        from ..data.deriv_info import market_hours
        h = market_hours(sym)
    except Exception:
        return None
    c = h.get("closes_in_min")
    return f"market closes in {c:.0f} min" if c is not None and c <= minutes else None


def manage(broker, s: dict, close_all_reason: str | None = None) -> float:
    """Checks every open position; returns total unrealized profit of those still open."""
    now = time.time()
    unreal = 0.0
    for t in open_positions(s["account"]):
        try:
            m = broker.mark(t)
        except Exception as e:
            say(f"could not price {t['symbol']}: {e}", "warn")
            continue
        p = m["pnl"]
        if m.get("closed"):          # closed on the broker's side (its own take profit / stop loss)
            record_close(t, p, m.get("exit") or m["price"], "closed by broker")
            continue
        if p > (t.get("max_profit") or 0):
            db.update("bot_trades", t["id"], {"max_profit": round(p, 4)})
            t["max_profit"] = p
        reason = close_all_reason or exit_reason(t, p, now) or _closing_soon(t["symbol"], FLAT_BEFORE_CLOSE_MIN)
        if reason:
            try:
                c = broker.close(t["broker_ref"], t)
                record_close(t, c["pnl"], c["exit"], reason)
                if t["mode"] == "short" and c["pnl"] > 0 and t.get("group_id"):
                    for o in open_positions(s["account"]):   # protect the partner position at breakeven
                        if o.get("group_id") == t["group_id"]:
                            db.update("bot_trades", o["id"], {"breakeven": 1})
                            say(f"First target hit on {t['symbol']}: second position protected at breakeven.")
            except Exception as e:
                say(f"close failed for {t['symbol']}: {e}", "warn")
        else:
            unreal += p
    return unreal


# ---------------------------------------------------------------------------------- entries ---
def _scalp_setup(sym: str, risk: dict) -> dict | None:
    df1 = market.closed_candles(sym, "M1", 400)
    df5 = market.closed_candles(sym, "M5", 300)
    r = engine.evaluate(sym, "M1", df1, df5)
    if engine.market_closed(r["bar_time"], "M1"):
        return None
    d = r["direction"]
    if r["components"].get("htf_trend", 0) * d < 0:          # never scalp against the 5-minute trend
        return None
    atr = r["atr"]
    tp_d, sl_d = SCALP_TP_ATR * atr, SCALP_SL_ATR * atr
    p_main = r["p_long"] if d > 0 else r["p_short"]
    p_tp = tp_probability(p_main, SCALP_TP_ATR / SCALP_SL_ATR)
    need = SCALP_MIN_P + risk["min_p_boost"]
    if p_tp < need:
        return None
    return {"mode": "scalp", "symbol": sym, "direction": d, "p": p_tp, "tp_dist": tp_d, "sl_dist": sl_d,
            "regime": r["regime"], "components": r["components"], "setup": r["plan"]["setup_type"],
            "count": 1 if p_tp < need + 0.06 else 2 if p_tp < need + 0.12 else 3}


_short_checked: dict[str, int] = {}


def _short_setup(sym: str, risk: dict) -> dict | None:
    bucket = int(time.time() // 900)          # a full analysis once per new 15-minute candle, not every minute
    if _short_checked.get(sym) == bucket:
        return None
    _short_checked[sym] = bucket
    r = engine.analyze(sym, "M15", source="live", journal=False)
    if r.get("market_closed") or not r.get("is_signal") or r["regime"] == "ranging":
        return None
    if r["plan"]["kind"] != "MARKET":        # the bot only enters at market; pending ideas are skipped
        return None
    if r["probability"] < config.MIN_PROBABILITY + risk["min_p_boost"]:
        return None
    sl_d = abs(r["entry"] - r["sl"])
    return {"mode": "short", "symbol": sym, "direction": r["direction"], "p": r["probability"],
            "tp_dist": sl_d, "sl_dist": sl_d, "regime": r["regime"], "components": r["components"],
            "setup": r["plan"]["setup_type"], "count": 2}


def look_for_entries(broker, s: dict, equity: float, risk: dict) -> int:
    if not risk["can_open"]:
        return 0
    opened = 0
    open_ = open_positions(s["account"])
    open_risk = sum(t["sl_usd"] for t in open_)
    max_risk = equity * s["max_open_risk_pct"] / 100
    learned = review.learning()
    for sym in s["symbols"]:
        inst = resolve(sym)
        ev = sentiment.event_risk(inst)
        if ev.get("blocked") or engine.is_closed_now(sym) or _closing_soon(sym, NO_ENTRY_BEFORE_CLOSE_MIN):
            continue
        for setup_fn in (_short_setup, _scalp_setup):
            try:
                st = setup_fn(sym, risk)
            except Exception as e:
                log.debug("setup %s: %s", sym, e)
                continue
            if not st:
                continue
            key = f"{st['mode']}|{inst.name}"
            if key in learned.get("paused", {}):
                continue
            mine = [t for t in open_ if t["symbol"] == inst.name]
            if len(mine) >= s["max_per_symbol"] or any(t["direction"] != st["direction"] for t in mine):
                continue
            mode_open = [t for t in open_ if t["mode"] == st["mode"]]
            limit = s["max_short_positions"] if st["mode"] == "short" else max_scalp_positions(equity)
            n = min(st["count"], limit - len(mode_open), s["max_per_symbol"] - len(mine))
            if n <= 0:
                continue
            opened += _open(broker, s, st, n, equity, risk, max_risk - open_risk, learned)
            open_ = open_positions(s["account"])
            open_risk = sum(t["sl_usd"] for t in open_)
            break                       # one setup per market per cycle
    return opened


def _open(broker, s: dict, st: dict, n: int, equity: float, risk: dict, risk_room: float, learned: dict) -> int:
    sym, d = st["symbol"], st["direction"]
    price = broker.price(sym)
    group = f"g{int(time.time() * 1000)}"
    done = 0
    for i in range(n):
        if st["mode"] == "scalp":
            base_tp = s["scalp_tp"] or auto_take_profit(equity)
            tp_usd = base_tp * learned.get("scalp_tp_factor", 1.0) * risk["risk_factor"]
            max_loss = equity * s["risk_scalp_pct"] / 100 * risk["risk_factor"]
            tp_d = st["tp_dist"]
        else:   # two positions: the first targets 1R, the second 2R; risk split between them
            max_loss = equity * s["risk_short_pct"] / 100 * risk["risk_factor"] / 2
            r_mult = 1 if i == 0 else 2
            tp_d = st["sl_dist"] * r_mult
            tp_usd = max_loss * r_mult          # shrunk by size_position if the stop needs it
        max_loss = min(max_loss, risk_room)
        if max_loss <= 0:
            say("Open-risk cap reached: no new positions until some close.")
            break
        size = size_position(price, tp_d, st["sl_dist"], tp_usd, max_loss, broker.multipliers(sym),
                             broker.min_stake(sym), broker.cost_fraction(sym))
        if size is None:
            say(f"{sym}: account too small for a {st['mode']} position within the risk limits - skipped.")
            break
        tp_price = price + d * tp_d
        sl_price = price - d * st["sl_dist"]
        try:
            o = broker.open(sym, d, size, tp_price, sl_price)
        except Exception as e:
            say(f"{sym}: order rejected - {e}", "warn")
            break
        db.insert("bot_trades", {
            "account": s["account"], "mode": st["mode"], "symbol": resolve(sym).name, "direction": d,
            "stake": size.stake, "multiplier": size.multiplier, "exposure": size.exposure, "cost": size.cost_usd,
            "entry": o["entry"], "tp_price": tp_price, "sl_price": sl_price, "tp_usd": size.tp_usd,
            "sl_usd": size.sl_usd, "opened_at": time.time(), "status": "open", "broker_ref": o["ref"],
            "group_id": group, "max_profit": 0.0, "p_win": round(st["p"], 3), "regime": st["regime"],
            "setup": st["setup"], "components": st["components"],
        })
        risk_room -= size.sl_usd
        done += 1
    if done:
        say(f"Opened {done} {st['mode']} {'BUY' if d > 0 else 'SELL'} {sym} "
            f"(chance {st['p']:.0%}, {st['setup'].lower()})")
    return done


# ------------------------------------------------------------------------------------ loop ---
def step(broker=None, now: float | None = None, entries: bool = True) -> dict:
    """One cycle: risk check, manage positions, maybe look for entries. Returns a status snapshot."""
    s = settings()
    broker = broker or brokers.get(s["account"])
    unreal = manage(broker, s)
    equity = broker.balance() + unreal
    risk = risk_check(s, broker, equity, unreal)
    if risk["close_all"]:
        manage(broker, s, close_all_reason=risk["reason"])
    opened = look_for_entries(broker, s, equity, risk) if entries and s["running"] else 0
    return {"equity": equity, "unrealized": unreal, "risk": risk, "opened": opened}


def _loop() -> None:
    last_entry = 0.0
    broker = None
    say("Bot started.")
    while settings()["running"]:
        s = settings()
        try:
            if broker is None or broker.name != s["account"]:
                broker = brokers.get(s["account"])
            due = time.time() - last_entry >= ENTRY_EVERY
            step(broker, entries=due)
            if due:
                last_entry = time.time()
                review.maybe_learn()
        except Exception as e:
            say(f"bot cycle error: {e}", "warn")
            broker = None
        time.sleep(MANAGE_EVERY)
    say("Bot stopped.")


def start() -> dict:
    global _thread
    s = settings()
    if s["account"] != "paper" and not s.get("live_confirmed"):
        raise ValueError("Live trading needs your confirmation first.")
    st = state()
    if (st.get("halted") or {}).get("kind") == "drawdown":
        raise ValueError("The bot hit its maximum drawdown. Review the results and reset it before restarting.")
    update_settings(running=True)
    with _lock:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_loop, daemon=True, name="autobot")
            _thread.start()
    return status()


def stop(close_positions: bool = False) -> dict:
    s = update_settings(running=False)
    if close_positions:
        try:
            manage(brokers.get(s["account"]), s, close_all_reason="stopped by you")
        except Exception as e:
            say(f"closing on stop failed: {e}", "warn")
    return status()


def resume_if_running() -> None:
    """After a restart (deploy), carry on if the bot was running."""
    if settings()["running"]:
        try:
            start()
        except Exception as e:
            say(f"could not resume: {e}", "warn")


def reset(start_balance: float | None = None) -> dict:
    """Clears a halt and the peak/day bookkeeping; for paper, also refills the paper account."""
    s = settings()
    if s["running"]:
        raise ValueError("Stop the bot before resetting.")
    if s["account"] == "paper":
        brokers.PaperBroker().reset(float(start_balance or s["paper_start_balance"]), s["currency"])
        db.execute("DELETE FROM bot_trades WHERE account='paper' AND status='open'")
        if start_balance:
            update_settings(paper_start_balance=float(start_balance))
    save_state({})
    say("Bot reset.")
    return status()


def status() -> dict:
    s = settings()
    st = state()
    out = {"settings": s, "state": st, "log": recent_log(), "thread_alive": bool(_thread and _thread.is_alive())}
    try:
        b = brokers.get(s["account"])
        bal = b.balance()
        pos = open_positions(s["account"])
        marks = []
        unreal = 0.0
        for t in pos:
            try:
                m = b.mark(t)
                t["pnl_now"] = round(m["pnl"], 4)
                t["price_now"] = m["price"]
                unreal += m["pnl"]
            except Exception:
                t["pnl_now"] = None
            marks.append(t)
        start = st.get("day_start_equity") or bal
        out.update(balance=round(bal, 2), equity=round(bal + unreal, 2), unrealized=round(unreal, 2),
                   positions=marks, realized_today=round(realized_today(s["account"]), 2),
                   day_pnl=round(bal + unreal - start, 2), currency=getattr(b, "currency", s["currency"]),
                   max_scalps=max_scalp_positions(bal + unreal),
                   scalp_tp_now=round((s["scalp_tp"] or auto_take_profit(bal + unreal)) *
                                      review.learning().get("scalp_tp_factor", 1.0), 2))
        tgt = s["daily_target"]
        out["target_progress"] = round(out["realized_today"] / tgt, 3) if tgt else None
        out["target_warning"] = (f"A daily target of {tgt} is {100 * tgt / max(bal, 1e-9):.1f}% of the account. "
                                 "Above ~2% a day is very aggressive; the bot will not take extra risk to reach it."
                                 if bal and tgt / bal > 0.02 else None)
    except Exception as e:
        out["error"] = str(e)
    return out
