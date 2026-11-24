"""Paper broker: opens a simulated trade for every analysis and resolves it
against real subsequent prices. Resolved trades feed the learning model."""
from __future__ import annotations

import logging
import time
from collections import defaultdict

import pandas as pd

from . import config, db, learning
from .data import market
from .engine import cost_in_price
from .symbols import resolve, tf_seconds

log = logging.getLogger("sigma.paper")


def open_trade(res: dict, source: str = "live", entry_price: float | None = None,
               opened_at: float | None = None) -> int:
    if not res.get("analysis_id"):
        return 0
    dup = db.q1("SELECT t.id FROM trades t JOIN analyses a ON a.id=t.analysis_id "
                "WHERE a.symbol=? AND a.timeframe=? AND a.bar_time=? AND t.direction=? AND t.source=? "
                "AND t.origin='robot'",
                (res["symbol"], res["timeframe"], res["bar_time"], res["direction"], source))
    if dup:
        return 0
    d = res["direction"]
    pl = res.get("plan") or {"kind": "MARKET", "order_type": "BUY" if d > 0 else "SELL", "pending_bars": 0}
    risk = abs(res["entry"] - res["sl"])
    opened = opened_at or time.time()
    secs = tf_seconds(res["timeframe"])
    if pl["kind"] == "MARKET":   # fill now at the live price, keeping the planned risk distance
        entry = entry_price if entry_price is not None else res["entry"]
        status, fill_by = "open", None
    else:                        # limit / stop: wait for price to reach the planned entry
        entry, status, fill_by = res["entry"], "pending", opened + pl["pending_bars"] * secs
    sl = entry - d * risk
    tp = entry + d * config.REWARD_R * risk
    demo = {}
    if source == "live" and (res.get("is_signal") or res.get("source") == "chat"):
        from . import account   # robot signals on your chosen kinds of market + everything you ask it to analyse
        inst = resolve(res["symbol"])
        if res.get("source") == "chat" or account.trades_class(inst.asset_class):
            demo = {"demo": 1, **account.size_trade(inst.quote, risk)}
    tid = db.insert("trades", {**demo,
        "analysis_id": res["analysis_id"], "symbol": res["symbol"], "timeframe": res["timeframe"],
        "source": source, "direction": d, "entry": entry, "sl": sl, "tp": tp,
        "opened_at": opened, "expires_at": (fill_by or opened) + config.MAX_HOLD_BARS * secs,
        "status": status, "setup_key": res["setup_key"], "order_type": pl["order_type"], "fill_by": fill_by,
    })
    if tid and demo and res.get("source") != "chat":
        from . import trade_alerts
        trade_alerts.run(trade_alerts.opened, tid, res)
    return tid


def fill_index(direction: int, kind: str, entry: float, bars: pd.DataFrame) -> tuple[int, float] | None:
    """First bar where a pending order fills, and the fill price (gaps fill at the open for stops,
    and a gap through a limit fills at the better open price)."""
    for i, (_, b) in enumerate(bars.iterrows()):
        o, hi, lo = float(b["open"]), float(b["high"]), float(b["low"])
        if kind == "LIMIT":
            if (direction > 0 and lo <= entry) or (direction < 0 and hi >= entry):
                return i, (min(o, entry) if direction > 0 else max(o, entry))
        else:  # STOP
            if (direction > 0 and hi >= entry) or (direction < 0 and lo <= entry):
                return i, (max(o, entry) if direction > 0 else min(o, entry))
    return None


def simulate(direction: int, entry: float, sl: float, tp: float, bars: pd.DataFrame,
             expires_at: float | None = None, cost_r: float = 0.0, breakeven_r: float | None = None) -> dict | None:
    """Walk bars in order. If SL and TP are both inside one bar we assume the
    stop was hit first (conservative). Returns None while still open.
    Breakeven: once a bar closes the trade's best move at +breakeven_r (default config.BREAKEVEN_R),
    the stop moves to the entry from the NEXT bar on (never inside the same bar - conservative).
    A trade stopped there ends as 'breakeven' (0R minus costs)."""
    be_at = config.BREAKEVEN_R if breakeven_r is None else breakeven_r
    be_on = False
    risk = abs(entry - sl)
    if risk <= 0:
        return {"status": "lost", "exit_price": entry, "r_multiple": -1.0, "closed_at": time.time(),
                "mfe_r": 0.0, "mae_r": 0.0}
    mfe = mae = 0.0
    for ts, b in bars.iterrows():
        t = ts.timestamp()
        if expires_at is not None and t >= expires_at:
            break
        hi, lo = float(b["high"]), float(b["low"])
        fav = (hi - entry) if direction > 0 else (entry - lo)
        adv = (entry - lo) if direction > 0 else (hi - entry)
        mfe, mae = max(mfe, fav / risk), max(mae, adv / risk)
        stop = entry if be_on else sl
        hit_sl = lo <= stop if direction > 0 else hi >= stop
        hit_tp = hi >= tp if direction > 0 else lo <= tp
        if hit_sl:
            if be_on:
                return {"status": "breakeven", "exit_price": entry, "r_multiple": -cost_r, "closed_at": t,
                        "mfe_r": mfe, "mae_r": mae}
            return {"status": "lost", "exit_price": sl, "r_multiple": -1.0 - cost_r, "closed_at": t,
                    "mfe_r": mfe, "mae_r": mae}
        if hit_tp:
            r = abs(tp - entry) / risk
            return {"status": "won", "exit_price": tp, "r_multiple": r - cost_r, "closed_at": t,
                    "mfe_r": mfe, "mae_r": mae}
        if be_at and not be_on and mfe >= be_at:
            be_on = True
    if expires_at is not None and len(bars) and bars.index[-1].timestamp() >= expires_at - 1:
        done = bars[bars.index.map(lambda x: x.timestamp()) < expires_at]
        px = float(done["close"].iloc[-1]) if len(done) else entry
        r = direction * (px - entry) / risk - cost_r
        return {"status": "expired", "exit_price": px, "r_multiple": r, "closed_at": expires_at,
                "mfe_r": mfe, "mae_r": mae}
    return None


def reach_before_stop(direction: int, entry: float, sl: float, bars: pd.DataFrame,
                      expires_at: float | None = None) -> float:
    """Furthest the trade went in its favour (in R) before the stop was hit or time ran out,
    ignoring any take profit - used to measure how often TP1, TP2, TP3... would have been reached.
    A bar that touches the stop counts as stopped first (conservative)."""
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    best = 0.0
    for ts, b in bars.iterrows():
        if expires_at is not None and ts.timestamp() >= expires_at:
            break
        hi, lo = float(b["high"]), float(b["low"])
        if (lo <= sl) if direction > 0 else (hi >= sl):
            break
        best = max(best, ((hi - entry) if direction > 0 else (entry - lo)) / risk)
    return best


def _resolution_tf(trade_tf: str, age_s: float) -> str:
    for tf, limit in (("M1", 4500 * 60), ("M5", 4500 * 300), ("M15", 4500 * 900), ("H1", 4500 * 3600)):
        if age_s < limit and tf_seconds(tf) <= tf_seconds(trade_tf):
            return tf
    return trade_tf


def _kind(order_type: str | None, direction: int, entry: float, ref_price: float | None) -> str:
    if order_type and order_type.endswith("LIMIT"):
        return "LIMIT"
    if order_type and order_type.endswith("STOP"):
        return "STOP"
    if ref_price is None:
        return "LIMIT"
    # user orders without a stored type: below price = buy limit, above = buy stop (mirrored for sells)
    return "LIMIT" if (entry - ref_price) * direction < 0 else "STOP"


def activate_pending() -> None:
    """Pending limit/stop orders become open trades when price reaches the entry;
    unfilled ones are cancelled at their deadline."""
    now = time.time()
    for t in db.q("SELECT t.*, a.price ref_price FROM trades t LEFT JOIN analyses a ON a.id=t.analysis_id "
                  "WHERE t.status='pending'"):
        deadline = t.get("fill_by") or t["expires_at"]
        try:
            bars = market.candles(t["symbol"], "M1", min(int((now - t["opened_at"]) / 60) + 3, 5000))
        except Exception as e:
            log.warning("pending %s: %s", t["symbol"], e)
            if now >= deadline:
                db.update("trades", t["id"], {"status": "cancelled", "closed_at": now})
            continue
        ts = bars.index.map(lambda x: x.timestamp())
        bars = bars[(ts + 60 > t["opened_at"]) & (ts < deadline)]
        hit = fill_index(t["direction"], _kind(t.get("order_type"), t["direction"], t["entry"], t.get("ref_price")),
                         t["entry"], bars)
        if hit:
            i, px = hit
            fill_t = bars.index[i].timestamp()
            risk = abs(t["entry"] - t["sl"])
            upd = {"status": "open", "opened_at": fill_t, "entry": px,
                   "expires_at": fill_t + config.MAX_HOLD_BARS * tf_seconds(t["timeframe"])}
            if t.get("origin") != "user":  # robot orders keep their planned risk; your own SL/TP stay exactly as you set them
                upd.update(sl=px - t["direction"] * risk, tp=px + t["direction"] * config.REWARD_R * risk)
            db.update("trades", t["id"], upd)
            _alert("filled", {**t, **upd})
        elif now >= deadline:
            upd = missed_or_cancelled(t, bars, now)
            db.update("trades", t["id"], upd)
            _alert("closed", {**t, **upd})


def _alert(kind: str, t: dict) -> None:
    if not t.get("demo"):
        return
    from . import trade_alerts
    trade_alerts.run(getattr(trade_alerts, kind), t)


def missed_or_cancelled(t: dict, bars: pd.DataFrame, now: float) -> dict:
    """An unfilled order is 'missed' when price still went the predicted way by at least 1R
    (measured from the planned entry) - the direction was right but the entry never came."""
    d, risk = t["direction"], abs(t["entry"] - t["sl"])
    if risk <= 0 or bars is None or not len(bars):
        return {"status": "cancelled", "closed_at": now}
    # how far beyond the planned entry price travelled in the trade's direction (a buy stop that was
    # never triggered stays below its entry, so it scores 0 and is simply cancelled)
    fav = (float(bars["high"].max()) - t["entry"]) if d > 0 else (t["entry"] - float(bars["low"].min()))
    move = max(fav, 0.0) / risk
    if move >= 1.0:
        return {"status": "missed", "closed_at": now, "missed_r": round(move, 2),
                "lesson": (f"Missed: price moved {move:.1f}R the predicted way without reaching the "
                           f"{t.get('order_type', 'order').lower()} at {t['entry']:.5g}. Direction right, entry too far.")}
    return {"status": "cancelled", "closed_at": now, "missed_r": round(move, 2)}


def monitor_once() -> list[dict]:
    """Resolve every open live trade that has hit SL/TP or expired."""
    activate_pending()
    open_trades = db.q("SELECT * FROM trades WHERE status='open' AND source='live'")
    if not open_trades:
        return []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    now = time.time()
    for t in open_trades:
        groups[(t["symbol"], _resolution_tf(t["timeframe"], now - t["opened_at"]))].append(t)
    closed = []
    still_open: list[dict] = []
    bars_by_trade: dict = {}
    for (sym, rtf), trades in groups.items():
        oldest = min(t["opened_at"] for t in trades)
        need = int((now - oldest) / tf_seconds(rtf)) + 3
        try:
            bars = market.candles(sym, rtf, min(max(need, 5), 5000))
        except Exception as e:
            log.warning("monitor %s: %s", sym, e)
            continue
        inst = resolve(sym)
        for t in trades:
            # include the bar in which the trade was opened only from the open time onward
            sub = bars[bars.index.map(lambda x: x.timestamp() + tf_seconds(rtf)) > t["opened_at"]]
            cost_r = cost_in_price(inst, t["entry"]) / max(abs(t["entry"] - t["sl"]), 1e-12)
            out = simulate(t["direction"], t["entry"], t["sl"], t["tp"], sub, t["expires_at"], cost_r)
            if out is None and now >= t["expires_at"] and len(sub):
                px = float(sub["close"].iloc[-1])
                risk = abs(t["entry"] - t["sl"])
                out = {"status": "expired", "exit_price": px, "closed_at": now,
                       "r_multiple": t["direction"] * (px - t["entry"]) / risk - cost_r, "mfe_r": 0, "mae_r": 0}
            if out:
                db.update("trades", t["id"], out)
                closed.append({**t, **out})
                _alert("closed", {**t, **out})
            elif t.get("demo"):
                still_open.append(t)
                bars_by_trade[t["id"]] = sub
    if still_open:
        try:
            from . import trade_alerts
            trade_alerts.check_breakeven(still_open, bars_by_trade)
        except Exception as e:
            log.debug("breakeven alerts: %s", e)
    if closed:
        learning.refit_and_save()
        _write_lessons(closed)
    return closed


def _write_lessons(closed: list[dict]) -> None:
    """Short post-mortems for real signals (not shadow trades)."""
    if not config.ANTHROPIC_API_KEY:
        return
    from . import agent
    for t in closed:
        a = db.q1("SELECT * FROM analyses WHERE id=?", (t["analysis_id"],))
        if not a or not (a["is_signal"] or t.get("origin") == "user"):
            continue
        try:
            lesson = agent.post_mortem(db.decode(a), t)
            if lesson:
                db.update("trades", t["id"], {"lesson": lesson})
        except Exception as e:
            log.debug("lesson failed: %s", e)


def open_positions() -> list[dict]:
    rows = db.q("SELECT t.*, a.is_signal, a.probability FROM trades t JOIN analyses a ON a.id=t.analysis_id "
                "WHERE t.status IN ('open','pending') AND t.source='live' ORDER BY t.opened_at DESC")
    return rows
