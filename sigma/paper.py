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
    entry = entry_price if entry_price is not None else res["entry"]
    risk = abs(res["entry"] - res["sl"])
    sl = entry - d * risk
    tp = entry + d * config.REWARD_R * risk
    opened = opened_at or time.time()
    return db.insert("trades", {
        "analysis_id": res["analysis_id"], "symbol": res["symbol"], "timeframe": res["timeframe"],
        "source": source, "direction": d, "entry": entry, "sl": sl, "tp": tp,
        "opened_at": opened, "expires_at": opened + config.MAX_HOLD_BARS * tf_seconds(res["timeframe"]),
        "status": "open", "setup_key": res["setup_key"],
    })


def simulate(direction: int, entry: float, sl: float, tp: float, bars: pd.DataFrame,
             expires_at: float | None = None, cost_r: float = 0.0) -> dict | None:
    """Walk bars in order. If SL and TP are both inside one bar we assume the
    stop was hit first (conservative). Returns None while still open."""
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
        hit_sl = lo <= sl if direction > 0 else hi >= sl
        hit_tp = hi >= tp if direction > 0 else lo <= tp
        if hit_sl:
            return {"status": "lost", "exit_price": sl, "r_multiple": -1.0 - cost_r, "closed_at": t,
                    "mfe_r": mfe, "mae_r": mae}
        if hit_tp:
            r = abs(tp - entry) / risk
            return {"status": "won", "exit_price": tp, "r_multiple": r - cost_r, "closed_at": t,
                    "mfe_r": mfe, "mae_r": mae}
    if expires_at is not None and len(bars) and bars.index[-1].timestamp() >= expires_at - 1:
        done = bars[bars.index.map(lambda x: x.timestamp()) < expires_at]
        px = float(done["close"].iloc[-1]) if len(done) else entry
        r = direction * (px - entry) / risk - cost_r
        return {"status": "expired", "exit_price": px, "r_multiple": r, "closed_at": expires_at,
                "mfe_r": mfe, "mae_r": mae}
    return None


def _resolution_tf(trade_tf: str, age_s: float) -> str:
    for tf, limit in (("M1", 4500 * 60), ("M5", 4500 * 300), ("M15", 4500 * 900), ("H1", 4500 * 3600)):
        if age_s < limit and tf_seconds(tf) <= tf_seconds(trade_tf):
            return tf
    return trade_tf


def activate_pending() -> None:
    """Pending user orders become open when price trades through the entry."""
    now = time.time()
    for t in db.q("SELECT * FROM trades WHERE status='pending'"):
        if now >= t["expires_at"]:
            db.update("trades", t["id"], {"status": "cancelled", "closed_at": now})
            continue
        try:
            bars = market.candles(t["symbol"], "M1", min(int((now - t["opened_at"]) / 60) + 3, 5000))
        except Exception as e:
            log.warning("pending %s: %s", t["symbol"], e)
            continue
        bars = bars[bars.index.map(lambda x: x.timestamp() + 60) > t["opened_at"]]
        hit = bars[(bars["low"] <= t["entry"]) & (bars["high"] >= t["entry"])]
        if len(hit):
            db.update("trades", t["id"], {"status": "open", "opened_at": hit.index[0].timestamp()})


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
