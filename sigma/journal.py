"""Signal journal: every analysis is graded against what the market did next.

Call grade (for every signal and lean, filled or not)
  From the analysis candle's close, did price move 1R in the called direction before it moved
  1R against it?  right / wrong; neither within the holding window -> flat; both inside one
  candle -> unclear. R = the plan's stop distance.

Order outcomes (paper trades): won / lost / expired / cancelled / missed. 'missed' = an unfilled
limit or stop where price still went the predicted way by 1R or more.

What the robot learns from it
  - call accuracy by side, timeframe, market and order type (shown in the app and the chat)
  - fill statistics per order type; if limit entries keep missing good moves, the planner
    places them closer to price, and switches to market orders if they miss far too often
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict

from . import config, db
from .data import market
from .symbols import tf_seconds

log = logging.getLogger("sigma.journal")
_stats_cache: dict = {"t": 0.0, "v": None}


def grade_call(direction: int, ref: float, risk: float, bars, horizon_end: float) -> dict | None:
    """Returns the grade, or None while it is still undecided and the window is still open."""
    if risk <= 0:
        return {"call_result": "unclear", "call_move_r": 0.0, "call_adverse_r": 0.0}
    best = worst = 0.0
    last_t = None
    for ts, b in bars.iterrows():
        t = ts.timestamp()
        if t >= horizon_end:
            break
        last_t = t
        hi, lo = float(b["high"]), float(b["low"])
        fav = ((hi - ref) if direction > 0 else (ref - lo)) / risk
        adv = ((ref - lo) if direction > 0 else (hi - ref)) / risk
        best, worst = max(best, fav), max(worst, adv)
        if fav >= 1 and adv >= 1:
            return {"call_result": "unclear", "call_move_r": round(best, 2), "call_adverse_r": round(worst, 2)}
        if fav >= 1:
            return {"call_result": "right", "call_move_r": round(best, 2), "call_adverse_r": round(worst, 2)}
        if adv >= 1:
            return {"call_result": "wrong", "call_move_r": round(best, 2), "call_adverse_r": round(worst, 2)}
    if last_t is not None and time.time() >= horizon_end:
        return {"call_result": "flat", "call_move_r": round(best, 2), "call_adverse_r": round(worst, 2)}
    return None


def grade_pending(limit: int = 80) -> int:
    """Grade analyses whose outcome is now known. Runs in the background monitor loop."""
    now = time.time()
    rows = db.q("SELECT id, symbol, timeframe, direction, price, entry, sl, bar_time FROM analyses "
                "WHERE graded_at IS NULL AND source IN ('live','chat','session') AND bar_time IS NOT NULL "
                "AND created_at < ? ORDER BY created_at LIMIT ?", (now - 120, limit))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["symbol"], r["timeframe"])].append(r)
    done = 0
    for (sym, tf), items in groups.items():
        secs = tf_seconds(tf)
        oldest = min(r["bar_time"] for r in items)
        try:
            bars = market.candles(sym, tf, min(int((now - oldest) / secs) + 3, 5000))
        except Exception as e:
            log.debug("journal %s: %s", sym, e)
            continue
        for r in items:
            start = r["bar_time"] + secs          # first candle after the analysis candle
            sub = bars[bars.index.map(lambda x: x.timestamp()) >= start]
            g = grade_call(r["direction"], r["price"], abs((r["entry"] or 0) - (r["sl"] or 0)), sub,
                           start + config.MAX_HOLD_BARS * secs)
            if g:
                db.update("analyses", r["id"], {**g, "graded_at": now})
                done += 1
    return done


def _acc(rows) -> dict:
    right = sum(1 for r in rows if r["call_result"] == "right")
    wrong = sum(1 for r in rows if r["call_result"] == "wrong")
    flat = sum(1 for r in rows if r["call_result"] == "flat")
    n = right + wrong
    return {"graded": len(rows), "right": right, "wrong": wrong, "flat": flat,
            "accuracy": (right / n) if n else None}


def call_accuracy(signals_only: bool = False) -> dict:
    sql = ("SELECT a.direction, a.timeframe, a.symbol, a.call_result, a.is_signal, t.order_type FROM analyses a "
           "LEFT JOIN trades t ON t.analysis_id=a.id AND t.origin='robot' "
           "WHERE a.call_result IS NOT NULL AND a.source IN ('live','chat','session')")
    if signals_only:
        sql += " AND a.is_signal=1"
    rows = db.q(sql)
    by = lambda key: {k: _acc([r for r in rows if key(r) == k]) for k in sorted({key(r) for r in rows}, key=str)}  # noqa: E731
    return {"overall": _acc(rows),
            "by_side": by(lambda r: "buy" if r["direction"] > 0 else "sell"),
            "by_timeframe": by(lambda r: r["timeframe"]),
            "by_order": by(lambda r: (r["order_type"] or "none").split()[-1] if r["order_type"] else "none"),
            "by_market": dict(sorted(by(lambda r: r["symbol"]).items(),
                                     key=lambda kv: -(kv[1]["accuracy"] or 0))[:15])}


def order_stats(cache_seconds: float = 60) -> dict:
    """Fill and outcome statistics per order kind (MARKET / LIMIT / STOP) for the robot's live trades."""
    if _stats_cache["v"] is not None and time.time() - _stats_cache["t"] < cache_seconds:
        return _stats_cache["v"]
    rows = db.q("SELECT order_type, status, r_multiple FROM trades WHERE origin='robot' AND source='live'")
    out = {}
    for kind in ("MARKET", "LIMIT", "STOP"):
        rs = [r for r in rows if (r["order_type"] or "MARKET").split()[-1] in (kind,) or
              (kind == "MARKET" and (r["order_type"] or "MARKET") in ("BUY", "SELL", "MARKET"))]
        done = [r for r in rs if r["status"] not in ("open", "pending")]
        filled = [r for r in done if r["status"] in ("won", "lost", "expired")]
        missed = sum(1 for r in done if r["status"] == "missed")
        cancelled = sum(1 for r in done if r["status"] == "cancelled")
        out[kind] = {"placed": len(done), "filled": len(filled), "missed": missed, "cancelled": cancelled,
                     "fill_rate": (len(filled) / len(done)) if done else None,
                     "miss_rate": (missed / len(done)) if done else None,
                     "win_rate_when_filled": (sum(1 for r in filled if (r["r_multiple"] or 0) > 0) / len(filled))
                     if filled else None}
    _stats_cache.update(t=time.time(), v=out)
    return out


def recent(limit: int = 150) -> list[dict]:
    return db.q("SELECT a.id, a.created_at, a.symbol, a.timeframe, a.source, a.direction, a.is_signal, a.entry, a.sl, "
                "a.probability, a.call_result, a.call_move_r, a.call_adverse_r, t.order_type, t.status, "
                "t.r_multiple, t.missed_r, t.lesson FROM analyses a LEFT JOIN trades t ON t.analysis_id=a.id "
                "AND t.origin='robot' WHERE a.source IN ('live','chat','session') AND (a.is_signal=1 OR a.source='chat') "
                "ORDER BY a.created_at DESC LIMIT ?", (limit,))
