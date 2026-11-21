"""The robot's brain: combines technical, fundamental and news analysis through
the learned model into a scored, risk-defined trade idea, and journals it."""
from __future__ import annotations

import logging
import time

import pandas as pd

from . import config, db, learning
from .analysis import fundamental, sentiment, technical
from .data import market
from .symbols import HIGHER_TF, Instrument, resolve, tf_seconds

log = logging.getLogger("sigma.engine")

COST_FRACTION = {"forex": 0.00008, "metal": 0.00012, "crypto": 0.0006, "synthetic": 0.0002,
                 "index": 0.0001, "stock": 0.0005}


def cost_in_price(inst: Instrument, price: float) -> float:
    """Rough round-trip spread/commission, charged on every paper trade."""
    return price * COST_FRACTION.get(inst.asset_class, 0.0003)


def trade_levels(direction: int, price: float, atr: float, ctx: dict) -> tuple[float, float, float]:
    """Stop beyond the nearest swing when it sits 1-3 ATR away, else SL_ATR_MULT x ATR;
    target at REWARD_R x risk."""
    entry = price
    swing = ctx.get("last_swing_low") if direction > 0 else ctx.get("last_swing_high")
    sl = entry - direction * config.SL_ATR_MULT * atr
    if swing is not None:
        dist = (entry - swing) * direction
        if 1.0 * atr <= dist + 0.2 * atr <= 3.0 * atr:
            sl = swing - direction * 0.2 * atr
    risk = abs(entry - sl)
    tp = entry + direction * config.REWARD_R * risk
    return entry, sl, tp


def resample_htf(df: pd.DataFrame, tf: str) -> pd.DataFrame | None:
    htf = HIGHER_TF[tf]
    if htf == tf:
        return None
    rule = {"M15": "15min", "H1": "1h", "H4": "4h", "D1": "1D"}[htf]
    out = df.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum"}).dropna()
    return out.iloc[:-1]  # last HTF bar may be incomplete


def evaluate(symbol: str, tf: str, df: pd.DataFrame, htf: pd.DataFrame | None = None,
             fund: dict | None = None, news: dict | None = None, events: dict | None = None,
             model: learning.Model | None = None, stats: dict | None = None,
             now: float | None = None) -> dict:
    """Pure function: everything needed is passed in (used live and in backtests)."""
    inst = resolve(symbol)
    m = model or learning.model()
    tech = technical.analyze(df, htf)
    comps = dict(tech["components"])
    comps["fundamental"] = float((fund or {}).get("score", 0.0))
    comps["news"] = float((news or {}).get("score", 0.0))
    regime = tech["regime"]

    raw_long = m.prob(comps, regime, 1)
    raw_short = m.prob(comps, regime, -1)
    direction = 1 if raw_long >= raw_short else -1
    raw_p = max(raw_long, raw_short)
    p_long, p_short = m.calibrate(raw_long), m.calibrate(raw_short)
    p = m.calibrate(raw_p)

    entry, sl, tp = trade_levels(direction, tech["price"], tech["atr"], tech["context"])
    risk = abs(entry - sl) or 1e-12
    cost_r = cost_in_price(inst, entry) / risk
    exp_r = m.expectancy(p) - cost_r

    setup_key = f"{inst.name}|{tf}|{regime}|{'L' if direction > 0 else 'S'}"
    blocked = None
    if events and events.get("blocked"):
        ev = events["blocking_event"]
        blocked = f"high-impact event: {ev['currency']} {ev['title']}"
    g = learning.gated(setup_key, stats)
    if g:
        blocked = f"setup switched off by learning ({g['n']} signals, {g['shrunk_expectancy']:+.2f}R)"
    is_signal = (p >= config.MIN_PROBABILITY and exp_r >= config.MIN_EXPECTANCY_R and not blocked)

    raw = sum(learning.PRIOR_BASE[k] * comps[k] for k in learning.PRIOR_BASE) / sum(learning.PRIOR_BASE.values())
    notes = list(tech["notes"]) + list((fund or {}).get("notes", []))
    return {
        "symbol": inst.name, "display": inst.display, "asset_class": inst.asset_class, "timeframe": tf,
        "bar_time": df.index[-1].timestamp(), "created_at": now or time.time(),
        "price": tech["price"], "atr": tech["atr"], "regime": regime, "adx": tech["adx"],
        "direction": direction, "p_long": p_long, "p_short": p_short, "probability": p, "raw_probability": raw_p,
        "expectancy": exp_r, "cost_r": cost_r, "raw_score": raw,
        "entry": entry, "sl": sl, "tp": tp, "reward_r": config.REWARD_R,
        "is_signal": bool(is_signal), "blocked_reason": blocked, "setup_key": setup_key,
        "components": comps, "context": tech["context"], "notes": notes,
        "fundamental": fund, "news": news, "events": events,
        "provider": df.attrs.get("provider"),
    }


def analyze(symbol: str, tf: str = "H1", source: str = "chat", journal: bool = True,
            open_trade: bool | None = None) -> dict:
    """Live analysis with fresh data from every source. Journals the analysis and,
    unless disabled, opens a paper trade for it (shadow trade if it is not a signal)."""
    inst = resolve(symbol)
    tf = tf.upper()
    tf_seconds(tf)
    df = market.closed_candles(inst.name, tf, 400)
    htf_name = HIGHER_TF[tf]
    htf = None
    if htf_name != tf:
        try:
            htf = market.closed_candles(inst.name, htf_name, 300)
        except Exception as e:
            log.warning("HTF data failed: %s", e)
    fund = news = None
    try:
        fund = fundamental.analyze(inst)
    except Exception as e:
        log.warning("fundamentals failed: %s", e)
    try:
        news = sentiment.analyze(inst)
    except Exception as e:
        log.warning("news failed: %s", e)
    events = sentiment.event_risk(inst)
    res = evaluate(inst.name, tf, df, htf, fund, news, events)
    res["source"] = source
    if journal:
        res["analysis_id"] = record(res, source)
        if res["analysis_id"] and (config.AUTO_PAPER_TRADE if open_trade is None else open_trade):
            from . import paper
            res["trade_id"] = paper.open_trade(res, source="live", entry_price=_live_entry(inst, res))
    return res


def _live_entry(inst: Instrument, res: dict) -> float:
    try:
        return market.last_price(inst.name)
    except Exception:
        return res["price"]


def record(res: dict, source: str) -> int:
    return db.insert("analyses", {
        "created_at": res["created_at"], "bar_time": res["bar_time"], "symbol": res["symbol"],
        "timeframe": res["timeframe"], "source": source, "price": res["price"],
        "direction": res["direction"], "raw_score": res["raw_score"], "probability": res["probability"],
        "raw_probability": res["raw_probability"],
        "expectancy": res["expectancy"], "is_signal": int(res["is_signal"]), "entry": res["entry"],
        "sl": res["sl"], "tp": res["tp"], "regime": res["regime"], "setup_key": res["setup_key"],
        "components": res["components"],
        "features": {"context": {k: v for k, v in res["context"].items() if k not in ("levels",)},
                     "fundamental": (res.get("fundamental") or {}).get("parts"),
                     "headlines": [h["title"] for h in ((res.get("news") or {}).get("headlines") or [])[:5]]},
        "notes": res["notes"], "blocked_reason": res["blocked_reason"],
    })
