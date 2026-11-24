"""The robot's brain: combines technical, fundamental and news analysis through
the learned model into a scored, risk-defined trade idea, and journals it."""
from __future__ import annotations

import logging
import time

import pandas as pd

from . import config, db, learning, plan
from .analysis import fundamental, sentiment, technical
from .data import market
from .symbols import HIGHER_TF, Instrument, resolve, tf_seconds

log = logging.getLogger("sigma.engine")

COST_FRACTION = {"forex": 0.00008, "metal": 0.00012, "crypto": 0.0006, "synthetic": 0.0002,
                 "index": 0.0001, "stock": 0.0005, "commodity": 0.0004, "basket": 0.0002}


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

    try:
        from .journal import order_stats
        fill_stats = order_stats()
    except Exception:
        fill_stats = None
    # Open-minded: build the COMPLETE case for both a buy and a sell - order plan, costs,
    # filters, take-profit odds - and only then let the facts pick the side.
    event_block = None
    if events and events.get("blocked"):
        ev = events["blocking_event"]
        event_block = f"high-impact event: {ev['currency']} {ev['title']}"
    sides = {}
    for d in (1, -1):
        raw_d = m.prob(comps, regime, d)
        p_d = m.calibrate(raw_d)
        order_d = plan.order(d, tech["price"], tech["atr"], tech["context"], comps, regime, fill_stats)
        risk_d = abs(order_d["entry"] - order_d["sl"]) or 1e-12
        cost_d = cost_in_price(inst, order_d["entry"]) / risk_d
        exp_d = m.expectancy(p_d) - cost_d
        key_d = f"{inst.name}|{tf}|{regime}|{'L' if d > 0 else 'S'}"
        block_d = event_block
        g = learning.gated(key_d, stats)
        if g:
            block_d = f"setup switched off by learning ({g['n']} signals, {g['shrunk_expectancy']:+.2f}R)"
        plan.add_take_profits(order_d, p_d, cost_d)
        sides[d] = {"direction": d, "raw_p": raw_d, "p": p_d, "order": order_d, "expectancy": exp_d,
                    "cost_r": cost_d, "setup_key": key_d, "blocked": block_d,
                    "qualifies": p_d >= config.MIN_PROBABILITY and exp_d >= config.MIN_EXPECTANCY_R and not block_d}
    balanced = abs(sides[1]["raw_p"] - sides[-1]["raw_p"]) < 1e-6
    # a side that passes every check beats one that does not; then the better expected value wins
    best = max(sides.values(), key=lambda x: (x["qualifies"], round(x["expectancy"], 9)))
    if not sides[1]["qualifies"] and not sides[-1]["qualifies"] and abs(sides[1]["expectancy"] - sides[-1]["expectancy"]) < 1e-9:
        best = max(sides.values(), key=lambda x: x["raw_p"])
    direction = best["direction"]
    order, p, raw_p, exp_r, cost_r = best["order"], best["p"], best["raw_p"], best["expectancy"], best["cost_r"]
    entry, sl, tp = order["entry"], order["sl"], order["tp"]
    setup_key, blocked = best["setup_key"], best["blocked"]
    if balanced:   # evidence perfectly balanced: no side is preferred
        blocked = blocked or "no directional edge: buy and sell evidence are equal"
    is_signal = best["qualifies"] and not balanced
    p_long, p_short = sides[1]["p"], sides[-1]["p"]
    both = {("buy" if d > 0 else "sell"): {
        "order_type": x["order"]["order_type"], "entry": x["order"]["entry"], "sl": x["order"]["sl"],
        "take_profits": x["order"]["take_profits"], "p_win": x["p"], "expected_r": x["expectancy"],
        "setup_type": x["order"]["setup_type"], "qualifies": x["qualifies"], "blocked": x["blocked"]}
        for d, x in sides.items()}

    raw = sum(learning.PRIOR_BASE[k] * comps[k] for k in learning.PRIOR_BASE) / sum(learning.PRIOR_BASE.values())
    notes = list(tech["notes"]) + list((fund or {}).get("notes", []))
    res = {
        "symbol": inst.name, "display": inst.display, "asset_class": inst.asset_class, "timeframe": tf,
        "bar_time": df.index[-1].timestamp(), "created_at": now or time.time(),
        "price": tech["price"], "atr": tech["atr"], "regime": regime, "adx": tech["adx"],
        "direction": direction, "p_long": p_long, "p_short": p_short, "probability": p, "raw_probability": raw_p,
        "expectancy": exp_r, "cost_r": cost_r, "raw_score": raw,
        "entry": entry, "sl": sl, "tp": tp, "reward_r": config.REWARD_R,
        "is_signal": bool(is_signal), "blocked_reason": blocked, "setup_key": setup_key,
        "components": comps, "context": tech["context"], "notes": notes,
        "fundamental": fund, "news": news, "events": events,
        "provider": df.attrs.get("provider"), "plan": order, "both_sides": both,
        "take_profits": order["take_profits"], "_sides": sides,
    }
    res["explanation"] = plan.explain(res, m.weights())
    return res


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
    try:
        apply_live(res, market.last_price(inst.name))
    except Exception as e:
        log.debug("live price unavailable: %s", e)
    closed = market_closed(res["bar_time"], tf)
    hours = _hours(inst.name)
    res["market_hours"] = hours
    if hours.get("open") is False and not closed:
        closed = 1
    if closed:
        res["market_closed"] = True
        res["is_signal"] = False
        res["blocked_reason"] = (f"market closed (Deriv trading hours){_opens(hours)}" if hours.get("open") is False
                                 else f"market closed - last candle closed {closed / 3600:.1f} h ago")
        if source == "live":
            return res          # scheduled scans skip closed markets entirely
        open_trade = False      # chat analysis is still shown and journalled, never traded
    if journal:
        res["analysis_id"] = record(res, source)
        if res["analysis_id"] and (config.AUTO_PAPER_TRADE if open_trade is None else open_trade):
            from . import paper
            res["trade_id"] = paper.open_trade(res, source="live", entry_price=_live_entry(inst, res))
    return res


def _hours(name: str) -> dict:
    try:
        from .data.deriv_info import market_hours
        return market_hours(name)
    except Exception as e:
        log.debug("market hours: %s", e)
        return {"open": None}


def _opens(h: dict) -> str:
    m = h.get("opens_in_min")
    return f" - opens in {m / 60:.1f} h" if m else ""


def is_closed_now(name: str) -> bool:
    """True only when Deriv's official hours say the market is shut right now (unknown = not closed)."""
    return _hours(name).get("open") is False


def apply_live(res: dict, live: float | None) -> dict:
    """Make every order in an analysis match the live price (see plan.adjust_to_live)."""
    if live is None or not res.get("_sides"):
        return res
    for d, sd in res["_sides"].items():
        o = plan.adjust_to_live(sd["order"], live, res["atr"], res["context"])
        if o is not sd["order"]:
            plan.add_take_profits(o, sd["p"], sd["cost_r"])
            sd["order"] = o
        name = "buy" if d > 0 else "sell"
        if res.get("both_sides", {}).get(name):
            res["both_sides"][name].update(order_type=o["order_type"], entry=o["entry"], sl=o["sl"],
                                           take_profits=o["take_profits"])
    o = res["_sides"][res["direction"]]["order"]
    res.update(plan=o, entry=o["entry"], sl=o["sl"], tp=o["tp"], take_profits=o["take_profits"], live_price=live)
    res["explanation"] = plan.explain(res, learning.model().weights())
    return res


def market_closed(bar_time: float, tf: str, now: float | None = None) -> float:
    """Seconds since the last candle closed, if that is long enough to mean the market
    is shut (weekend, holiday); otherwise 0."""
    age = (now or time.time()) - (bar_time + tf_seconds(tf))
    limit = max(3 * tf_seconds(tf), 3600) if tf != "D1" else 3 * 86400
    return age if age > limit else 0.0


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
        "components": {k: round(v, 4) for k, v in res["components"].items()},
        # the full snapshot is kept for signals and for analyses you asked for; scanner leans and
        # backtests keep only their scores (enough for learning) so the database stays small
        "features": ({"context": {k: v for k, v in res["context"].items() if k not in ("levels",)},
                      "fundamental": (res.get("fundamental") or {}).get("parts"),
                      "headlines": [h["title"] for h in ((res.get("news") or {}).get("headlines") or [])[:5]]}
                     if (res.get("is_signal") or source == "chat") and source != "backtest" else {}),
        "notes": res["notes"], "blocked_reason": res["blocked_reason"],
    })
