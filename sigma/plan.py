"""Turns an analysis into a concrete order and a plain-language explanation.

Order types
  BUY / SELL               enter now at market
  BUY LIMIT / SELL LIMIT   wait for a pullback to a better price (support/EMA for buys, resistance/EMA for sells)
  BUY STOP / SELL STOP     enter only if price breaks through a nearby level (breakout confirmation)

Rules (symmetric for buys and sells)
  1. Breakout: a resistance (buy) / support (sell) sits within 0.6 ATR, the market is not ranging,
     and structure or trend agrees  ->  STOP order 0.1 ATR beyond the level.
  2. Pullback: price has already run more than 1 ATR from EMA20 in the trade direction, or is
     stretched (overbought for a buy / oversold for a sell), and a pullback level sits 0.3-1.5 ATR
     away  ->  LIMIT order at that level.
  3. Otherwise  ->  market order.
"""
from __future__ import annotations

from . import config
from .symbols import HIGHER_TF

PENDING_BARS = 6   # a pending order that has not filled after this many candles is cancelled

LABELS = {"trend": "Trend", "htf_trend": "Higher-timeframe trend", "momentum": "Momentum",
          "structure": "Market structure", "mean_reversion": "Overbought/oversold", "levels": "Support/resistance room",
          "pattern": "Candle pattern", "divergence": "RSI divergence", "fundamental": "Fundamentals", "news": "News",
          "crowd": "Crowd positioning"}


def _fmt(x):
    if x is None:
        return "-"
    return f"{x:,.5f}" if abs(x) < 10 else f"{x:,.2f}"


def stop_loss(direction: int, entry: float, atr: float, ctx: dict) -> float:
    swing = ctx.get("last_swing_low") if direction > 0 else ctx.get("last_swing_high")
    sl = entry - direction * config.SL_ATR_MULT * atr
    if swing is not None:
        dist = (entry - swing) * direction
        if 1.0 * atr <= dist + 0.2 * atr <= 3.0 * atr:
            sl = swing - direction * 0.2 * atr
    return sl


LIMIT_MIN_SAMPLES = 15


def order(direction: int, price: float, atr: float, ctx: dict, comps: dict, regime: str,
          fill_stats: dict | None = None) -> dict:
    d = direction
    ema20 = ctx.get("ema20")
    sup, res = ctx.get("nearest_support"), ctx.get("nearest_resistance")
    side = "BUY" if d > 0 else "SELL"
    # the level in front of the trade (breakout) and behind it (pullback)
    ahead = res if d > 0 else sup
    behind = [lv for lv in ((sup if d > 0 else res), ema20) if lv is not None and (price - lv) * d > 0]
    behind = [lv for lv in behind if 0.3 * atr <= abs(price - lv) <= 1.5 * atr]
    pull = min(behind, key=lambda lv: abs(price - lv)) if behind else None
    extension = ((price - ema20) * d / atr) if ema20 and atr else 0.0
    stretched_with_trade = comps.get("mean_reversion", 0) * d < -0.4
    agrees = comps.get("structure", 0) * d > 0 or comps.get("trend", 0) * d > 0.3

    if ahead is not None and 0 < (ahead - price) * d <= 0.6 * atr and regime != "ranging" and agrees:
        entry = ahead + d * 0.1 * atr
        kind, why = "STOP", (f"price is pressing against {'resistance' if d > 0 else 'support'} at {_fmt(ahead)}; "
                             f"the order only fills if it breaks through, which confirms the move")
        setup = "Breakout"
    elif pull is not None and (extension > 1.0 or stretched_with_trade) and _limit_mode(fill_stats) != "market":
        lvl = "EMA20" if ema20 is not None and abs(pull - ema20) < 1e-12 else ("support" if d > 0 else "resistance")
        if _limit_mode(fill_stats) == "shallow":   # learned: deep limits kept missing good moves
            entry = price + 0.5 * (pull - price)
            kind, why = "LIMIT", (f"price has run {extension:.1f} ATR from its average; the journal shows deep limit "
                                  f"entries often missed the move, so this limit sits halfway to {lvl} ({_fmt(pull)})")
        else:
            entry = pull + d * 0.05 * atr
            kind, why = "LIMIT", (f"price has already run {extension:.1f} ATR from its average; waiting for a pullback "
                                  f"to {lvl} at {_fmt(pull)} gives a better price and a tighter stop")
        setup = "Trend pullback" if comps.get("trend", 0) * d > 0 else "Pullback entry"
    else:
        entry = price
        kind, why = "MARKET", "price is at a reasonable entry now, with no nearby level to wait for"
        setup = _setup_type(d, comps)
    sl = stop_loss(d, entry, atr, ctx)
    risk = abs(entry - sl)
    tp = entry + d * config.REWARD_R * risk
    return {"order_type": side if kind == "MARKET" else f"{side} {kind}", "kind": kind, "side": side,
            "entry": entry, "sl": sl, "tp": tp, "risk": risk, "order_reason": why, "setup_type": setup,
            "pending_bars": PENDING_BARS if kind != "MARKET" else 0}


def adjust_to_live(order: dict, live: float, atr: float, ctx: dict) -> dict:
    """Re-check the planned order against the price NOW. The plan is built on the last closed candle;
    by the time a deep analysis finishes price may have moved, so the order type must follow it:
      - market order, price already moved in the trade's favour by > 0.25 ATR  -> LIMIT at the planned
        entry (wait for price to come back rather than chase it)
      - market order, price moved against the trade                             -> still MARKET, at the
        live price (a better price than planned), stop and targets rebuilt from there
      - limit order, price already through the limit                            -> fills now: MARKET
      - stop order, price already through the trigger                           -> triggered: MARKET
    """
    d = 1 if order["side"] == "BUY" else -1
    planned = order["entry"]
    gap = (live - planned) * d / atr if atr else 0.0     # >0: price already moved the trade's way
    o = dict(order)
    o["live_price"] = live
    note = None
    if o["kind"] == "MARKET":
        if gap > 0.25:
            o.update(kind="LIMIT", order_type=f"{o['side']} LIMIT", pending_bars=PENDING_BARS,
                     order_reason=(f"price has already moved {gap:.1f} ATR in the trade's direction since the analysis "
                                   f"candle closed ({_fmt(planned)} -> {_fmt(live)}); chasing it would cut the reward, "
                                   f"so the order waits for a retrace to {_fmt(planned)}"))
            note = "converted market -> limit"
        elif gap < -0.05:
            o["entry"] = live
            o["sl"] = stop_loss(d, live, atr, ctx)
            o["risk"] = abs(live - o["sl"])
            o["tp"] = live + d * config.REWARD_R * o["risk"]
            note = "market entry moved to the live price"
    elif o["kind"] == "LIMIT" and (live - planned) * d <= 0:
        o.update(kind="MARKET", order_type=o["side"], pending_bars=0, entry=live,
                 order_reason=f"price is already at or beyond the planned limit {_fmt(planned)}, so it can be taken now")
        o["sl"] = stop_loss(d, live, atr, ctx)
        o["risk"] = abs(live - o["sl"])
        o["tp"] = live + d * config.REWARD_R * o["risk"]
        note = "limit already reached -> market"
    elif o["kind"] == "STOP" and (live - planned) * d >= 0:
        o.update(kind="MARKET", order_type=o["side"], pending_bars=0, entry=live,
                 order_reason=f"the breakout level {_fmt(planned)} has already broken, so the trade is live now")
        o["sl"] = stop_loss(d, live, atr, ctx)
        o["risk"] = abs(live - o["sl"])
        o["tp"] = live + d * config.REWARD_R * o["risk"]
        note = "stop already triggered -> market"
    o["live_adjustment"] = note
    return o


def tp_probability(p_at_main: float, r: float) -> float:
    """Chance of reaching +r R before the stop, from the calibrated chance of reaching the main
    target (REWARD_R). With no edge, the chance of hitting +r before -1 is 1/(1+r) (a fair random
    walk); the robot's measured edge (in log-odds) is carried over to every target."""
    import math
    lg = lambda x: math.log(min(max(x, 1e-6), 1 - 1e-6) / (1 - min(max(x, 1e-6), 1 - 1e-6)))  # noqa: E731
    edge = lg(p_at_main) - lg(1 / (1 + config.REWARD_R))
    return 1 / (1 + math.exp(-(lg(1 / (1 + r)) + edge)))


def add_take_profits(order: dict, p_main: float, cost_r: float = 0.0) -> dict:
    """TP1..TPn at config.TP_LEVELS multiples of risk, each with its chance of being reached
    before the stop loss, and the expected R of closing the whole trade there."""
    d = 1 if order["side"] == "BUY" else -1
    tps = []
    for i, r in enumerate(sorted(config.TP_LEVELS), 1):
        pr = tp_probability(p_main, r)
        tps.append({"name": f"TP{i}", "r": r, "price": order["entry"] + d * r * order["risk"],
                    "p_reach": pr, "expected_r_if_closed_here": pr * r - (1 - pr) - cost_r,
                    "main": abs(r - config.REWARD_R) < 1e-9})
    order["take_profits"] = tps
    return tps


def _limit_mode(fill_stats: dict | None) -> str:
    """normal / shallow / market, learned from how often limit orders missed moves."""
    st = (fill_stats or {}).get("LIMIT") or {}
    if (st.get("placed") or 0) < LIMIT_MIN_SAMPLES or st.get("miss_rate") is None:
        return "normal"
    if st["miss_rate"] >= 0.55:
        return "market"
    if st["miss_rate"] >= 0.35:
        return "shallow"
    return "normal"


def _setup_type(d: int, comps: dict) -> str:
    c = {k: comps.get(k, 0) * d for k in comps}
    if c.get("divergence", 0) > 0 and c.get("trend", 0) < 0:
        return "Divergence reversal"
    if c.get("mean_reversion", 0) > 0.4 and c.get("trend", 0) <= 0.2:
        return "Mean-reversion reversal"
    if c.get("levels", 0) > 0.4 and c.get("pattern", 0) > 0:
        return "Support/resistance bounce"
    if c.get("trend", 0) > 0.3 and c.get("htf_trend", 0) > 0:
        return "Trend continuation"
    return "Mixed-signal setup"


def _component_text(k: str, v: float, ctx: dict, tf: str) -> str:
    if k == "trend":
        e20, e50 = ctx.get("ema20"), ctx.get("ema50")
        st = {1: "bullish", -1: "bearish"}.get(ctx.get("supertrend"), "neutral")
        rel = "above" if (e20 or 0) > (e50 or 0) else "below"
        return f"EMA20 ({_fmt(e20)}) is {rel} EMA50 ({_fmt(e50)}); SuperTrend is {st}"
    if k == "htf_trend":
        word = "up" if v > 0.2 else "down" if v < -0.2 else "sideways"
        return f"the {HIGHER_TF.get(tf, 'higher')} chart is trending {word}"
    if k == "momentum":
        hist = ctx.get("macd_hist") or 0
        return f"RSI {ctx.get('rsi', 50):.0f}, MACD histogram {'positive' if hist > 0 else 'negative'}"
    if k == "structure":
        return ctx.get("structure", "")
    if k == "mean_reversion":
        b, st = ctx.get("bb_pct_b", 0.5), ctx.get("stoch_k", 50)
        state = "overbought" if v < -0.3 else "oversold" if v > 0.3 else "neither overbought nor oversold"
        return f"price is {state} (Bollinger %B {b:.2f}, Stochastic {st:.0f})"
    if k == "levels":
        return f"nearest support {_fmt(ctx.get('nearest_support'))}, resistance {_fmt(ctx.get('nearest_resistance'))}"
    if k == "pattern":
        pats = [f"{'bullish' if s > 0 else 'bearish' if s < 0 else ''} {p.replace('_', ' ')}".strip()
                for p, s in (ctx.get("patterns") or {}).items()]
        return ", ".join(pats) or "no clear candle pattern"
    if k == "divergence":
        return "bullish RSI divergence (price lower low, RSI higher low)" if v > 0 else \
               "bearish RSI divergence (price higher high, RSI lower high)" if v < 0 else "none"
    if k == "fundamental":
        return "macro data " + ("favours" if v > 0.1 else "weighs on" if v < -0.1 else "is neutral for") + " the instrument"
    if k == "news":
        return "recent headlines are " + ("positive" if v > 0.1 else "negative" if v < -0.1 else "mixed")
    if k == "crowd":
        return ("most retail traders are short - contrarian bullish" if v > 0.1 else
                "most retail traders are long - contrarian bearish" if v < -0.1 else "retail traders are evenly split")
    return ""


def explain(res: dict, weights: dict) -> dict:
    """Ranks every factor by how much it pushed the robot toward (or against) its chosen side."""
    d, regime, tf, ctx, comps = res["direction"], res["regime"], res["timeframe"], res["context"], res["components"]
    contrib = []
    for k, v in comps.items():
        w = weights.get(k, 0) + weights.get(f"{k}@{regime}", 0)
        c = w * v * d
        if abs(v) < 0.05 or abs(c) < 0.01:
            continue
        contrib.append({"factor": LABELS.get(k, k), "key": k, "score": round(v, 2), "impact": round(c, 3),
                        "detail": _component_text(k, v, ctx, tf)})
    contrib.sort(key=lambda x: -abs(x["impact"]))
    pro = [c for c in contrib if c["impact"] > 0][:5]
    con = [c for c in contrib if c["impact"] < 0][:4]
    plan = res["plan"]
    side = "buy" if d > 0 else "sell"
    lead = "; ".join(f"{c['factor'].lower()} ({c['detail']})" for c in pro[:3]) or "a slight edge in the combined evidence"
    against = "; ".join(f"{c['factor'].lower()}" for c in con[:2])
    summary = (f"{plan['setup_type']}: the robot leans {side.upper()} because {lead}."
               + (f" Working against it: {against}." if against else "")
               + f" Order: {plan['order_type']} at {_fmt(plan['entry'])} - {plan['order_reason']}.")
    invalid = (f"The idea is wrong if price trades {'below' if d > 0 else 'above'} {_fmt(plan['sl'])} (stop loss)."
               + (f" A pending order not filled within {plan['pending_bars']} {tf} candles is cancelled."
                  if plan["pending_bars"] else ""))
    return {"summary": summary, "for": pro, "against": con, "invalidation": invalid}
