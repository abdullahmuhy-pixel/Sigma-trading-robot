"""Review a trade the user is considering: grade it against the robot's live
analysis, then track it as a paper trade so the user sees how it would have gone."""
from __future__ import annotations

import math
import time

from . import config, db, engine, learning
from .data import market
from .symbols import resolve, tf_seconds


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def adjusted_probability(p_at_default_rr: float, rr: float) -> float:
    """The model's P(win) is for a target at REWARD_R. For another reward:risk the
    no-edge baseline is 1/(1+rr); keep the model's edge (in log-odds) and move the baseline."""
    edge = _logit(p_at_default_rr) - _logit(1 / (1 + config.REWARD_R))
    return 1 / (1 + math.exp(-(_logit(1 / (1 + rr)) + edge)))


def review_trade(symbol: str, timeframe: str, direction: str, entry: float, stop_loss: float,
                 take_profit: float | None = None, track: bool = True, note: str = "") -> dict:
    d = 1 if direction.lower() in ("buy", "long", "1") else -1
    issues: list[str] = []
    if (entry - stop_loss) * d <= 0:
        return {"error": f"For a {'buy' if d > 0 else 'sell'} the stop loss must be "
                         f"{'below' if d > 0 else 'above'} the entry."}
    risk = abs(entry - stop_loss)
    if take_profit is None:
        take_profit = entry + d * config.REWARD_R * risk
        issues.append(f"No take profit given; assumed {config.REWARD_R:g}R.")
    if (take_profit - entry) * d <= 0:
        return {"error": "Take profit is on the wrong side of the entry."}
    rr = abs(take_profit - entry) / risk

    res = engine.analyze(symbol, timeframe, source="chat")
    inst = resolve(symbol)
    m = learning.model()
    atr = res["atr"]
    price = res["price"]
    try:
        price = market.last_price(inst.name)
    except Exception:
        pass

    raw = m.prob(res["components"], res["regime"], d)
    p_model = m.calibrate(raw)
    p = adjusted_probability(p_model, rr)
    cost_r = engine.cost_in_price(inst, entry) / risk
    exp_r = p * rr - (1 - p) - cost_r

    agrees = res["direction"] == d
    sl_atr = risk / atr if atr else None
    if sl_atr is not None and sl_atr < 0.7:
        issues.append(f"Stop is only {sl_atr:.2f} ATR away - normal noise can hit it.")
    if sl_atr is not None and sl_atr > 4:
        issues.append(f"Stop is {sl_atr:.1f} ATR away - very wide; size down.")
    if rr < 1:
        issues.append(f"Reward:risk is {rr:.2f} - you need a win rate above {1 / (1 + rr):.0%} to break even.")
    swing = res["context"].get("last_swing_low" if d > 0 else "last_swing_high")
    if swing is not None and (stop_loss - swing) * d > 0 and abs(stop_loss - swing) < 1.5 * atr:
        issues.append(f"Stop sits inside the last swing ({swing:.5g}); a stop just beyond it is usually safer.")
    blockers = [lv["price"] for lv in res["context"].get("levels", [])
                if min(entry, take_profit) < lv["price"] < max(entry, take_profit)
                and abs(lv["price"] - entry) > 0.3 * atr]
    if blockers:
        issues.append("Levels between entry and target that may stall price: "
                      + ", ".join(f"{b:.5g}" for b in sorted(blockers, key=lambda b: abs(b - entry))[:4]))
    if not agrees:
        issues.append(f"The robot's own read favours the {'buy' if res['direction'] > 0 else 'sell'} side here.")
    ev = res.get("events") or {}
    high_soon = [e for e in ev.get("upcoming", []) if e["impact"].lower() == "high" and e["time"] - time.time() < 6 * 3600]
    if high_soon:
        issues.append("High-impact news within 6h: " + "; ".join(f"{e['currency']} {e['title']}" for e in high_soon[:3]))
    dist_atr = abs(price - entry) / atr if atr else 0
    kind = "MARKET" if dist_atr <= 0.25 else ("LIMIT" if (entry - price) * d < 0 else "STOP")
    side = "BUY" if d > 0 else "SELL"
    order = side if kind == "MARKET" else f"{side} {kind}"

    if exp_r >= config.MIN_EXPECTANCY_R and agrees and not high_soon:
        verdict = "GOOD - the numbers support this trade"
    elif exp_r > 0:
        verdict = "CAUTION - small or uncertain edge; see the issues"
    else:
        verdict = "AVOID - negative expected value on current evidence"

    trade_id = None
    if track and res.get("analysis_id"):
        now = time.time()
        trade_id = db.insert("trades", {
            "analysis_id": res["analysis_id"], "symbol": inst.name, "timeframe": res["timeframe"],
            "source": "live", "origin": "user", "direction": d, "entry": entry, "sl": stop_loss,
            "tp": take_profit, "opened_at": now,
            "expires_at": now + config.MAX_HOLD_BARS * tf_seconds(res["timeframe"]),
            "status": "open" if kind == "MARKET" else "pending", "setup_key": res["setup_key"], "note": note,
            "order_type": order, "fill_by": None if kind == "MARKET" else now + config.MAX_HOLD_BARS * tf_seconds(res["timeframe"]),
        })

    return {
        "symbol": res["display"], "timeframe": res["timeframe"], "your_trade": {
            "direction": "BUY" if d > 0 else "SELL", "entry": entry, "stop_loss": stop_loss,
            "take_profit": take_profit, "reward_risk": round(rr, 2), "order_type": order,
            "stop_in_atr": None if sl_atr is None else round(sl_atr, 2)},
        "current_price": price, "verdict": verdict,
        "p_win_estimate": round(p, 3), "breakeven_win_rate": round(1 / (1 + rr), 3),
        "expected_R_after_costs": round(exp_r, 3),
        "robot_view": {"direction": "BUY" if res["direction"] > 0 else "SELL", "is_signal": res["is_signal"],
                       "p_win": round(res["probability"], 3), "entry": res["entry"], "sl": res["sl"], "tp": res["tp"],
                       "regime": res["regime"]},
        "component_scores_for_your_direction": {k: round(v * d, 2) for k, v in res["components"].items()},
        "issues": issues, "notes": res["notes"],
        "tracking": (f"Tracked as paper trade #{trade_id} ({'waiting for price to reach entry' if kind != 'MARKET' else 'live now'}); "
                     "I'll grade it when it closes.") if trade_id else "Not tracked.",
        "your_track_record": learning.performance("live", origin="user"),
    }
