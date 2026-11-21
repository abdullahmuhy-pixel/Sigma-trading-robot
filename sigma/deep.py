"""Deep analysis (about 1-2 minutes): slower, more thorough than the instant scan.

1. Top-down: the same engine on D1, H4, H1 and the chosen timeframe (plus M15 for entry timing),
   and how many higher timeframes agree with the trade.
2. Look-alike history: replays up to 2,000 past candles of THIS instrument through the engine and
   finds the setups most similar to now (same direction, closest indicator readings). Their real
   outcomes are an independent check on the model.
3. Blended verdict: the model's probability is combined with the look-alike win rate (more
   look-alikes = more weight), and a signal also needs the higher timeframes not to oppose it.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from . import backtest, config, engine, learning, paper, plan
from .data import market
from .symbols import TIMEFRAMES, resolve

log = logging.getLogger("sigma.deep")
TECH = ["trend", "htf_trend", "momentum", "structure", "mean_reversion", "levels", "pattern", "divergence"]
MODEL_WEIGHT = 30.0     # the model's view counts like this many look-alike trades
_replay_cache: dict[str, tuple[float, list]] = {}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def replay(symbol: str, tf: str, bars: int = 2000) -> list[dict]:
    key = f"{symbol}:{tf}"
    hit = _replay_cache.get(key)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    df = market.history(symbol, tf, bars + backtest.WINDOW)
    rep = backtest.run(symbol, tf, df=df, store=False, step=2)
    res = rep.get("_results", [])
    _replay_cache[key] = (time.time(), res)
    return res


def lookalikes(res: dict, history: list[dict], k: int = 40, direction: int | None = None) -> dict:
    d = direction if direction is not None else res["direction"]
    now = np.array([res["components"].get(c, 0.0) for c in TECH])
    same = [t for t in history if t["res"]["direction"] == d]
    scored = sorted(((_cos(now, np.array([t["res"]["components"].get(c, 0.0) for c in TECH])), t) for t in same),
                    key=lambda x: -x[0])
    top = [(s, t) for s, t in scored[:k] if s >= 0.6]
    rs = [t["r_multiple"] for _, t in top]
    out = {"found": len(top), "searched": len(history),
           "avg_similarity": round(float(np.mean([s for s, _ in top])), 3) if top else None,
           **learning.summarize(rs)}
    out["reach"] = [t.get("reach_r", 0.0) for _, t in top]
    out["examples"] = [{"when": time.strftime("%Y-%m-%d %H:%M", time.gmtime(t["opened_at"])),
                        "order": t["res"]["plan"]["order_type"], "result_R": round(t["r_multiple"], 2),
                        "similarity": round(s, 2)} for s, t in top[:6]]
    return out


def analyze(symbol: str, tf: str = "H1", source: str = "chat", progress=None, journal: bool = True) -> dict:
    t0 = time.time()
    say = progress or (lambda msg: None)
    inst = resolve(symbol)
    tf = tf.upper()

    say("Reading the main timeframe with fundamentals and news")
    main = engine.analyze(inst.name, tf, source=source, journal=False)

    say("Top-down: checking higher and lower timeframes")
    order = list(TIMEFRAMES)
    frames = {}
    for t in ("D1", "H4", "H1", "M15"):
        if t == tf or (order.index(t) < order.index(tf) and t != "M15"):
            continue
        try:
            r = engine.evaluate(inst.name, t, market.closed_candles(inst.name, t, 400))
            frames[t] = {"direction": "LONG" if r["direction"] > 0 else "SHORT", "dir": r["direction"],
                         "p_win": round(r["probability"], 3), "regime": r["regime"], "setup": r["plan"]["setup_type"]}
        except Exception as e:
            frames[t] = {"error": str(e)}
    higher = [f for t, f in frames.items() if "dir" in f and order.index(t) > order.index(tf)]

    say("Replaying this instrument's history to find look-alike setups for BOTH a buy and a sell (the slow part)")
    try:
        hist = replay(inst.name, tf)
    except Exception as e:
        log.warning("replay failed: %s", e)
        hist = []

    say("Weighing the buy case against the sell case")
    cases = {}
    for d in (1, -1):
        sd = main["_sides"][d]
        analog = lookalikes(main, hist, direction=d) if hist else {"found": 0}
        n = analog.get("found", 0) or 0
        wins = round((analog.get("win_rate") or 0) * n)
        p = (sd["p"] * MODEL_WEIGHT + wins) / (MODEL_WEIGHT + n)
        e = (sd["expectancy"] * MODEL_WEIGHT + (analog.get("total_r") or 0.0)) / (MODEL_WEIGHT + n)
        reach = analog.get("reach", [])
        tps = []
        for tpi in sd["order"]["take_profits"]:
            hits = sum(1 for x in reach if x >= tpi["r"])
            p_tp = (tpi["p_reach"] * MODEL_WEIGHT + hits) / (MODEL_WEIGHT + n)
            tps.append({**tpi, "p_reach": p_tp, "p_reach_model": tpi["p_reach"],
                        "history_hits": f"{hits}/{n}" if n else None,
                        "expected_r_if_closed_here": p_tp * tpi["r"] - (1 - p_tp) - sd["cost_r"]})
        agree = sum(1 for f in higher if f["dir"] == d)
        reasons = []
        if higher and agree / len(higher) < 0.5:
            reasons.append(f"only {agree} of {len(higher)} higher timeframes agree")
        if sd["blocked"]:
            reasons.append(sd["blocked"])
        cases[d] = {"p": p, "e": e, "tps": tps, "analog": analog, "agree": agree, "reasons": reasons,
                    "qualifies": p >= config.MIN_PROBABILITY and e >= config.MIN_EXPECTANCY_R and not reasons}

    best_d = max((1, -1), key=lambda d: (cases[d]["qualifies"], round(cases[d]["e"], 9)))
    c, sd = cases[best_d], main["_sides"][best_d]
    if best_d != main["direction"]:   # the deeper evidence favours the other side: switch to its full plan
        main.update(direction=best_d, plan=sd["order"], entry=sd["order"]["entry"], sl=sd["order"]["sl"],
                    tp=sd["order"]["tp"], setup_key=sd["setup_key"], raw_probability=sd["raw_p"], cost_r=sd["cost_r"])
    p_model, e_model = sd["p"], sd["expectancy"]
    sd["order"]["take_profits"] = c["tps"]
    main.update(probability=c["p"], expectancy=c["e"], take_profits=c["tps"],
                is_signal=bool(c["qualifies"] and "no directional edge" not in (main.get("blocked_reason") or "")),
                blocked_reason="; ".join(c["reasons"]) or None, deep=True)
    main["explanation"] = plan.explain(main, learning.model().weights())
    for d in (1, -1):
        side = main["both_sides"]["buy" if d > 0 else "sell"]
        side.update(p_win=cases[d]["p"], expected_r=cases[d]["e"], take_profits=cases[d]["tps"],
                    qualifies=cases[d]["qualifies"], blocked="; ".join(cases[d]["reasons"]) or None,
                    higher_tf_agree=f"{cases[d]['agree']}/{len(higher)}" if higher else "n/a",
                    lookalikes=cases[d]["analog"].get("found", 0),
                    lookalike_win_rate=cases[d]["analog"].get("win_rate"))
    for f in frames.values():
        if "dir" in f:
            f["agrees"] = f.pop("dir") == best_d
    analog, agree = c["analog"], c["agree"]
    if journal and not main.get("market_closed"):
        main["analysis_id"] = engine.record(main, source)
        if main["analysis_id"]:   # every analysis you ask for is paper-traded
            main["trade_id"] = paper.open_trade(main, source="live", entry_price=engine._live_entry(inst, main))
    secs = time.time() - t0
    return {**main, "deep": {
        "seconds": round(secs, 1), "model_p_win": round(p_model, 3), "model_expected_R": round(e_model, 3),
        "blended_p_win": round(c["p"], 3), "blended_expected_R": round(c["e"], 3),
        "timeframes": frames, "higher_tf_alignment": f"{agree}/{len(higher)}" if higher else "n/a",
        "lookalike_history": {k: v for k, v in analog.items() if k != "reach"},
        "method": "probability = (model x 30 + look-alike wins) / (30 + look-alikes)",
    }}
