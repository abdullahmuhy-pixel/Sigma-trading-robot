"""Walk-forward backtest. Replays history bar by bar with the same engine used
live, opens a trade for every analysis, and periodically refits the model using
only trades that had already closed at that point (no look-ahead). The trades
are stored with source='backtest' so they bootstrap the learning model.

Fundamentals and news are not available historically, so they are zero here;
their weights are learned from live paper trades only."""
from __future__ import annotations

import logging
import time

import pandas as pd

from . import config, db, learning
from .data import market
from .engine import cost_in_price, evaluate, resample_htf
from .paper import simulate
from .symbols import resolve, tf_seconds

log = logging.getLogger("sigma.backtest")
WINDOW = 300

_progress: dict[str, dict] = {}


def progress() -> dict:
    return _progress


def run(symbol: str, tf: str, bars: int = 3000, df: pd.DataFrame | None = None,
        refit_every: int = 150, step: int = 1, store: bool = True) -> dict:
    inst = resolve(symbol)
    tf = tf.upper()
    if df is None:
        df = market.history(inst.name, tf, bars + WINDOW)
    df = df.sort_index()
    key = f"{inst.name}:{tf}"
    _progress[key] = {"done": 0, "total": max(0, len(df) - WINDOW - 1), "status": "running"}
    if store:
        with db.tx() as c:  # rerunning replaces the previous backtest for this market
            c.execute("DELETE FROM trades WHERE source='backtest' AND symbol=? AND timeframe=?", (inst.name, tf))
            c.execute("DELETE FROM analyses WHERE source='backtest' AND symbol=? AND timeframe=?", (inst.name, tf))

    secs = tf_seconds(tf)
    model = learning.refit(before=times_start(df), use_db=store)
    pending: list[dict] = []          # trades not yet resolved at the current bar
    results: list[dict] = []
    since_refit = 0
    times = [ts.timestamp() for ts in df.index]

    for i in range(WINDOW, len(df) - 1, step):
        window = df.iloc[i - WINDOW + 1:i + 1]
        now = times[i] + secs     # analysis happens when bar i closes

        # resolve pending trades with information up to this bar
        still = []
        for p in pending:
            if p["closed_at"] <= now:
                results.append(p)
                since_refit += 1
            else:
                still.append(p)
        pending = still
        if since_refit >= refit_every:
            model = learning.refit(before=now, use_db=store, extra_rows=[
                {"direction": t["direction"], "r_multiple": t["r_multiple"], "source": "backtest",
                 "components": t["res"]["components"], "regime": t["res"]["regime"],
                 "raw_probability": t["res"]["raw_probability"]} for t in results])
            since_refit = 0

        try:
            res = evaluate(inst.name, tf, window, resample_htf(window, tf), model=model,
                           stats=_stats_before(results), now=now)
        except Exception as e:
            log.debug("bar %s skipped: %s", i, e)
            continue

        entry = float(df["open"].iloc[i + 1])     # fill at next bar's open
        d = res["direction"]
        risk = abs(res["entry"] - res["sl"])
        sl, tp = entry - d * risk, entry + d * config.REWARD_R * risk
        expires = now + config.MAX_HOLD_BARS * secs
        future = df.iloc[i + 1:i + 2 + config.MAX_HOLD_BARS]
        cost_r = cost_in_price(inst, entry) / max(risk, 1e-12)
        out = simulate(d, entry, sl, tp, future, expires, cost_r)
        if out is None:
            break  # ran out of future data
        trade = {**out, "direction": d, "entry": entry, "sl": sl, "tp": tp, "opened_at": now,
                 "expires_at": expires, "res": res}
        pending.append(trade)
        _progress[key]["done"] = i - WINDOW

    results += pending
    if store:
        _store(results)
        learning.refit_and_save()
    _progress[key]["status"] = "done"
    return report(results, inst.name, tf)


def times_start(df: pd.DataFrame) -> float:
    return df.index[0].timestamp()


def _stats_before(results: list[dict]) -> dict:
    agg: dict[str, dict] = {}
    for t in results:
        if not t["res"]["is_signal"]:
            continue
        a = agg.setdefault(t["res"]["setup_key"], {"n": 0, "wins": 0, "sum_r": 0.0})
        a["n"] += 1
        a["wins"] += t["r_multiple"] > 0
        a["sum_r"] += t["r_multiple"]
    for a in agg.values():
        a["shrunk_expectancy"] = learning.shrunk_expectancy(a["sum_r"], a["n"])
    return agg


def _store(results: list[dict]) -> None:
    from .engine import record
    for t in results:
        res = t["res"]
        aid = record(res, "backtest")
        if not aid:
            continue
        db.insert("trades", {
            "analysis_id": aid, "symbol": res["symbol"], "timeframe": res["timeframe"], "source": "backtest",
            "direction": t["direction"], "entry": t["entry"], "sl": t["sl"], "tp": t["tp"],
            "opened_at": t["opened_at"], "expires_at": t["expires_at"], "status": t["status"],
            "exit_price": t["exit_price"], "closed_at": t["closed_at"], "r_multiple": t["r_multiple"],
            "mfe_r": t["mfe_r"], "mae_r": t["mae_r"], "setup_key": res["setup_key"], "learned": 1,
        })


def report(results: list[dict], symbol: str, tf: str) -> dict:
    allr = [t["r_multiple"] for t in results]
    sig = [t["r_multiple"] for t in results if t["res"]["is_signal"]]
    half = len(results) // 2
    late_sig = [t["r_multiple"] for t in results[half:] if t["res"]["is_signal"]]
    return {
        "symbol": symbol, "timeframe": tf, "analyses": len(results),
        "all_trades": learning.summarize(allr),
        "signals": learning.summarize(sig),
        "signals_second_half": learning.summarize(late_sig),
        "finished_at": time.time(),
    }
