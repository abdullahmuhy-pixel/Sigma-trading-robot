"""Your real account and profit goals, and an honest plan for reaching them.

You tell Sigma (in the chat) your real balance, currency and a target for today, this week or
this month. Sigma then:
  - turns the target into a required return and the number of winning trades it needs,
  - checks it against the robot's own track record with a Monte Carlo simulation
    (chance of reaching the target, chance of a painful drawdown first),
  - flags targets that would need dangerous risk, and
  - finds the current best setups and sizes each one for your account.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from . import config, db, learning

PERIOD_TRADING_DAYS = {"day": 1, "week": 5, "month": 21}
SAFE_RISK_PCT = 1.0       # risk per trade we recommend
MAX_RISK_PCT = 2.0        # above this, we warn plainly


def get() -> dict | None:
    return db.kv_get("goals")


def set_goal(balance: float, currency: str, target: float, period: str,
             risk_pct: float | None = None, trades_per_day: int | None = None) -> dict:
    period = {"day": "day", "daily": "day", "today": "day", "week": "week", "weekly": "week",
              "month": "month", "monthly": "month"}.get(period.strip().lower(), period)
    if period not in PERIOD_TRADING_DAYS:
        raise ValueError("period must be day, week or month")
    if balance <= 0 or target <= 0:
        raise ValueError("balance and target must be positive")
    g = {"balance": float(balance), "currency": currency.upper()[:3], "target": float(target), "period": period,
         "risk_pct": float(risk_pct or SAFE_RISK_PCT), "trades_per_day": int(trades_per_day or 3),
         "set_at": time.time(), "progress": (get() or {}).get("progress", [])}
    db.kv_set("goals", g)
    return g


def log_result(amount: float, note: str = "") -> dict:
    g = get()
    if not g:
        raise ValueError("no goal set yet")
    g.setdefault("progress", []).append({"t": time.time(), "amount": float(amount), "note": note})
    g["balance"] = g["balance"] + float(amount)
    db.kv_set("goals", g)
    return g


def _period_start(period: str) -> float:
    now = datetime.now(ZoneInfo(config.DISPLAY_TZ))
    d0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "week":
        d0 = d0 - timedelta(days=d0.weekday())
    elif period == "month":
        d0 = d0.replace(day=1)
    return d0.timestamp()


def edge() -> dict:
    """The robot's measured edge on signals: live first, backtest if live is thin, shrunk toward zero."""
    live = learning.performance("live", True)
    bt = learning.performance("backtest", True)
    src, perf = ("live", live) if (live["trades"] or 0) >= 30 else ("backtest", bt)
    n = perf["trades"] or 0
    if n == 0:
        return {"source": "none", "trades": 0, "win_rate": None, "avg_win_r": config.REWARD_R, "avg_loss_r": -1.0,
                "expectancy_r": 0.0, "note": "No track record yet: run backtests so the plan uses real numbers."}
    k = 20   # shrink toward break-even until there is plenty of data
    wr = (perf["win_rate"] * n + k * (1 / (1 + config.REWARD_R))) / (n + k)
    aw = perf.get("avg_win_r") or config.REWARD_R
    al = perf.get("avg_loss_r") or -1.0
    return {"source": src, "trades": n, "win_rate": wr, "avg_win_r": aw, "avg_loss_r": al,
            "expectancy_r": wr * aw + (1 - wr) * al}


def simulate(balance: float, target: float, risk_pct: float, n_trades: int, win_rate: float,
             avg_win_r: float, avg_loss_r: float, sims: int = 20000, max_dd_pct: float = 10.0,
             seed: int = 7) -> dict:
    """Monte Carlo of n_trades at fixed % risk (compounding). Returns the chance of reaching the
    target by the end of the period and the chance of falling max_dd_pct below the start first."""
    rng = np.random.default_rng(seed)
    wins = rng.random((sims, n_trades)) < win_rate
    r = np.where(wins, avg_win_r, avg_loss_r)
    growth = np.cumprod(1 + r * risk_pct / 100, axis=1)
    eq = balance * growth
    reached = (eq[:, -1] - balance) >= target
    dd = (eq.min(axis=1) <= balance * (1 - max_dd_pct / 100))
    return {"p_reach_target": float(reached.mean()), "p_drawdown_first": float(dd.mean()),
            "median_result": float(np.median(eq[:, -1] - balance)),
            "bad_case_10pct": float(np.percentile(eq[:, -1] - balance, 10))}


def plan() -> dict:
    g = get()
    if not g:
        return {"error": "No goal yet. Tell me your real balance, currency, and the profit you want per day, week or month."}
    e = edge()
    bal, tgt, risk = g["balance"], g["target"], g["risk_pct"]
    days = PERIOD_TRADING_DAYS[g["period"]]
    req_pct = 100 * tgt / bal
    monthly_equiv = req_pct * 21 / days
    risk_amount = bal * risk / 100
    win_money = risk_amount * e["avg_win_r"]
    exp_money = risk_amount * e["expectancy_r"]
    n_trades = max(1, days * g["trades_per_day"])
    sim = simulate(bal, tgt, risk, n_trades, e["win_rate"] or 1 / (1 + config.REWARD_R),
                   e["avg_win_r"], e["avg_loss_r"])
    # smallest risk % (up to the max) that gives at least a 50% chance - shown only for honesty
    needed = None
    for rp in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        if simulate(bal, tgt, rp, n_trades, e["win_rate"] or 1 / 3, e["avg_win_r"], e["avg_loss_r"],
                    sims=4000)["p_reach_target"] >= 0.5:
            needed = rp
            break
    warnings = []
    if monthly_equiv > 50:
        warnings.append(f"This target equals about {monthly_equiv:.0f}% a month. That is not realistic without "
                        f"gambling the account; consider a smaller target.")
    elif monthly_equiv > 15:
        warnings.append(f"This target equals about {monthly_equiv:.0f}% a month - very aggressive. Even good "
                        f"professional traders rarely sustain more than a few percent a month.")
    if risk > MAX_RISK_PCT:
        warnings.append(f"Risking {risk}% per trade is high: 10 losses in a row (which happens) would cost about "
                        f"{100 * (1 - (1 - risk / 100) ** 10):.0f}% of the account.")
    if needed is None or needed > MAX_RISK_PCT:
        warnings.append("Reaching this target with a 50% chance would need more than 2% risk per trade on the "
                        "robot's current track record. Sigma will not size trades above your risk setting.")
    if e["expectancy_r"] <= 0:
        warnings.append("The robot's measured edge is not positive yet, so no target is reliably reachable. "
                        "Keep paper trading until the track record improves.")
    start = _period_start(g["period"])
    made = sum(p["amount"] for p in g.get("progress", []) if p["t"] >= start)
    return {
        "goal": {k: g[k] for k in ("balance", "currency", "target", "period", "risk_pct", "trades_per_day")},
        "required_return_pct": round(req_pct, 2), "monthly_equivalent_pct": round(monthly_equiv, 1),
        "risk_per_trade": round(risk_amount, 2), "profit_per_full_win": round(win_money, 2),
        "expected_profit_per_trade": round(exp_money, 2),
        "full_wins_needed": int(np.ceil(tgt / win_money)) if win_money > 0 else None,
        "trades_available_in_period": n_trades,
        "robot_edge": e, "simulation": {k: round(v, 3) if "p_" in k else round(v, 2) for k, v in sim.items()},
        "risk_pct_for_even_odds": needed,
        "progress_this_period": round(made, 2), "remaining": round(max(0.0, tgt - made), 2),
        "warnings": warnings,
    }


def find_trades(timeframe: str = "H1", symbols: list[str] | None = None, max_trades: int = 3) -> dict:
    """Quick-scan the watchlist, then deep-check the best candidates and size them for the goal."""
    from . import deep, engine
    from .agent import position_size
    g = get()
    p = plan()
    if not g:
        return p
    cands = []
    for s in symbols or config.WATCHLIST:
        try:
            r = engine.analyze(s, timeframe, source="chat", journal=False)
            if not r.get("market_closed"):
                cands.append(r)
        except Exception:
            pass
    cands.sort(key=lambda r: (-int(r["is_signal"]), -r["expectancy"]))
    picks = []
    for r in cands[:max_trades + 2]:
        try:
            d = deep.analyze(r["symbol"], timeframe, source="chat")   # journalled + on the demo account
        except Exception:
            continue
        size = position_size(d["symbol"], g["balance"], g["risk_pct"], d["entry"], d["sl"], g["currency"])
        risk_amt = g["balance"] * g["risk_pct"] / 100
        picks.append({
            "symbol": d["display"], "code": d["symbol"], "timeframe": timeframe, "order_type": d["plan"]["order_type"],
            "setup": d["plan"]["setup_type"], "entry": d["entry"], "stop_loss": d["sl"], "take_profit": d["tp"],
            "is_signal": d["is_signal"], "p_win": round(d["probability"], 3), "expected_R": round(d["expectancy"], 3),
            "lots": size.get("lots"), "units": size.get("units"),
            "if_take_profit_hits": round(risk_amt * config.REWARD_R, 2), "if_stop_loss_hits": round(-risk_amt, 2),
            "share_of_remaining_goal": round(risk_amt * config.REWARD_R / p["remaining"], 2) if p.get("remaining") else None,
            "why": (d.get("explanation") or {}).get("summary"), "blocked": d.get("blocked_reason"),
        })
        if len([x for x in picks if x["is_signal"]]) >= max_trades:
            break
    picks.sort(key=lambda x: (-int(x["is_signal"]), -x["expected_R"]))
    return {"plan": p, "trades": picks[:max_trades + 1],
            "note": ("Only trades marked is_signal passed every check. Others are the best available leans, "
                     "shown for information - not recommended for real money.")}
