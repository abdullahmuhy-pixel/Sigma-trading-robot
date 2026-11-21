import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

os.environ["SIGMA_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")

from sigma import config, db, learning  # noqa: E402
from sigma.analysis import indicators as ta, technical  # noqa: E402
from sigma.analysis.sentiment import score_headlines  # noqa: E402
from sigma.engine import evaluate, trade_levels  # noqa: E402
from sigma.paper import simulate  # noqa: E402
from sigma.symbols import resolve  # noqa: E402


def make_df(closes, spread=0.5):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="1h", tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    o = c.shift().fillna(c.iloc[0])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + spread, "low": np.minimum(o, c) - spread,
                         "close": c, "volume": 0.0})


def gbm(n=1500, seed=1, drift=0.0, vol=0.004, start=100.0):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    c = start * np.exp(np.cumsum(r))
    df = make_df(c, 0)
    wig = np.abs(rng.normal(0, vol * 0.6, n)) * c
    df["high"] = np.maximum(df["open"], df["close"]) + wig
    df["low"] = np.minimum(df["open"], df["close"]) - wig
    return df


# ------------------------------------------------------------ indicators --
def test_sma_ema():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert ta.sma(s, 3).iloc[-1] == 4.0
    # EMA span 3 -> alpha 0.5, recursive from first value: 1, 1.5, 2.25, 3.125, 4.0625
    assert ta.ema(s, 3).iloc[-1] == pytest.approx(4.0625)


def test_rsi_wilder_reference():
    # Classic Wilder example (New Concepts in Technical Trading Systems / StockCharts table)
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
              45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64]
    r = ta.rsi(pd.Series(closes), 14)
    assert r.iloc[14] == pytest.approx(70.46, abs=0.05)
    assert r.iloc[15] == pytest.approx(66.25, abs=0.05)
    assert r.iloc[16] == pytest.approx(66.48, abs=0.05)
    assert r.iloc[19] == pytest.approx(57.91, abs=0.1)


def test_rsi_extremes():
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert ta.rsi(up).iloc[-1] == 100.0
    assert ta.rsi(-up).iloc[-1] == pytest.approx(0.0)


def test_atr_constant_range():
    df = make_df([100.0] * 50, spread=1.0)  # every bar high-low = 2
    assert ta.atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_macd_zero_on_flat():
    line, sig, hist = ta.macd(pd.Series([5.0] * 60))
    assert abs(line.iloc[-1]) < 1e-12 and abs(hist.iloc[-1]) < 1e-12


def test_bollinger_and_pivots():
    lo, mid, up = ta.bollinger(pd.Series([1, 2, 3, 4, 5] * 4, dtype=float), 20, 2)
    assert mid.iloc[-1] == 3.0
    assert up.iloc[-1] - mid.iloc[-1] == pytest.approx(2 * math.sqrt(2))
    p = ta.pivots(110, 90, 100)
    assert p == {"P": 100, "R1": 110, "S1": 90, "R2": 120, "S2": 80}


def test_adx_strong_trend():
    df = make_df(np.linspace(100, 200, 120), spread=0.3)
    a, pdi, mdi = ta.adx(df)
    assert a.iloc[-1] > 40 and pdi.iloc[-1] > mdi.iloc[-1]


def test_supertrend_direction():
    up = make_df(np.linspace(100, 150, 80))
    dn = make_df(np.linspace(150, 100, 80))
    assert ta.supertrend(up).iloc[-1] == 1
    assert ta.supertrend(dn).iloc[-1] == -1


def test_swings_no_lookahead():
    df = gbm(200)
    highs, lows = ta.swings(df, 3, 3)
    assert all(i <= len(df) - 4 for i, _ in highs + lows)


# ------------------------------------------------------------- technical --
def test_technical_scores_bounded_and_directional():
    up = technical.analyze(gbm(400, seed=3, drift=0.002))
    dn = technical.analyze(gbm(400, seed=3, drift=-0.002))
    for res in (up, dn):
        assert all(-1 <= v <= 1 for v in res["components"].values())
    assert up["components"]["trend"] > 0.5 > -0.5 > dn["components"]["trend"]


# ------------------------------------------------------------- levels/R ---
def test_trade_levels_risk_reward():
    e, sl, tp = trade_levels(1, 100.0, 2.0, {})
    assert sl == pytest.approx(100 - config.SL_ATR_MULT * 2)
    assert (tp - e) / (e - sl) == pytest.approx(config.REWARD_R)
    e, sl, tp = trade_levels(-1, 100.0, 2.0, {"last_swing_high": 103.0})
    assert sl == pytest.approx(103.4) and tp == pytest.approx(100 - config.REWARD_R * 3.4)


def test_simulate_outcomes():
    idx = pd.date_range("2025-01-01", periods=4, freq="1h", tz="UTC")
    bars = pd.DataFrame({"open": [100, 101, 102, 103], "high": [101, 102, 104.5, 104],
                         "low": [99.5, 100.5, 101, 102], "close": [101, 102, 103, 103.5]}, index=idx)
    won = simulate(1, 100, 98, 104, bars)
    assert won["status"] == "won" and won["r_multiple"] == pytest.approx(2.0)
    lost = simulate(-1, 100, 102, 96, bars)
    assert lost["status"] == "lost" and lost["r_multiple"] == pytest.approx(-1.0)
    both = pd.DataFrame({"open": [100], "high": [105], "low": [97], "close": [100]}, index=idx[:1])
    assert simulate(1, 100, 98, 104, both)["status"] == "lost"  # conservative
    exp = simulate(1, 100, 90, 120, bars, expires_at=idx[2].timestamp(), cost_r=0.1)
    assert exp["status"] == "expired" and exp["exit_price"] == 102
    assert exp["r_multiple"] == pytest.approx((102 - 100) / 10 - 0.1)
    assert simulate(1, 100, 90, 120, bars) is None  # still open


# -------------------------------------------------------------- learning --
def test_summarize_numbers():
    s = learning.summarize([2, -1, -1, 2, 2, -1])
    assert s["trades"] == 6 and s["win_rate"] == 0.5
    assert s["expectancy_r"] == pytest.approx(0.5) and s["total_r"] == 3
    assert s["profit_factor"] == pytest.approx(2.0) and s["max_drawdown_r"] == 2


def test_breakeven_prior_means_zero_edge():
    m = learning.Model()
    p = m.prob({}, "transition", 1)
    assert p == pytest.approx(1 / (1 + config.REWARD_R))
    m.meta["avg_win_r"], m.meta["avg_loss_r"] = config.REWARD_R, -1
    assert m.expectancy(p) == pytest.approx(0.0, abs=1e-9)


def test_model_learns_which_component_works():
    rng = np.random.default_rng(0)
    n = 3000
    X, y = [], []
    for _ in range(n):
        comps = {b: rng.uniform(-1, 1) for b in learning.BASE}
        d = rng.choice([-1, 1])
        # truth: only mean_reversion predicts; trend is useless
        z = learning.prior_bias() + 2.0 * comps["mean_reversion"] * d
        y.append(float(rng.random() < 1 / (1 + math.exp(-z))))
        X.append(learning.featurize(comps, "transition", d))
    m = learning.Model()
    m.fit(np.array(X), np.array(y), np.ones(n), iters=1500)
    w = m.weights()
    assert w["mean_reversion"] > 1.2
    assert abs(w["trend"]) < 0.25  # prior 0.6 was pulled toward the truth (0)


def test_shrinkage():
    assert learning.shrunk_expectancy(-5, 5) == pytest.approx(-1 / 3)
    assert learning.shrunk_expectancy(-50, 50) == pytest.approx(-50 / 60)


# ---------------------------------------------------------------- news ----
def test_headline_scoring_direction():
    now = 1_700_000_000
    items = [{"title": "ECB signals another hike as inflation accelerates", "summary": "", "published": now},
             {"title": "Fed cuts rates, dovish outlook", "summary": "", "published": now}]
    s, used = score_headlines(items, "EUR", "USD", now)
    assert s > 0 and len(used) == 2
    s2, _ = score_headlines(items, "USD", "EUR", now)
    assert s2 == pytest.approx(-s)


def test_symbols():
    assert resolve("eur/usd").deriv == "frxEURUSD"
    assert resolve("XAUUSD").asset_class == "metal"
    assert resolve("BTCUSDT").binance == "BTCUSDT"
    assert resolve("R_75").asset_class == "synthetic"
    assert resolve("boom1000").name == "BOOM1000"


# -------------------------------------------------------------- engine ----
def test_evaluate_and_backtest_pipeline():
    db.connect(os.environ["SIGMA_DB"])
    df = gbm(1500, seed=7)
    res = evaluate("EURUSD", "H1", df.iloc[:400])
    assert res["direction"] in (1, -1) and 0 < res["probability"] < 1
    assert res["setup_key"].startswith("EURUSD|H1|")
    risk = abs(res["entry"] - res["sl"])
    assert abs(res["tp"] - res["entry"]) == pytest.approx(config.REWARD_R * risk)

    from sigma import backtest
    rep = backtest.run("EURUSD", "H1", df=df, refit_every=100)
    assert rep["analyses"] > 700
    n_db = db.q1("SELECT COUNT(*) n FROM trades WHERE source='backtest'")["n"]
    assert n_db == rep["analyses"]
    # every stored R must be consistent with its exit price
    for t in db.q("SELECT * FROM trades WHERE source='backtest' LIMIT 200"):
        risk = abs(t["entry"] - t["sl"])
        gross = t["direction"] * (t["exit_price"] - t["entry"]) / risk
        assert gross - t["r_multiple"] >= -1e-9          # costs only ever reduce R
        assert gross - t["r_multiple"] < 0.5
    m = learning.model()
    assert m.meta["n_backtest"] == n_db
    # random walk: no real edge, so calibration must learn to distrust the model
    assert m.meta["calib"][1] < learning.CALIB_SLOPE0
    assert rep["signals_second_half"]["trades"] < rep["signals"]["trades"] / 2


def mirror(df):
    C = 2 * df["close"].mean()
    return pd.DataFrame({"open": C - df["open"], "high": C - df["low"], "low": C - df["high"],
                         "close": C - df["close"], "volume": df["volume"]}, index=df.index)


def test_no_directional_bias_mirror():
    """Flip a chart upside down: every score must flip and the decision must reverse
    with exactly the same probability. Any built-in lean would break this."""
    from sigma import engine
    for seed in range(12):
        for drift in (0.0, 0.0015, -0.0015):
            df = gbm(420, seed=seed, drift=drift)
            a, b = technical.analyze(df), technical.analyze(mirror(df))
            for k in a["components"]:
                assert a["components"][k] == pytest.approx(-b["components"][k], abs=1e-9), k
            ra = engine.evaluate("EURUSD", "H1", df, engine.resample_htf(df, "H1"))
            rb = engine.evaluate("EURUSD", "H1", mirror(df), engine.resample_htf(mirror(df), "H1"))
            assert ra["direction"] == -rb["direction"]
            assert ra["probability"] == pytest.approx(rb["probability"], abs=1e-9)


def test_balanced_evidence_is_not_a_signal():
    from sigma import engine
    flat = make_df([100.0 + 0.0 * i for i in range(400)], spread=0.5)
    r = engine.evaluate("EURUSD", "H1", flat)
    assert not r["is_signal"]


def test_order_types_are_chosen_by_rule_and_mirror():
    from sigma import plan
    atr = 1.0
    base = {"ema20": 100.0, "nearest_support": None, "nearest_resistance": None,
            "last_swing_low": None, "last_swing_high": None}
    comps = {"trend": 0.6, "structure": 0.5, "mean_reversion": 0.0}
    # resistance 0.4 ATR above in a trending market -> BUY STOP just beyond it
    o = plan.order(1, 100.5, atr, {**base, "nearest_resistance": 100.9}, comps, "trending")
    assert o["order_type"] == "BUY STOP" and o["entry"] == pytest.approx(101.0)
    # mirror: support 0.4 ATR below, bearish evidence -> SELL STOP
    neg = {k: -v for k, v in comps.items()}
    o = plan.order(-1, 99.5, atr, {**base, "nearest_support": 99.1}, neg, "trending")
    assert o["order_type"] == "SELL STOP" and o["entry"] == pytest.approx(99.0)
    # price ran 1.4 ATR above EMA20 -> BUY LIMIT back near the EMA
    o = plan.order(1, 101.4, atr, base, comps, "trending")
    assert o["order_type"] == "BUY LIMIT" and o["entry"] == pytest.approx(100.05)
    o = plan.order(-1, 98.6, atr, base, neg, "trending")
    assert o["order_type"] == "SELL LIMIT" and o["entry"] == pytest.approx(99.95)
    # nothing nearby -> market order
    o = plan.order(1, 100.2, atr, base, comps, "trending")
    assert o["order_type"] == "BUY" and o["kind"] == "MARKET"
    # ranging market: no breakout orders
    o = plan.order(1, 100.5, atr, {**base, "nearest_resistance": 100.9}, comps, "ranging")
    assert "STOP" not in o["order_type"]
    # every plan keeps reward = REWARD_R x risk from ITS entry
    assert abs(o["tp"] - o["entry"]) == pytest.approx(config.REWARD_R * abs(o["entry"] - o["sl"]))


def test_pending_fill_rules():
    from sigma.paper import fill_index
    idx = pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC")
    bars = pd.DataFrame({"open": [100, 99.5, 98.0], "high": [100.4, 99.8, 98.5], "low": [99.6, 99.2, 97.5],
                         "close": [99.6, 99.3, 98.2]}, index=idx)
    assert fill_index(1, "LIMIT", 99.3, bars) == (1, 99.3)       # buy limit touched on bar 1
    assert fill_index(1, "LIMIT", 98.8, bars) == (2, 98.0)       # gapped below: filled at the better open
    assert fill_index(-1, "STOP", 98.8, bars) == (2, 98.0)       # sell stop gapped through: filled at the open
    assert fill_index(1, "STOP", 101.0, bars) is None            # never reached: stays unfilled


def test_signal_explanation_is_complete():
    from sigma import engine
    r = engine.evaluate("EURUSD", "H1", gbm(400, seed=5, drift=-0.002))
    ex = r["explanation"]
    side = "SELL" if r["direction"] < 0 else "BUY"
    assert side in ex["summary"] and r["plan"]["order_type"] in ex["summary"]
    assert ex["for"] and all(c["impact"] > 0 for c in ex["for"]) and all(c["impact"] < 0 for c in ex["against"])
    assert "stop loss" in ex["invalidation"]


def test_goal_monte_carlo_matches_exact_maths():
    from math import comb
    from sigma import goals
    bal, tgt, risk, n, p, w, l = 10_000, 300, 1.0, 15, 0.4, 2.0, -1.0
    # with fixed % risk the final balance depends only on how many trades win
    exact = sum(comb(n, k) * p**k * (1 - p)**(n - k) for k in range(n + 1)
                if bal * (1 + w * risk / 100)**k * (1 + l * risk / 100)**(n - k) - bal >= tgt)
    mc = goals.simulate(bal, tgt, risk, n, p, w, l, sims=60000)["p_reach_target"]
    assert mc == pytest.approx(exact, abs=0.01)
    assert goals.simulate(bal, tgt, risk, n, 1.0, w, l)["p_reach_target"] == 1.0


def test_goal_plan_numbers_and_warnings():
    from sigma import goals
    db.connect(os.environ["SIGMA_DB"])
    goals.set_goal(50_000, "ZAR", 1_000, "weekly", risk_pct=1)
    p = goals.plan()
    assert p["goal"]["period"] == "week" and p["required_return_pct"] == pytest.approx(2.0)
    assert p["monthly_equivalent_pct"] == pytest.approx(2.0 * 21 / 5, abs=0.1)
    assert p["risk_per_trade"] == pytest.approx(500.0)
    assert p["full_wins_needed"] == int(np.ceil(1000 / (500 * p["robot_edge"]["avg_win_r"])))
    goals.set_goal(1_000, "USD", 500, "day")           # 50% in a day
    assert any("not realistic" in w for w in goals.plan()["warnings"])
    goals.log_result(120, "EURUSD win")
    pl = goals.plan()
    assert pl["progress_this_period"] == 120 and pl["remaining"] == 380 and pl["goal"]["balance"] == 1_120


def test_open_minded_takes_the_other_side_when_one_is_blocked():
    """The robot must not fixate on one direction: if the stronger-looking side fails a check,
    the other side is judged on its own merits instead of answering 'no trade'."""
    from sigma import engine
    m = learning.Model()
    m.bias = 2.0                      # make both sides attractive so only the filter decides
    m.meta["calib"] = [0.0, 1.0]
    df = gbm(420, seed=4, drift=0.002)
    first = engine.evaluate("EURUSD", "H1", df, model=m, stats={})
    assert first["is_signal"]
    gate = {first["setup_key"]: {"n": 50, "shrunk_expectancy": -0.5}}   # switch that side off
    second = engine.evaluate("EURUSD", "H1", df, model=m, stats=gate)
    assert second["direction"] == -first["direction"] and second["is_signal"]
    assert set(second["both_sides"]) == {"buy", "sell"}
    assert second["both_sides"]["buy" if first["direction"] > 0 else "sell"]["blocked"]


def test_take_profit_odds():
    from sigma import engine, plan
    base = 1 / (1 + config.REWARD_R)
    for r in (1, 2, 3, 4):   # no edge: the fair random-walk chance of +r before -1
        assert plan.tp_probability(base, r) == pytest.approx(1 / (1 + r))
    assert plan.tp_probability(0.45, config.REWARD_R) == pytest.approx(0.45)
    res = engine.evaluate("EURUSD", "H1", gbm(420, seed=2, drift=0.001))
    tps = res["take_profits"]
    assert [t["name"] for t in tps] == ["TP1", "TP2", "TP3", "TP4"]
    assert all(a["p_reach"] > b["p_reach"] for a, b in zip(tps, tps[1:]))      # further = less likely
    d, risk = res["direction"], abs(res["entry"] - res["sl"])
    for t in tps:
        assert t["price"] == pytest.approx(res["entry"] + d * t["r"] * risk)
    main = next(t for t in tps if t["main"])
    assert main["p_reach"] == pytest.approx(res["probability"]) and main["price"] == pytest.approx(res["tp"])


def test_reach_before_stop():
    from sigma.paper import reach_before_stop
    idx = pd.date_range("2025-01-01", periods=4, freq="1h", tz="UTC")
    bars = pd.DataFrame({"open": [100, 101, 103, 104], "high": [101, 103.5, 104.2, 106],
                         "low": [99.5, 100.5, 102, 97], "close": [101, 103, 104, 98]}, index=idx)
    # long from 100, stop 98 (risk 2): best high before the stop bar is 104.2 -> 2.1R
    assert reach_before_stop(1, 100, 98, bars) == pytest.approx(2.1)
    assert reach_before_stop(-1, 100, 102, bars) == pytest.approx(0.25)   # short: stopped on bar 1, best low 99.5


def test_order_follows_live_price_users_btc_case():
    """The user's BTC/USD case: SELL planned at 60,314.40 on the closed candle, but price is already
    59,810.89 when the analysis finishes -> it must become a SELL LIMIT at 60,314.40."""
    from sigma import plan
    o = {"side": "SELL", "kind": "MARKET", "order_type": "SELL", "entry": 60314.40, "sl": 61753.06,
         "tp": 60314.40 - 2 * 1438.66, "risk": 1438.66, "pending_bars": 0, "order_reason": "", "setup_type": "x"}
    a = plan.adjust_to_live(o, 59810.89, atr=600.0, ctx={})
    assert a["order_type"] == "SELL LIMIT" and a["entry"] == pytest.approx(60314.40)
    assert a["sl"] == pytest.approx(61753.06) and a["pending_bars"] == plan.PENDING_BARS
    # price moved against a sell (higher): sell at the better live price, levels rebuilt from it
    b = plan.adjust_to_live(o, 60600.0, atr=600.0, ctx={})
    assert b["order_type"] == "SELL" and b["entry"] == pytest.approx(60600.0)
    assert (b["sl"] - b["entry"]) == pytest.approx(config.SL_ATR_MULT * 600.0)
    # mirror for buys
    ob = {**o, "side": "BUY", "order_type": "BUY", "entry": 100.0, "sl": 98.5, "risk": 1.5}
    assert plan.adjust_to_live(ob, 100.6, atr=1.0, ctx={})["order_type"] == "BUY LIMIT"
    # a limit already reached / a stop already broken become market orders
    lim = {**o, "kind": "LIMIT", "order_type": "SELL LIMIT", "entry": 60314.40}
    assert plan.adjust_to_live(lim, 60400.0, atr=600.0, ctx={})["order_type"] == "SELL"
    stp = {**o, "kind": "STOP", "order_type": "SELL STOP", "entry": 60000.0}
    assert plan.adjust_to_live(stp, 59900.0, atr=600.0, ctx={})["order_type"] == "SELL"
    # small moves (under a quarter ATR) leave the plan alone
    assert plan.adjust_to_live(o, 60250.0, atr=600.0, ctx={})["order_type"] == "SELL"


def _bars(rows, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_call_grading_rules():
    from sigma.journal import grade_call
    far = 1e12
    up = _bars([(100, 100.5, 99.6, 100.4), (100.4, 101.2, 100.2, 101.0)])
    assert grade_call(1, 100, 1.0, up, far)["call_result"] == "right"        # +1R first
    assert grade_call(-1, 100, 1.0, up, far)["call_result"] == "wrong"     # the sell call lost 1R first
    down = _bars([(100, 100.3, 98.9, 99.0)])
    assert grade_call(1, 100, 1.0, down, far)["call_result"] == "wrong"
    assert grade_call(-1, 100, 1.0, down, far)["call_result"] == "right"
    both = _bars([(100, 101.5, 98.5, 100)])
    assert grade_call(1, 100, 1.0, both, far)["call_result"] == "unclear"
    quiet = _bars([(100, 100.4, 99.7, 100.1)] * 3)
    assert grade_call(1, 100, 1.0, quiet, far) is None                       # window still open: wait
    assert grade_call(1, 100, 1.0, quiet, quiet.index[-1].timestamp() + 1)["call_result"] == "flat"


def test_missed_limit_is_recorded():
    from sigma.paper import missed_or_cancelled
    sell_limit = {"direction": -1, "entry": 60314.40, "sl": 61753.06, "order_type": "SELL LIMIT"}
    # price never came back up to 60,314 but fell to 58,500: the call was right, the entry missed
    fell = _bars([(59810, 59900, 59200, 59300), (59300, 59350, 58500, 58600)])
    out = missed_or_cancelled(sell_limit, fell, 0)
    assert out["status"] == "missed" and out["missed_r"] == pytest.approx((60314.40 - 58500) / 1438.66, abs=0.01)
    assert "Direction right" in out["lesson"]
    # price drifted sideways: simply cancelled
    flat = _bars([(59810, 59900, 59700, 59800)])
    assert missed_or_cancelled(sell_limit, flat, 0)["status"] == "cancelled"
    # an untriggered buy stop is a plain cancel
    buy_stop = {"direction": 1, "entry": 101.0, "sl": 99.0, "order_type": "BUY STOP"}
    assert missed_or_cancelled(buy_stop, _bars([(100, 100.8, 99.5, 100.2)]), 0)["status"] == "cancelled"


def test_planner_learns_from_missed_limits():
    from sigma import plan
    base = {"ema20": 100.0, "nearest_support": None, "nearest_resistance": None,
            "last_swing_low": None, "last_swing_high": None}
    comps = {"trend": 0.6, "structure": 0.5, "mean_reversion": 0.0}
    normal = plan.order(1, 101.4, 1.0, base, comps, "trending", {"LIMIT": {"placed": 30, "miss_rate": 0.1}})
    assert normal["order_type"] == "BUY LIMIT" and normal["entry"] == pytest.approx(100.05)
    shallow = plan.order(1, 101.4, 1.0, base, comps, "trending", {"LIMIT": {"placed": 30, "miss_rate": 0.4}})
    assert shallow["order_type"] == "BUY LIMIT" and shallow["entry"] == pytest.approx(100.7)
    market = plan.order(1, 101.4, 1.0, base, comps, "trending", {"LIMIT": {"placed": 30, "miss_rate": 0.6}})
    assert market["order_type"] == "BUY"
    few = plan.order(1, 101.4, 1.0, base, comps, "trending", {"LIMIT": {"placed": 5, "miss_rate": 0.9}})
    assert few["entry"] == pytest.approx(100.05)        # not enough evidence to change anything
