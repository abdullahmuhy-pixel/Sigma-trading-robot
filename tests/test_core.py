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
    df = gbm(900, seed=7)
    res = evaluate("EURUSD", "H1", df.iloc[:400])
    assert res["direction"] in (1, -1) and 0 < res["probability"] < 1
    assert res["setup_key"].startswith("EURUSD|H1|")
    risk = abs(res["entry"] - res["sl"])
    assert abs(res["tp"] - res["entry"]) == pytest.approx(config.REWARD_R * risk)

    from sigma import backtest
    rep = backtest.run("EURUSD", "H1", df=df, refit_every=100)
    assert rep["analyses"] > 400
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
