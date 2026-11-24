"""Changes made after the 24 Nov paper-trades report."""
import numpy as np
import pandas as pd
import pytest

from sigma import config, learning, paper


def bars(rows):
    idx = pd.date_range("2026-11-24", periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_breakeven_after_one_r():
    # buy at 100, stop 99, target 102: price reaches +1.4R, then falls through the entry
    b = bars([[100, 100.5, 99.8, 100.4], [100.4, 101.4, 100.2, 101.2], [101.2, 101.3, 99.9, 100.0], [100, 100.1, 98.5, 98.8]])
    r = paper.simulate(1, 100, 99, 102, b, cost_r=0.03)
    assert r["status"] == "breakeven" and r["r_multiple"] == pytest.approx(-0.03) and r["mfe_r"] == pytest.approx(1.4)
    # the same path without the rule is a full loss
    assert paper.simulate(1, 100, 99, 102, b, cost_r=0.03, breakeven_r=0)["status"] == "lost"


def test_breakeven_never_inside_the_same_bar_and_winners_still_win():
    # one bar goes +1.2R and back below entry: the stop was still the original one in that bar
    b = bars([[100, 101.2, 99.95, 100.1], [100.1, 102.1, 100.05, 102.0]])
    assert paper.simulate(1, 100, 99, 102, b)["status"] == "won"
    # a sell: +1R, then the target
    s = bars([[100, 100.2, 98.9, 99.0], [99.0, 99.5, 97.9, 98.0]])
    assert paper.simulate(-1, 100, 101, 98, s)["status"] == "won"


def test_calibration_never_says_certain():
    m = learning.Model()
    m.meta["calib"] = [0.0, 9.0]                      # a wildly stretched refit
    assert m.calibrate(0.999) <= learning.P_CEIL and m.calibrate(0.001) >= learning.P_FLOOR
    rng = np.random.default_rng(0)
    raw = list(np.where(rng.random(60) < 0.5, 0.9, 0.1))
    y = [1.0 if p > 0.5 else 0.0 for p in raw]       # a perfectly separated small sample
    a, c = learning.fit_calibration(raw, y, np.ones(60))
    assert c <= learning.CALIB_SLOPE_MAX


def test_momentum_against_the_trade_blocks_that_side(monkeypatch):
    import os, tempfile
    from sigma import db, engine
    db.connect(os.path.join(tempfile.mkdtemp(), "m.db"))
    learning._model = None
    idx = pd.date_range("2026-11-01", periods=300, freq="15min", tz="UTC")
    c = 1.1 + np.cumsum(np.random.default_rng(4).normal(0, 0.0005, 300))
    df = pd.DataFrame({"open": c, "high": c + 0.0004, "low": c - 0.0004, "close": c, "volume": 0.0}, index=idx)
    real = engine.technical.analyze if hasattr(engine, "technical") else None
    res = engine.evaluate("EURUSD", "M15", df)
    mom = res["components"]["momentum"]
    sides = res["both_sides"]
    if mom >= config.MOMENTUM_VETO:
        assert "momentum is against the sell" in (sides["sell"]["blocked"] or "")
        assert not sides["sell"]["qualifies"]
    elif mom <= -config.MOMENTUM_VETO:
        assert "momentum is against the buy" in (sides["buy"]["blocked"] or "")
        assert not sides["buy"]["qualifies"]
    else:
        assert "momentum" not in (sides["buy"]["blocked"] or "") + (sides["sell"]["blocked"] or "")
