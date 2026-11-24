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


def test_demo_account_skips_synthetic_signals_but_keeps_chat(monkeypatch):
    import os, tempfile, time
    from sigma import account, db
    db.connect(os.path.join(tempfile.mkdtemp(), "mk.db"))
    assert account.settings()["markets"] == ["forex", "metal", "crypto"]

    def trade(sym, source="live"):
        aid = db.insert("analyses", {"created_at": time.time(), "bar_time": time.time() + hash(sym + source) % 1000,
                                     "symbol": sym, "timeframe": "H1", "source": source, "direction": 1})
        res = {"analysis_id": aid, "symbol": sym, "timeframe": "H1", "bar_time": time.time() + hash(sym + source) % 1000,
               "direction": 1, "entry": 100.0, "sl": 99.0, "setup_key": "k", "is_signal": True, "source": source,
               "plan": {"kind": "MARKET", "order_type": "BUY", "pending_bars": 0}}
        return db.q1("SELECT demo FROM trades WHERE id=?", (paper.open_trade(res),))["demo"]
    assert trade("EURUSD") == 1 and trade("XAUUSD") == 1 and trade("BTCUSD") == 1
    assert not trade("R_75")                         # synthetic robot signal: learned from, not on the demo account
    assert trade("BOOM1000", source="chat") == 1     # you asked for it in the chat: always on the demo account
    account.update_settings(markets=["forex", "metal", "crypto", "synthetic"])
    assert trade("R_100") == 1
    with pytest.raises(ValueError):
        account.update_settings(markets=[])


def test_watchlist_reports_market_types():
    import os, tempfile
    from sigma import db, watchlist
    db.connect(os.path.join(tempfile.mkdtemp(), "w.db"))
    watchlist.set_symbols(["EURUSD", "R_75", "XAUUSD"])
    w = watchlist.get()
    assert w["classes"] == {"EURUSD": "forex", "R_75": "synthetic", "XAUUSD": "metal"}
    assert all(watchlist.resolve(s).asset_class in ("forex", "metal", "crypto") for s in w["focus"])


def test_fx_metals_preset(monkeypatch):
    import os, tempfile
    from sigma import db, watchlist
    from sigma.data import deriv_catalog
    db.connect(os.path.join(tempfile.mkdtemp(), "p.db"))
    monkeypatch.setattr(deriv_catalog, "catalog", lambda force=False: [
        {"symbol": f"frx{p}", "market": "forex"} for p in ["AUDCAD", "AUDJPY", "EURUSD", "GBPUSD", "USDJPY", "NZDUSD", "AUDNZD", "EURGBP"]]
        + [{"symbol": "R_75", "market": "synthetic_index"}, {"symbol": "frxXAUUSD", "market": "commodities"}])
    w = watchlist.preset("fx_metals")
    assert w["symbols"][:3] == ["EURUSD", "GBPUSD", "USDJPY"]
    assert set(w["symbols"]) == {"EURUSD", "GBPUSD", "USDJPY", "AUDJPY", "NZDUSD", "EURGBP", "USDZAR", "XAUUSD", "XAGUSD"}
    # without the live catalogue: Deriv's known list (AUDCAD, AUDCHF, AUDNZD have none of the four currencies)
    monkeypatch.setattr(deriv_catalog, "catalog", lambda force=False: [])
    syms = watchlist.preset("fx_metals")["symbols"]
    assert len(syms) == 25 + 1 + 2 and "AUDCAD" not in syms and "GBPNOK" in syms
