import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from sigma import db, watchlist
from sigma.data import deriv_catalog, market

IDX = pd.date_range("2026-11-24", periods=50, freq="15min", tz="UTC")


def leg(level, seed):
    c = level * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.001, 50)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0005, "low": np.minimum(o, c) * 0.9995,
                         "close": c, "volume": 0.0}, index=IDX)


@pytest.fixture()
def deriv(monkeypatch):
    feeds = {"frxUSDJPY": leg(155.0, 1), "frxUSDCHF": leg(0.80, 2), "frxUSDCAD": leg(1.40, 3),
             "frxEURUSD": leg(1.16, 4), "frxGBPUSD": leg(1.31, 5), "frxEURJPY": leg(180.0, 6)}
    items = [{"symbol": k, "name": f"{k[3:6]}/{k[6:]}", "market": "forex", "market_label": "Forex"} for k in feeds]
    deriv_catalog._cache.update(t=1e18, items=items + deriv_catalog.computed_pairs(items))
    calls = []

    def fake(code, tf, count=500, end=None):
        calls.append(code)
        if code not in feeds:
            raise market.DataError(f"{code} not offered")
        return feeds[code].iloc[-count:]
    monkeypatch.setattr(market, "deriv_candles", fake)
    monkeypatch.setitem(market._FETCH, "deriv", fake)
    from sigma.data import http
    http._cache.clear()
    yield feeds, calls
    deriv_catalog._cache.update(t=0.0, items=[])


def test_chfjpy_is_usdjpy_over_usdchf(deriv):
    feeds, calls = deriv
    df = market.candles("CHFJPY", "M15", 50)
    assert df.attrs["provider"] == "cross" and "frxCHFJPY" not in calls   # Deriv not asked for a pair it lacks
    expect = feeds["frxUSDJPY"]["close"] / feeds["frxUSDCHF"]["close"]
    assert np.allclose(df["close"].values, expect.values)
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-12).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-12).all()
    # the estimated range never exceeds both legs at their extremes at once
    worst = feeds["frxUSDJPY"]["high"] / feeds["frxUSDCHF"]["low"]
    assert (df["high"] <= worst + 1e-9).all()


def test_eur_cross_and_listed_pairs(deriv):
    feeds, calls = deriv
    eurchf = market.candles("EURCHF", "M15", 50)                          # EURUSD x USDCHF
    assert np.allclose(eurchf["close"], feeds["frxEURUSD"]["close"] * feeds["frxUSDCHF"]["close"])
    calls.clear()
    assert market.candles("EURJPY", "M15", 50).attrs["provider"] == "deriv" and calls == ["frxEURJPY"]


def test_picker_lists_calculated_pairs(deriv):
    names = {i["symbol"]: i for i in deriv_catalog.catalog()}
    assert names["CHFJPY"]["computed"] and "calculated" in names["CHFJPY"]["name"]
    assert "CADJPY" in names and "CADCHF" in names and "EURJPY" not in names     # EURJPY is a real Deriv pair
    assert deriv_catalog.search("chf/jpy")[0]["symbol"] == "CHFJPY"
    assert "CHFJPY" not in deriv_catalog.deriv_codes()


def test_one_time_setup(deriv):
    db.connect(os.path.join(tempfile.mkdtemp(), "s.db"))
    assert watchlist.apply_default_setup_once()
    w = watchlist.get()
    assert w["timeframes"] == ["M15", "H1", "H4"] and w["enabled"]
    assert {"CHFJPY", "CADJPY", "EURJPY", "USDJPY", "XAUUSD", "XAGUSD"} <= set(w["symbols"])
    watchlist.set_timeframes(["H1"])                                         # your later edits are kept
    assert not watchlist.apply_default_setup_once() and watchlist.get()["timeframes"] == ["H1"]


def test_timeframe_fix_after_first_setup(deriv):
    db.connect(os.path.join(tempfile.mkdtemp(), "s2.db"))
    db.kv_set("setup_fx_metals_v1", {"at": 1})                  # the first setup already ran (all timeframes)
    watchlist.set_symbols(["EURUSD", "CHFJPY"])
    watchlist.set_timeframes(["M5", "M15", "M30", "H1", "H4", "D1"])
    assert watchlist.apply_default_setup_once()
    w = watchlist.get()
    assert w["timeframes"] == ["M15", "H1", "H4"] and w["symbols"] == ["EURUSD", "CHFJPY"]   # markets untouched
