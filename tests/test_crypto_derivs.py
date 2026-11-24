import os
import tempfile

import pytest

from sigma import db
from sigma.data import crypto_derivs as cd, http, positioning


@pytest.fixture()
def fresh(monkeypatch):
    db.connect(os.path.join(tempfile.mkdtemp(), "d.db"))
    cd._fail_until.clear()
    positioning._fail_until.clear()
    http._cache.clear()
    monkeypatch.setattr(cd, "okx_long_share", lambda coin: None)
    yield


HL = [{"universe": [{"name": "BTC"}, {"name": "ETH"}, {"name": "SOL"}]},
      [{"funding": "0.0000125", "openInterest": "30000", "markPx": "90000", "prevDayPx": "88000"},
       {"funding": "0.00008", "openInterest": "500000", "markPx": "3000", "prevDayPx": "3000"},
       {"funding": "-0.00006", "openInterest": "1000", "markPx": "150", "prevDayPx": "150"}]]


def test_funding_score():
    assert cd.funding_score(0.10 / 8760) == pytest.approx(0)            # normal funding -> neutral
    assert cd.funding_score(0.70 / 8760) == pytest.approx(-1)           # longs paying 70%/yr -> crowded long
    assert cd.funding_score(-0.50 / 8760) == pytest.approx(1)           # shorts paying -> crowded short
    assert cd.funding_score(0.0000125) == pytest.approx(-(0.0000125 * 8760 - 0.10) / 0.60)


def test_parse_hyperliquid():
    d = cd.parse_hyperliquid(HL)
    assert d["BTC"]["funding_hourly"] == 0.0000125 and d["BTC"]["open_interest_usd"] == 30000 * 90000


def test_parse_kraken():
    js = {"tickers": [{"symbol": "PF_XBTUSD", "markPrice": 90000, "fundingRate": 1.8, "openInterest": 1500},
                      {"symbol": "PI_XBTUSD", "markPrice": 90000, "fundingRate": 5},       # inverse: skipped
                      {"symbol": "PF_ETHUSD", "markPrice": 3000, "fundingRate": 0.03, "suspended": True},
                      {"symbol": "in_xbtusd", "last": 90000}]}
    d = cd.parse_kraken(js)
    assert list(d) == ["BTC"] and d["BTC"]["funding_hourly"] == pytest.approx(1.8 / 90000)
    # real Kraken ticker (23 Nov 2026): the relative rate is used directly and matches absolute / price
    real = {"tickers": [{"symbol": "PF_XBTUSD", "markPrice": 61405.81371692881, "fundingRate": 0.4227187708575,
                         "relativeFundingRate": 0.000006888375, "openInterest": 2005.6476, "suspended": False}]}
    k = cd.parse_kraken(real)["BTC"]
    assert k["funding_hourly"] == 0.000006888375
    assert 0.4227187708575 / 61405.81371692881 == pytest.approx(k["funding_hourly"], rel=1e-3)


def test_analyze_and_crowd_factor(fresh, monkeypatch):
    monkeypatch.setattr(cd, "hyperliquid", lambda: cd.parse_hyperliquid(HL))
    monkeypatch.setattr(cd, "kraken", lambda: (_ for _ in ()).throw(AssertionError("Kraken not needed")))
    eth = cd.analyze("ETHUSD")
    assert eth["funding_apr_pct"] == pytest.approx(70.1, abs=0.1) and eth["score"] == -1.0
    assert "crowded long" in eth["notes"][0]
    sol = cd.analyze("SOLUSD")
    assert sol["score"] == 1.0 and "crowded short" in sol["notes"][0]
    assert cd.analyze("EURUSD") is None and cd.analyze("R_75") is None
    # folded into the crowd factor (no Myfxbook/IG set up)
    c = positioning.analyze("ETHUSD")
    assert c["score"] == -1.0 and "futures" in c["sources"]


def test_kraken_fallback_and_oi_history(fresh, monkeypatch):
    monkeypatch.setattr(cd, "hyperliquid", lambda: {})
    oi = {"v": 1000.0}
    monkeypatch.setattr(cd, "kraken", lambda: {"BTC": {"funding_hourly": 0.0, "open_interest": oi["v"], "mark": 100.0}})
    t0 = 1_800_000_000
    a = cd.analyze("BTCUSD", now=t0)
    assert a["funding_source"] == "Kraken" and a["oi_change_4h_pct"] is None
    oi["v"] = 1100.0
    b = cd.analyze("BTCUSD", now=t0 + 4 * 3600)
    assert b["oi_change_4h_pct"] == pytest.approx(10.0)


def test_all_sources_down_is_neutral(fresh, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("blocked")
    monkeypatch.setattr(cd._session, "post", boom)
    monkeypatch.setattr(cd._session, "get", boom)
    assert cd.analyze("BTCUSD") is None
    assert positioning.analyze("BTCUSD")["score"] == 0.0
    assert cd.hyperliquid() == {}                                         # backed off, no new request
