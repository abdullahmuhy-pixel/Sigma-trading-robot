import os
import tempfile

import numpy as np
import pytest

from sigma import config, db, learning
from sigma.data import positioning as pos


@pytest.fixture()
def fresh(monkeypatch):
    db.connect(os.path.join(tempfile.mkdtemp(), "c.db"))
    pos._fail_until.clear()
    pos._lock_state.clear()
    from sigma.data import http
    http._cache.clear()
    yield


class Resp:
    def __init__(self, js, status=200, headers=None, text=""):
        self._js, self.status_code, self.headers, self.text = js, status, headers or {}, text or str(js)

    def json(self):
        return self._js


def test_contrarian_score():
    assert pos.contrarian_score(50) == 0
    assert pos.contrarian_score(75) == -1 and pos.contrarian_score(90) == -1     # crowded long -> bearish
    assert pos.contrarian_score(30) == pytest.approx(0.8)                         # crowded short -> bullish


def test_myfxbook_flow(fresh, monkeypatch):
    monkeypatch.setattr(config, "MYFXBOOK_EMAIL", "me@x.com")
    monkeypatch.setattr(config, "MYFXBOOK_PASSWORD", "pw")
    calls = []

    def get(url, params=None, timeout=None):
        calls.append(url)
        if url.endswith("login.json"):
            return Resp({"error": False, "session": f"S{len(calls)}"})
        if params["session"] == "S1":          # first session has expired
            return Resp({"error": True, "message": "Invalid session"})
        return Resp({"error": False, "symbols": [
            {"name": "EURUSD", "shortPercentage": 28, "longPercentage": 72, "avgShortPrice": "1.0950",
             "avgLongPrice": "1.1040", "longPositions": 9000, "shortPositions": 3500},
            {"name": "XAUUSD", "shortPercentage": 60, "longPercentage": 40}]})
    monkeypatch.setattr(pos._session, "get", get)
    out = pos.analyze("EURUSD")
    assert out["long_pct"] == 72 and out["score"] == pytest.approx(-0.88)
    assert "crowded long" in out["notes"][0] and "1.104" in out["notes"][1]
    assert sum(u.endswith("login.json") for u in calls) == 2                       # re-logged in once
    n = len(calls)
    assert pos.analyze("XAUUSD")["score"] == pytest.approx(0.4)
    assert len(calls) == n                                                          # whole table cached


def test_ig_flow(fresh, monkeypatch):
    for k, v in (("IG_API_KEY", "k"), ("IG_USERNAME", "u"), ("IG_PASSWORD", "p")):
        monkeypatch.setattr(config, k, v)
    monkeypatch.setattr(pos._session, "post", lambda url, headers=None, json=None, timeout=None:
                        Resp({}, 200, {"CST": "c", "X-SECURITY-TOKEN": "x"}))
    seen = []

    def get(url, headers=None, params=None, timeout=None):
        seen.append(url)
        assert headers["CST"] == "c" and headers["X-IG-API-KEY"] == "k"
        if url.endswith("/markets"):
            return Resp({"markets": [{"epic": "CS.D.EURUSD.CFD.IP"}]})
        if "/markets/" in url:
            return Resp({"instrument": {"marketId": "EURUSD"}})
        return Resp({"marketId": "EURUSD", "longPositionPercentage": 35.0, "shortPositionPercentage": 65.0})
    monkeypatch.setattr(pos._session, "get", get)
    out = pos.analyze("EURUSD")
    assert out["sources"]["ig"]["long_pct"] == 35 and out["score"] == pytest.approx(0.6)
    assert db.kv_get("ig_market_ids")["EURUSD"] == "EURUSD"                       # id remembered


def test_nothing_connected_is_neutral(fresh):
    assert pos.analyze("EURUSD")["score"] == 0.0
    assert pos.analyze("R_75") == {"score": 0.0, "sources": {}, "notes": []}


def test_failure_backs_off(fresh, monkeypatch):
    monkeypatch.setattr(config, "MYFXBOOK_EMAIL", "me@x.com")
    monkeypatch.setattr(config, "MYFXBOOK_PASSWORD", "pw")
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise ConnectionError("down")
    monkeypatch.setattr(pos._session, "get", boom)
    assert pos.analyze("EURUSD")["score"] == 0 and pos.analyze("EURUSD")["score"] == 0
    assert len(calls) == 1


def test_learned_weights_survive_new_factor():
    old_features = [f for f in learning.FEATURES if not f.startswith("crowd")]
    th = np.linspace(-1, 1, len(old_features))
    m = learning.Model.from_json({"features": old_features, "theta": th.tolist(), "bias": 0.1,
                                  "meta": {"n_live": 50}})
    w = m.weights()
    assert w["trend"] == pytest.approx(th[old_features.index("trend")], abs=1e-4)
    assert w["news@ranging"] == pytest.approx(th[old_features.index("news@ranging")], abs=1e-4)
    assert w["crowd"] == 0.3 and w["crowd@trending"] == 0.0                         # new factor starts at its prior
    assert m.meta["n_live"] == 50 and m.bias == 0.1


def test_engine_uses_crowd(monkeypatch):
    import pandas as pd
    from sigma import engine
    idx = pd.date_range("2026-11-01", periods=300, freq="15min", tz="UTC")
    c = 1.1 + np.cumsum(np.random.default_rng(1).normal(0, 0.0005, 300))
    df = pd.DataFrame({"open": c, "high": c + 0.0004, "low": c - 0.0004, "close": c, "volume": 0.0}, index=idx)
    db.connect(os.path.join(tempfile.mkdtemp(), "e.db"))
    learning._model = None
    a = engine.evaluate("EURUSD", "M15", df)
    b = engine.evaluate("EURUSD", "M15", df, crowd={"score": -1.0, "notes": ["crowded long"]})
    assert b["components"]["crowd"] == -1.0 and "crowded long" in b["notes"]
    assert b["p_short"] > a["p_short"] and b["p_long"] < a["p_long"]               # crowd long tilts toward sell
