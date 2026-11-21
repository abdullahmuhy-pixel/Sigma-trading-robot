"""End-to-end live pipeline with the network replaced by synthetic data."""
import os
import sys
import tempfile
import time
import types

import numpy as np
import pandas as pd
import pytest

from sigma import config, db, engine, learning, paper
from sigma.data import market, news, macro


def synth(tf_secs, n, end=None, seed=0, start=1.1000):
    end = end or time.time()
    idx = pd.to_datetime((np.arange(n) - n) * tf_secs + (end // tf_secs) * tf_secs, unit="s", utc=True)
    rng = np.random.default_rng(seed)
    c = start * np.exp(np.cumsum(rng.normal(0.0002, 0.001, n)))
    o = np.r_[c[0], c[:-1]]
    w = np.abs(rng.normal(0, 0.0006, n)) * c
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + w, "low": np.minimum(o, c) - w,
                       "close": c, "volume": 0.0}, index=idx)
    df.attrs["provider"] = "synthetic"
    return df


@pytest.fixture()
def offline(monkeypatch):
    db.connect(os.path.join(tempfile.mkdtemp(), "p.db"))
    learning._model = None
    from sigma.symbols import tf_seconds
    monkeypatch.setattr(market, "candles", lambda s, tf, count=500: synth(tf_seconds(tf), count, seed=len(s)))
    monkeypatch.setattr(market, "last_price", lambda s: {"USDZAR": 18.0, "EURUSD": 1.1}.get(s) or
                        (_ for _ in ()).throw(market.DataError("x")))
    monkeypatch.setattr(news, "headlines", lambda *a, **k: [
        {"title": "ECB hawkish as inflation accelerates", "summary": "", "published": time.time() - 3600,
         "url": "", "source": "t"}])
    monkeypatch.setattr(news, "calendar", lambda: [
        {"time": time.time() + 600, "currency": "USD", "impact": "High", "title": "Non-Farm Payrolls",
         "forecast": "", "previous": "", "actual": ""}])
    monkeypatch.setattr(macro, "fred_series", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("off")))
    monkeypatch.setattr(macro, "cot", lambda a: None)
    yield


def test_live_analysis_journal_trade_and_learning(offline):
    res = engine.analyze("EURUSD", "H1", source="live")
    assert res["analysis_id"] and res["trade_id"]
    assert res["news"]["score"] > 0                       # hawkish ECB -> EUR positive
    assert res["events"]["blocked"] and not res["is_signal"]  # NFP in 10 minutes blocks signals
    # same bar again -> not duplicated
    again = engine.analyze("EURUSD", "H1", source="live")
    assert not again["analysis_id"]
    assert db.q1("SELECT COUNT(*) n FROM trades")["n"] == 1

    t = db.q1("SELECT * FROM trades")
    assert abs(abs(t["tp"] - t["entry"]) - config.REWARD_R * abs(t["entry"] - t["sl"])) < 1e-9
    # force the trade to have expired and resolve it
    db.update("trades", t["id"], {"opened_at": time.time() - 40 * 3600, "expires_at": time.time() - 3600})
    closed = paper.monitor_once()
    assert len(closed) == 1 and closed[0]["status"] in ("won", "lost", "expired")
    assert learning.model().meta["n_live"] == 1          # model refitted on the result


def test_chat_tools_offline(offline):
    from sigma import agent
    out = agent.run_tool("analyze_market", {"symbol": "XAUUSD", "timeframe": "H4"})
    assert out["symbol"] == "XAU/USD" and out["direction"] in ("LONG", "SHORT")
    assert set(out["component_scores"]) == set(learning.BASE)
    rep = agent.run_tool("learning_report", {})
    assert "weights" in rep and rep["trained_on"]["live_trades"] == 0
    scan = agent.run_tool("scan_watchlist", {"timeframe": "H1", "symbols": ["EURUSD", "R_75"]})
    assert len(scan) == 2
    syn = agent.run_tool("analyze_market", {"symbol": "R_75", "timeframe": "M15"})
    assert syn["fundamental"]["score"] == 0
    # 1% of 10,000 USD = 100 USD risk over 50 pips -> 20,000 units = 0.2 lots
    ps = agent.position_size("EURUSD", 10_000, 1, 1.1000, 1.0950, "USD")
    assert ps["units"] == pytest.approx(20_000) and ps["lots"] == pytest.approx(0.2)
    # ZAR account: 1 USD = 18 ZAR -> 1% of 180,000 ZAR = 1,800 ZAR = 100 USD -> same 0.2 lots
    ps = agent.position_size("EURUSD", 180_000, 1, 1.1000, 1.0950, "ZAR")
    assert ps["lots"] == pytest.approx(0.2)


def test_server_routes_with_stub_fastapi(offline, monkeypatch):
    """FastAPI is not installed in this sandbox: stub it and call the route functions."""
    fa = types.ModuleType("fastapi")

    class App:
        def __init__(self, **k): pass
        def _d(self, *a, **k): return lambda f: f
        get = post = on_event = middleware = _d

    class HTTPException(Exception):
        def __init__(self, code, detail=""): super().__init__(detail)

    class JSONResponse(dict):
        def __init__(self, content, status_code=200):
            super().__init__(content)
            self.cookies = {}
        def set_cookie(self, k, v, **kw): self.cookies[k] = v
    fa.FastAPI, fa.HTTPException, fa.Request = App, HTTPException, object
    resp = types.ModuleType("fastapi.responses")
    resp.FileResponse = lambda p, **k: p
    resp.JSONResponse = JSONResponse
    resp.RedirectResponse = lambda url: url
    monkeypatch.setitem(sys.modules, "fastapi", fa)
    monkeypatch.setitem(sys.modules, "fastapi.responses", resp)
    sys.modules.pop("sigma.server", None)
    from sigma import server
    a = server.analyze("GBPUSD", "M15")
    assert a["symbol"] == "GBP/USD"
    assert len(server.candles("GBPUSD", "M15", 50)) == 50
    assert isinstance(server.signals(all=True), list) and server.signals(all=True)
    assert isinstance(server.trades(), list)
    p = server.performance()
    assert p["live_all"]["open"] >= 1
    assert "weights" in server.learning_state()
    st = server.status()
    assert st["tz"] == "Africa/Johannesburg"
    assert server.chat(server.ChatIn(message="hi"))["reply"]  # no key -> friendly message

    # chat history sync: messages written by one device are visible to another via after_id
    db.insert("chat", {"session": "default", "created_at": time.time(), "role": "user", "content": "from phone"})
    first = server.chat_history()
    newer = server.chat_history(after_id=first[-2]["id"]) if len(first) > 1 else first
    assert newer[-1]["content"] == "from phone"
    # login
    monkeypatch.setattr(config, "APP_PASSWORD", "s3cret")
    assert not server.is_authorised(None)
    r = server.login(server.LoginIn(password="s3cret"))
    assert server.is_authorised(r.cookies[server.COOKIE])
    with pytest.raises(Exception):
        server.login(server.LoginIn(password="wrong"))
    assert "/sw.js" in server.PUBLIC and "/api/chat" not in server.PUBLIC


def test_review_user_trade(offline):
    from sigma import review
    price = market.candles("EURUSD", "H1", 400)["close"].iloc[-1]
    out = review.review_trade("EURUSD", "H1", "buy", float(price), float(price) - 0.0030, float(price) + 0.0060)
    assert out["your_trade"]["reward_risk"] == pytest.approx(2.0)
    assert out["breakeven_win_rate"] == pytest.approx(1 / 3, abs=1e-3)
    p, rr = out["p_win_estimate"], 2.0
    assert out["expected_R_after_costs"] <= p * rr - (1 - p) + 1e-6
    assert out["verdict"].split()[0] in ("GOOD", "CAUTION", "AVOID")
    t = db.q1("SELECT * FROM trades WHERE origin='user'")
    assert t and t["entry"] == pytest.approx(price)
    # wrong-side stop is rejected
    assert "error" in review.review_trade("EURUSD", "H1", "sell", 1.1, 1.09, 1.08)
    # far-away limit order stays pending until price reaches it, then expires -> cancelled
    far = review.review_trade("EURUSD", "H1", "buy", float(price) * 0.5, float(price) * 0.49, float(price) * 0.52)
    assert far["your_trade"]["order_type"] == "limit"
    pid = db.q1("SELECT id FROM trades WHERE status='pending'")["id"]
    db.update("trades", pid, {"expires_at": time.time() - 1})
    paper.activate_pending()
    assert db.q1("SELECT status FROM trades WHERE id=?", (pid,))["status"] == "cancelled"
    # user trades never enter the robot's learning data
    assert all(r["source"] in ("live", "backtest") for r in learning.training_rows())
    n_user_closed = db.q1("SELECT COUNT(*) n FROM trades WHERE origin='user' AND r_multiple IS NOT NULL")["n"]
    assert len(learning.training_rows()) == db.q1(
        "SELECT COUNT(*) n FROM trades WHERE origin='robot' AND status!='open' AND r_multiple IS NOT NULL")["n"]
    _ = n_user_closed


def test_adjusted_probability():
    from sigma import review
    base = 1 / (1 + config.REWARD_R)
    # no edge stays no edge at any reward:risk
    for rr in (1, 2, 3):
        assert review.adjusted_probability(base, rr) == pytest.approx(1 / (1 + rr))
    # an edge is kept, and a bigger target lowers the win probability
    assert review.adjusted_probability(0.45, 1) > review.adjusted_probability(0.45, 3)


def test_postgres_translation():
    assert db.to_pg("SELECT * FROM t WHERE a=? AND b=?") == "SELECT * FROM t WHERE a=%s AND b=%s"
    assert db.to_pg("INSERT OR IGNORE INTO t (a) VALUES (?) RETURNING id") == \
        "INSERT INTO t (a) VALUES (%s) ON CONFLICT DO NOTHING RETURNING id"
    assert db.to_pg("INSERT OR IGNORE INTO t (a) VALUES (?)").endswith("ON CONFLICT DO NOTHING")
    sch = db.pg_schema()
    assert "AUTOINCREMENT" not in sch and " REAL" not in sch and "BIGSERIAL PRIMARY KEY" in sch
    assert "DOUBLE PRECISION" in sch


def test_closed_market_never_signals_or_trades(offline, monkeypatch):
    from sigma.symbols import tf_seconds
    friday = time.time() - 40 * 3600   # last candle 40 h ago, like a Saturday afternoon
    monkeypatch.setattr(market, "candles",
                        lambda s, tf, count=500: synth(tf_seconds(tf), count, end=friday, seed=3))
    assert engine.market_closed(friday - 900, "M15") > 0
    assert engine.market_closed(time.time() - 1800, "M15") == 0     # one fresh bar ago: open
    live = engine.analyze("EURUSD", "M15", source="live")
    assert live["market_closed"] and not live["is_signal"] and "analysis_id" not in live
    chat = engine.analyze("EURUSD", "M15", source="chat")
    assert chat["analysis_id"] and not chat.get("trade_id")
    assert db.q1("SELECT COUNT(*) n FROM trades")["n"] == 0


def test_every_deriv_instrument_resolves(monkeypatch):
    from sigma.data import deriv_catalog, market as mk
    from sigma.symbols import resolve
    new_api = [  # new Deriv API field names
        {"underlying_symbol": "R_75", "underlying_symbol_name": "Volatility 75 Index", "market": "synthetic_index",
         "submarket": "random_index", "exchange_is_open": 1, "is_trading_suspended": 0, "pip_size": 0.0001},
        {"underlying_symbol": "1HZ150V", "underlying_symbol_name": "Volatility 150 (1s) Index", "market": "synthetic_index",
         "submarket": "random_index", "exchange_is_open": 1, "is_trading_suspended": 0, "pip_size": 0.01},
        {"underlying_symbol": "JD50", "underlying_symbol_name": "Jump 50 Index", "market": "synthetic_index",
         "submarket": "jump_index", "exchange_is_open": 1, "is_trading_suspended": 0, "pip_size": 0.01}]
    legacy = [  # legacy field names
        {"symbol": "cryXRPUSD", "display_name": "XRP/USD", "market": "cryptocurrency", "submarket": "non_stable_coin",
         "exchange_is_open": 1, "pip": 0.0001},
        {"symbol": "frxXAUUSD", "display_name": "Gold/USD", "market": "commodities", "submarket": "metals",
         "exchange_is_open": 0, "pip": 0.01},
        {"symbol": "WLDUSD", "display_name": "USD Basket", "market": "basket_index", "submarket": "forex_basket",
         "exchange_is_open": 1, "pip": 0.001},
        {"symbol": "OTC_AEX", "display_name": "Netherlands 25", "market": "indices", "submarket": "europe_OTC",
         "exchange_is_open": 0, "pip": 0.01}]
    monkeypatch.setattr(mk, "deriv_active_symbols", lambda: new_api + legacy)
    deriv_catalog._cache.update(t=0, items=[])
    items = deriv_catalog.catalog(force=True)
    assert len(items) == 7 and all(i["symbol"] and i["name"] for i in items)
    assert resolve("V75").name == "R_75"
    assert resolve("Volatility 150 (1s) Index").deriv == "1HZ150V"
    assert resolve("v150 1s").asset_class == "synthetic"
    assert resolve("Jump 50").name == "JD50"
    x = resolve("cryXRPUSD")
    assert x.asset_class == "crypto" and x.deriv == "cryXRPUSD"
    assert resolve("frxXAUUSD").asset_class == "metal" and resolve("frxXAUUSD").name == "XAUUSD"
    b = resolve("WLDUSD")
    assert b.asset_class == "basket" and not b.has_fundamentals
    assert resolve("OTC_AEX").asset_class == "index"
    assert [i["symbol"] for i in deriv_catalog.search("jump")] == ["JD50"]
    assert {i["market_label"] for i in deriv_catalog.search(market="cryptocurrency")} == {"Crypto"}
    # crypto prices now come from Deriv first, so they match the Deriv platform
    from sigma.data.market import _routes
    assert _routes(resolve("BTCUSD"))[0] == ("deriv", "cryBTCUSD")
