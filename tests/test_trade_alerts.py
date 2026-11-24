import os
import tempfile
import time

import pandas as pd
import pytest

from sigma import db, paper, push, trade_alerts as ta


@pytest.fixture()
def env(monkeypatch):
    db.connect(os.path.join(tempfile.mkdtemp(), "ta.db"))
    sent = []
    monkeypatch.setattr(push, "notify", lambda title, body, url="/", tag=None: sent.append((title, body)))
    monkeypatch.setattr(ta, "run", lambda fn, *a: fn(*a))                 # synchronous for tests
    from sigma import agent
    monkeypatch.setattr(agent, "fx_rate", lambda a, b: 1.0 if a == b else {"JPYUSD": 1 / 155.0}.get(a + b, 1.0))
    ta.update_settings(real_balance=1000, currency="USD", risk_pct=1.0)
    return sent


def background_trade(sym="EURUSD", source="live", kind="MARKET", entry=1.1000, sl=1.0980, order="BUY"):
    aid = db.insert("analyses", {"created_at": time.time(), "bar_time": time.time() - 60, "symbol": sym,
                                 "timeframe": "H1", "source": source, "direction": 1, "price": 1.1010})
    res = {"analysis_id": aid, "symbol": sym, "timeframe": "H1", "bar_time": time.time() - 60, "direction": 1,
           "entry": entry, "sl": sl, "setup_key": "k", "is_signal": True, "source": source, "probability": 0.58,
           "price": 1.1010, "take_profits": [{"name": "TP1", "price": 1.1020, "p_reach": 0.6}],
           "plan": {"kind": kind, "order_type": order, "pending_bars": 4 if kind != "MARKET" else 0}}
    return paper.open_trade(res, entry_price=entry if kind == "MARKET" else None)


def test_lot_sizes():
    db.connect(os.path.join(tempfile.mkdtemp(), "l.db"))
    ta.update_settings(real_balance=1000, currency="USD", risk_pct=1.0)
    from sigma import agent
    l = ta.lot_size("EURUSD", 1.1000, 1.0980)                  # $10 risk / (20 pips x $10 per pip per lot)
    assert l["lots"] == 0.05 and l["risk_amount"] == 10.0
    g = ta.lot_size("XAUUSD", 2650.0, 2640.0)                  # $10 / ($10 move x 100 oz) = 0.01
    assert g["lots"] == 0.01 and not g.get("too_small")
    tiny = ta.lot_size("XAUUSD", 2650.0, 2610.0)               # needs 0.0025 lots: the 0.01 minimum risks more
    assert tiny["too_small"] and tiny["actual_risk"] == pytest.approx(40.0)


def test_background_trade_alerts_with_lot_size(env):
    sent = env
    tid = background_trade()
    assert len(sent) == 1
    title, body = sent[0]
    assert title.startswith("📈 Trade taken: BUY EUR/USD (H1)")
    assert "Stop loss 1.09800" in body and "TP1 1.10200 (60%)" in body and "Your lot size: 0.05" in body
    a = db.q1("SELECT * FROM alerts WHERE kind='trade_opened'")
    assert a["symbol"] == "EURUSD" and "lot size" in a["body"]
    assert db.q1("SELECT content FROM chat")["content"].startswith("**📈 Trade taken")
    assert db.q1("SELECT demo FROM trades WHERE id=?", (tid,))["demo"] == 1


def test_chat_trades_and_synthetics_are_not_announced(env):
    sent = env
    background_trade(source="chat")                            # you asked for it - no alert
    background_trade(sym="R_75", entry=40000, sl=39900)        # synthetic: not on the demo account, no alert
    assert sent == []


def test_fill_breakeven_and_close(env, monkeypatch):
    sent = env
    tid = background_trade(kind="LIMIT", order="BUY LIMIT", entry=1.0995, sl=1.0975)
    assert sent[-1][0].startswith("⏳ Order placed: BUY LIMIT")
    t = db.q1("SELECT * FROM trades WHERE id=?", (tid,))
    idx = pd.date_range(pd.Timestamp(t["opened_at"], unit="s", tz="UTC").ceil("min"), periods=6, freq="1min")
    m1 = pd.DataFrame({"open": [1.1005, 1.0996, 1.1000, 1.1012, 1.1020, 1.1030],
                       "high": [1.1006, 1.1000, 1.1010, 1.1016, 1.1020, 1.1018],     # up to +1.25R, below TP
                       "low": [1.1000, 1.0994, 1.0998, 1.1008, 1.1012, 1.1010],
                       "close": [1.1001, 1.0999, 1.1009, 1.1014, 1.1016, 1.1012], "volume": 0.0}, index=idx)
    from sigma.data import market
    monkeypatch.setattr(market, "candles", lambda s, tf, n=500: m1)
    monkeypatch.setattr(time, "time", lambda: idx[-1].timestamp() + 30)
    paper.activate_pending()
    assert sent[-1][0].startswith("✅ Filled: BUY LIMIT EUR/USD") and "lot size" in sent[-1][1]
    monkeypatch.setattr(paper.learning, "refit_and_save", lambda: None)
    paper.monitor_once()
    titles = [s[0] for s in sent]
    assert any(x.startswith("🛡 EUR/USD is +1R: move your stop to entry") for x in titles)
    paper.monitor_once()                                       # announced once only
    assert sum(1 for s in sent if s[0].startswith("🛡")) == 1


def test_switches(env):
    sent = env
    ta.update_settings(opened=False)
    background_trade()
    assert sent == []
    ta.update_settings(opened=True, enabled=False)
    background_trade()
    assert sent == []
    with pytest.raises(ValueError):
        ta.update_settings(risk_pct=9)
