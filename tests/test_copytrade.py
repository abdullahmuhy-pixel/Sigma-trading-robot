import os
import tempfile
import time

import pytest

from sigma import copytrade as ct, db
from sigma.autobot import brokers
from sigma.autobot.sizing import pnl


@pytest.fixture()
def env(monkeypatch):
    db.connect(os.path.join(tempfile.mkdtemp(), "copy.db"))
    ct._log.clear()
    price = {"v": 1.1000}
    monkeypatch.setattr(brokers.market, "last_price", lambda s, max_age=2.0: price["v"])
    ct.update_settings(account="paper", paper_start_balance=1000, risk_pct=1.0, running=True,
                       started_at=time.time() - 60)
    b = ct.broker()
    return price, b


def paper_trade(direction=1, entry=1.1000, sl=1.0980, tp=1.1040, opened=None, status="open", **kw):
    return db.insert("trades", {"analysis_id": 1, "symbol": "EURUSD", "timeframe": "M15", "source": "live",
                                "direction": direction, "entry": entry, "sl": sl, "tp": tp,
                                "opened_at": opened or time.time() - 20, "expires_at": time.time() + 9999,
                                "status": status, "setup_key": "x", "order_type": "BUY", "demo": 1, **kw})


def test_trail_levels():
    s = ct.settings()
    assert ct.trail_stop_r(0.9, s) is None
    assert ct.trail_stop_r(1.0, s) == pytest.approx(ct.BE_LOCK_R)      # +1R: breakeven (+costs)
    assert ct.trail_stop_r(1.8, s) == pytest.approx(0.8)               # trails 1R behind the best


def test_copy_sizes_from_real_balance_and_trails(env):
    price, b = env
    tid = paper_trade()
    ct.step(b)
    c = ct.open_copies("paper")[0]
    assert c["trade_id"] == tid and c["sl_price"] == 1.0980 and c["tp_price"] == 1.1040
    # the loss at the stop is 1% of the $1,000 test balance, whatever the paper trade risked
    assert c["sl_usd"] == pytest.approx(10.0, abs=0.01)
    assert pnl(c["exposure"], 1, c["entry"], 1.0980, c["cost"]) == pytest.approx(-10.0, abs=0.02)
    assert c["stake"] >= c["sl_usd"]
    price["v"] = 1.10245                                              # +1.225R
    ct.step(b)
    c = ct.open_copies("paper")[0]
    assert c["best_r"] == pytest.approx(1.225, abs=1e-3) and c["trail_r"] == pytest.approx(0.225, abs=1e-3)
    price["v"] = 1.1036                                               # +1.8R: stop now +0.8R
    ct.step(b)
    assert ct.open_copies("paper")[0]["trail_r"] == pytest.approx(0.8, abs=1e-3)
    price["v"] = 1.1015                                               # back to +0.75R: trailing stop hits
    ct.step(b)
    done = db.q1("SELECT * FROM copy_trades WHERE status='closed'")
    assert done["close_reason"].startswith("trailing stop") and done["pnl"] > 0
    assert b.balance() == pytest.approx(1000 + done["pnl"], abs=1e-3)
    # the paper trade itself is untouched
    assert db.q1("SELECT status, sl FROM trades WHERE id=?", (tid,)) == {"status": "open", "sl": 1.0980}


def test_take_profit_and_stop(env):
    price, b = env
    paper_trade()
    ct.step(b)
    price["v"] = 1.1041
    ct.step(b)
    assert db.q1("SELECT close_reason FROM copy_trades")["close_reason"] == "take profit"
    paper_trade(direction=-1, entry=1.1041, sl=1.1061, tp=1.1001)
    price["v"] = 1.1041
    ct.step(b)
    price["v"] = 1.1061                                               # exactly at the stop
    ct.step(b)
    last = db.q1("SELECT * FROM copy_trades WHERE direction=-1")
    # 1% of the balance after the first win, costs included
    assert last["close_reason"] == "stop loss" and last["pnl"] == pytest.approx(-last["sl_usd"], abs=0.01)
    assert last["sl_usd"] == pytest.approx(0.01 * (1000 + db.q1("SELECT pnl FROM copy_trades WHERE id=1")["pnl"]), abs=0.01)


def test_skips_late_moved_and_filtered(env, monkeypatch):
    price, b = env
    ct.update_settings(started_at=time.time() - 3600)
    paper_trade(opened=time.time() - 400)                             # seen too late
    paper_trade(entry=1.0990, sl=1.0970, tp=1.1030)                    # price already +0.5R from entry
    ct.step(b)
    notes = [r["note"] for r in db.q("SELECT note FROM copy_trades WHERE status='skipped' ORDER BY id")]
    assert "too late" in notes[0] and "already moved" in notes[1]
    assert not ct.open_copies("paper")
    # robot-signals-only mode ignores trades from your own chat analyses
    db.insert("analyses", {"id": 77, "symbol": "EURUSD", "timeframe": "M15", "created_at": time.time(),
                           "bar_time": time.time(), "source": "chat", "direction": 1, "probability": 0.6,
                           "is_signal": 0, "entry": 1.1, "sl": 1.09, "tp": 1.12, "price": 1.1})
    ct.update_settings(copy="signals")
    paper_trade(analysis_id=77)
    assert ct.candidates(ct.settings()) == []


def test_closes_when_paper_trade_closes_and_respects_limits(env):
    price, b = env
    ct.update_settings(max_open=1)
    tid = paper_trade()
    paper_trade()
    ct.step(b)
    assert len(ct.open_copies("paper")) == 1
    assert "already open" in db.q1("SELECT note FROM copy_trades WHERE status='skipped'")["note"]
    db.update("trades", tid, {"status": "expired"})
    ct.step(b)
    assert db.q1("SELECT close_reason FROM copy_trades WHERE trade_id=?", (tid,))["close_reason"] == "paper trade closed (expired)"


def test_daily_loss_guard_and_live_gate(env):
    price, b = env
    db.insert("copy_trades", {"trade_id": 0, "account": "paper", "symbol": "EURUSD", "direction": 1,
                              "opened_at": time.time() - 99, "closed_at": time.time(), "status": "closed", "pnl": -35.0})
    paper_trade()
    ct.step(b)
    assert "daily loss limit" in db.q1("SELECT note FROM copy_trades WHERE status='skipped'")["note"]
    with pytest.raises(ValueError):
        ct.update_settings(risk_pct=5)
    s = ct.update_settings(account="deriv", running=True)
    assert s["running"] is False                                   # real account needs the typed confirmation
    with pytest.raises(ValueError):
        ct.start()
