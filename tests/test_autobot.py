import os
import tempfile
import time

import pytest

from sigma import db
from sigma.autobot import bot, brokers, review
from sigma.autobot.sizing import auto_take_profit, max_scalp_positions, pnl, size_position


@pytest.fixture()
def fresh(monkeypatch):
    db.connect(os.path.join(tempfile.mkdtemp(), "bot.db"))
    bot._log.clear()
    yield


# ------------------------------------------------------------------------------- sizing ---
def test_size_position_numbers():
    # V75 at 40,000; target 24 points away, stop 36 points away; want $0.50 NET profit, max loss $1
    s = size_position(40000, 24, 36, 0.50, 1.0, [20, 40, 60, 100, 200, 400], 1.0, 0.0002)
    expo = 0.50 / (24 / 40000 - 0.0002)                                   # 1,250
    assert s.exposure == pytest.approx(expo / (expo * (36 / 40000 + 0.0002)), rel=1e-3)  # scaled to the $1 cap
    assert expo * (36 / 40000 + 0.0002) > 1.0                       # unscaled stop would cost $1.375
    assert s.sl_usd <= 1.0
    assert s.multiplier in (200, 400) and s.stake >= s.sl_usd * 1.15 - 0.01
    # profit at the target price after costs is exactly the dollar target; loss at the stop is the cap
    assert pnl(s.exposure, 1, 40000, 40024, s.cost_usd) == pytest.approx(s.tp_usd, abs=2e-3)
    assert pnl(s.exposure, 1, 40000, 39964, s.cost_usd) == pytest.approx(-s.sl_usd, abs=2e-3)
    # costs that eat half the move or more -> no trade
    assert size_position(40000, 4, 36, 0.50, 1.0, [400], 1.0, 0.0002) is None


def test_size_position_respects_max_loss_and_min_stake():
    # the stop would cost $3 but only $1 is allowed: everything scales down by a third
    s = size_position(1.08, 0.0002, 0.0006, 1.0, 1.0, [50, 100, 500], 1.0, 0.0)
    assert s.sl_usd == pytest.approx(1.0) and s.tp_usd == pytest.approx(1.0 / 3, abs=0.01)  # no costs here
    # a tiny trade whose stake would fall under the minimum is refused
    assert size_position(1.08, 0.01, 0.01, 0.05, 0.05, [500], 5.0, 0.0) is None


def test_account_scaling():
    assert auto_take_profit(100) == 0.50 and auto_take_profit(500) == 1.00 and auto_take_profit(10000) == 10.0
    assert max_scalp_positions(100) == 2 and max_scalp_positions(500) == 6 and max_scalp_positions(10000) == 10


# --------------------------------------------------------------------------------- exits ---
def test_exit_rules():
    now = time.time()
    t = {"tp_usd": 1.0, "sl_usd": 1.5, "opened_at": now - 60, "mode": "scalp", "max_profit": 0.0}
    assert bot.exit_reason(t, 1.0, now) == "take profit"
    assert bot.exit_reason(t, -1.5, now) == "stop loss"
    assert bot.exit_reason(t, 0.3, now) is None
    # reached 70% of the target, now back to 10%: close and keep the little that is left
    assert bot.exit_reason({**t, "max_profit": 0.7}, 0.10, now).startswith("protect profit")
    assert bot.exit_reason({**t, "max_profit": 0.7}, 0.30, now) is None
    assert bot.exit_reason({**t, "opened_at": now - 16 * 60}, 0.2, now) == "time exit in profit"
    assert bot.exit_reason({**t, "opened_at": now - 31 * 60}, -0.2, now) == "time exit"
    sh = {**t, "mode": "short", "breakeven": 1, "opened_at": now - 600}
    assert bot.exit_reason(sh, 0.01, now).startswith("breakeven")


# ---------------------------------------------------------------------------------- risk ---
def test_risk_rules(fresh):
    s = bot.update_settings(daily_target=2.0, daily_loss_pct=3.0, max_drawdown_pct=8.0)
    r = bot.risk_check(s, None, 100.0, 0.0)
    assert r["can_open"] and r["risk_factor"] == 1.0 and r["day_start_equity"] == 100.0
    r = bot.risk_check(s, None, 96.9, 0.0)                     # -3.1% on the day
    assert not r["can_open"] and r["close_all"] and "daily loss" in r["reason"]
    st = bot.state()
    st["day"] = "2000-01-01"                                      # next day: the daily halt clears
    bot.save_state(st)
    r = bot.risk_check(s, None, 96.9, 0.0)
    assert r["can_open"]
    s = bot.update_settings(daily_loss_pct=10.0)              # so only the drawdown rule can trigger here
    r = bot.risk_check(s, None, 91.9, 0.0)                     # 8.1% below the 100 peak
    assert not r["can_open"] and "drawdown" in r["reason"]
    with pytest.raises(ValueError):
        bot.start()                                              # needs a reset after a drawdown stop
    bot.save_state({})


def test_target_and_cooldown(fresh):
    s = bot.update_settings(daily_target=2.0)
    now = time.time()
    for p in (0.6, 0.5):
        db.insert("bot_trades", {"account": "paper", "mode": "scalp", "symbol": "R_75", "direction": 1,
                                 "opened_at": now - 60, "closed_at": now, "status": "closed", "pnl": p,
                                 "tp_usd": 0.5, "sl_usd": 0.75})
    r = bot.risk_check(s, None, 101.1, 0.0)
    assert r["can_open"] and r["risk_factor"] == 0.5             # half the target banked: smaller trades
    db.insert("bot_trades", {"account": "paper", "mode": "scalp", "symbol": "R_75", "direction": 1,
                             "opened_at": now - 60, "closed_at": now, "status": "closed", "pnl": 1.0,
                             "tp_usd": 0.5, "sl_usd": 0.75})
    assert "target reached" in bot.risk_check(s, None, 102.1, 0.0)["reason"]
    for _ in range(3):
        bot.record_close({"id": 0, "mode": "scalp", "direction": 1, "symbol": "R_75"}, -0.5, 1, "stop loss")
    assert bot.state()["cooldown_until"] > time.time() + 29 * 60


def test_settings_validation(fresh):
    with pytest.raises(ValueError):
        bot.update_settings(daily_loss_pct=50)
    with pytest.raises(ValueError):
        bot.update_settings(account="xm")
    s = bot.update_settings(account="deriv", running=True)
    assert s["running"] is False            # live needs explicit confirmation first
    with pytest.raises(ValueError):
        bot.start()


# ----------------------------------------------------------------- full paper cycle ---
def test_paper_scalps_open_and_close_at_target(fresh, monkeypatch):
    bot.update_settings(account="paper", symbols=["R_75"], paper_start_balance=100, running=True, daily_target=5)
    price = {"R_75": 40000.0}
    monkeypatch.setattr(brokers.market, "last_price", lambda s, max_age=2.0: price["R_75"])
    monkeypatch.setattr(bot.sentiment, "event_risk", lambda inst: {"blocked": False})
    monkeypatch.setattr(bot, "_short_setup", lambda sym, risk: None)
    monkeypatch.setattr(bot, "_scalp_setup", lambda sym, risk: {
        "mode": "scalp", "symbol": sym, "direction": 1, "p": 0.80, "tp_dist": 24.0, "sl_dist": 36.0,
        "regime": "trending", "components": {}, "setup": "Trend continuation", "count": 3})
    b = brokers.PaperBroker(100.0)
    out = bot.step(b)
    pos = bot.open_positions("paper")
    assert out["opened"] == 2 and len(pos) == 2              # $100 account: at most 2 scalps at once
    assert all(p["tp_usd"] <= 0.50 + 1e-9 and p["sl_usd"] <= 1.0 + 1e-9 for p in pos)   # 1% risk cap on $100
    price["R_75"] = 40024.0                                  # price reaches the target distance
    bot.step(b, entries=False)
    closed = db.q("SELECT * FROM bot_trades WHERE status='closed'")
    assert len(closed) == 2 and all(c["close_reason"] == "take profit" for c in closed)
    net = sum(c["pnl"] for c in closed)
    assert b.balance() == pytest.approx(100 + net, abs=1e-3) and net == pytest.approx(sum(c["tp_usd"] for c in closed), abs=5e-3)


def test_short_trade_pair_breakeven(fresh, monkeypatch):
    bot.update_settings(account="paper", symbols=["EURUSD"], paper_start_balance=1000, running=True, daily_target=50)
    price = {"v": 1.1000}
    monkeypatch.setattr(brokers.market, "last_price", lambda s, max_age=2.0: price["v"])
    monkeypatch.setattr(bot.sentiment, "event_risk", lambda inst: {"blocked": False})
    monkeypatch.setattr(bot, "_scalp_setup", lambda sym, risk: None)
    monkeypatch.setattr(bot, "_short_setup", lambda sym, risk: {
        "mode": "short", "symbol": sym, "direction": -1, "p": 0.5, "tp_dist": 0.0020, "sl_dist": 0.0020,
        "regime": "trending", "components": {}, "setup": "Trend continuation", "count": 2})
    b = brokers.PaperBroker(1000.0)
    bot.step(b)
    pos = bot.open_positions("paper")
    assert len(pos) == 2
    # risk 1% of $1,000 split over two positions = $5 each at the stop
    assert all(p["sl_usd"] == pytest.approx(5.0) for p in pos)
    tps = sorted(p["tp_usd"] for p in pos)                                    # 1R and 2R, net of costs
    assert tps[1] == pytest.approx(2 * tps[0], rel=0.05) and 4 < tps[0] < 5
    price["v"] = 1.0979                                                       # just past 1R for the sell
    bot.step(b, entries=False)
    left = bot.open_positions("paper")
    assert len(left) == 1 and left[0]["breakeven"] == 1                       # second one protected


def test_relearn_pauses_losers_and_shrinks_targets(fresh):
    now = time.time()
    for i in range(25):
        db.insert("bot_trades", {"account": "paper", "mode": "scalp", "symbol": "R_100", "direction": 1,
                                 "opened_at": now - 600, "closed_at": now, "status": "closed",
                                 "pnl": -0.4 if i % 3 else 0.5, "tp_usd": 0.5, "sl_usd": 0.75,
                                 "max_profit": 0.35 if i % 3 else 0.5,
                                 "close_reason": "protect profit" if i % 3 else "take profit"})
    L = review.relearn()
    assert "scalp|R_100" in L["paused"]
    assert L["scalp_tp_factor"] == pytest.approx(0.9)            # many trades reversed after 60% of target
    st = review.stats()
    assert st["scalp"]["trades"] == 25 and st["scalp"]["gave_back_after_profit"] > 0


def test_raising_daily_limit_lifts_todays_halt(fresh):
    s = bot.update_settings(daily_loss_pct=3.0, daily_target=15)
    bot.risk_check(s, None, 100.0, 0.0)
    r = bot.risk_check(s, None, 95.63, 0.0)                       # -4.37 on the day: halted at 3%
    assert not r["can_open"] and "daily loss" in r["reason"]
    st = bot.state()
    st["halted"].pop("limit_pct")                                  # a halt saved by the older version
    bot.save_state(st)
    s = bot.update_settings(daily_loss_pct=10.0)
    r = bot.risk_check(s, None, 95.63, 0.0)
    assert r["can_open"] and bot.state()["halted"] is None
    s = bot.update_settings(daily_loss_pct=4.0)                    # lowering it below today's loss halts again
    assert not bot.risk_check(s, None, 95.63, 0.0)["can_open"]
