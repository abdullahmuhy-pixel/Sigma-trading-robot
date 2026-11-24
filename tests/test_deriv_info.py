import os
import tempfile
import time
from datetime import datetime, timezone

import pytest

from sigma import db, engine
from sigma.data import deriv_info

DAY = datetime(2026, 11, 24, tzinfo=timezone.utc)          # a Tuesday
T0 = DAY.timestamp()

TT = {"trading_times": {"markets": [
    {"name": "Forex", "submarkets": [{"name": "Major Pairs", "symbols": [
        {"name": "EUR/USD", "underlying_symbol": "frxEURUSD", "trading_days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
         "times": {"open": ["00:00:00"], "close": ["23:59:59"], "settlement": "23:59:59"}, "events": []}]}]},
    {"name": "Stock Indices", "submarkets": [{"name": "American indices", "symbols": [
        {"name": "US Tech 100", "underlying_symbol": "OTC_NDX", "trading_days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
         "times": {"open": ["01:00:00", "13:30:00"], "close": ["10:00:00", "20:00:00"]},
         "events": [{"dates": "Fridays", "descrip": "Closes early (at 19:00)"}]}]}]},
    {"name": "Commodities", "submarkets": [{"name": "Metals", "symbols": [
        {"name": "Gold/USD", "underlying_symbol": "frxXAUUSD", "trading_days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
         "times": {"open": ["--"], "close": ["--"]}, "events": [{"dates": "today", "descrip": "Closed all day"}]}]}]},
]}}


@pytest.fixture()
def hours(monkeypatch):
    db.connect(os.path.join(tempfile.mkdtemp(), "h.db"))
    parsed = deriv_info.parse_trading_times(TT, DAY)
    monkeypatch.setattr(deriv_info, "trading_times", lambda day=None: parsed)
    real = deriv_info.resolve

    def res(name):                       # offline there is no Deriv catalogue to map index names
        if name in ("US100", "OTC_NDX"):
            from types import SimpleNamespace
            return SimpleNamespace(name="US100", deriv="OTC_NDX", asset_class="index")
        return real(name)
    monkeypatch.setattr(deriv_info, "resolve", res)
    return parsed


def test_parse_trading_times(hours):
    assert hours["frxEURUSD"]["sessions"] == [(T0, T0 + 86399)]
    nd = hours["OTC_NDX"]["sessions"]
    assert nd == [(T0 + 3600, T0 + 36000), (T0 + 13.5 * 3600, T0 + 20 * 3600)]   # two sessions with a break
    assert hours["frxXAUUSD"]["sessions"] == []                                   # closed all day
    assert hours["OTC_NDX"]["events"][0]["descrip"].startswith("Closes early")


def test_market_hours(hours):
    h = deriv_info.market_hours("US100", now=T0 + 12 * 3600)          # in the midday break
    assert h["open"] is False and h["opens_in_min"] == pytest.approx(90)
    h = deriv_info.market_hours("US100", now=T0 + 19.5 * 3600)
    assert h["open"] is True and h["closes_in_min"] == pytest.approx(30)
    assert deriv_info.market_hours("XAUUSD", now=T0 + 3600)["open"] is False
    assert deriv_info.market_hours("EURUSD", now=T0 + 3600)["open"] is True
    assert deriv_info.market_hours("R_75")["open"] is True                # synthetic: always open


def test_unknown_hours_never_block(monkeypatch):
    monkeypatch.setattr(deriv_info, "trading_times", lambda day=None: {})
    assert deriv_info.market_hours("EURUSD")["open"] is None
    assert engine.is_closed_now("EURUSD") is False


def test_parse_calendar_both_formats():
    msg = {"economic_calendar": {"events": [
        {"currency": "usd", "event_name": "Non-Farm Payrolls", "impact": 5, "release_date": 1795000000,
         "actual": {"display_value": "180K"}, "forecast": {"display_value": "175K"}, "previous": {}},
        {"currency": "EUR", "event_name": "ECB minutes", "impact": "2", "release_date": "1795003600"},
        {"currency": "GBP", "event_name": "BoE rate", "impact": "High", "release_date": "2026-11-26T12:00:00+00:00"},
        {"currency": "JPY", "event_name": "broken", "release_date": None}]}}
    ev = deriv_info.parse_calendar(msg)
    assert [e["currency"] for e in ev] == ["USD", "EUR", "GBP"]
    assert [e["impact"] for e in ev] == ["High", "Low", "High"]
    assert ev[0]["actual"] == "180K" and ev[0]["forecast"] == "175K" and ev[0]["previous"] == ""


def test_merge_calendars_dedupes():
    ff = [{"time": 1000, "currency": "USD", "impact": "Medium", "title": "Non-Farm Employment Change",
           "forecast": "175K", "previous": "", "actual": "", "source": "forexfactory"},
          {"time": 5000, "currency": "EUR", "impact": "High", "title": "CPI Flash Estimate y/y",
           "forecast": "", "previous": "", "actual": "", "source": "forexfactory"}]
    dv = [{"time": 1200, "currency": "USD", "impact": "High", "title": "Non-Farm Payrolls Employment Change",
           "forecast": "", "previous": "150K", "actual": "", "source": "deriv"},
          {"time": 1100, "currency": "USD", "impact": "Low", "title": "Crude Oil Inventories",
           "forecast": "", "previous": "", "actual": "", "source": "deriv"}]
    m = deriv_info.merge_calendars(ff, dv)
    assert len(m) == 3
    nfp = next(e for e in m if "Non-Farm" in e["title"])
    assert nfp["impact"] == "High" and nfp["forecast"] == "175K" and nfp["previous"] == "150K"
    assert nfp["source"] == "both"


def test_backoff_after_failure(monkeypatch):
    calls = []

    def boom(payload, timeout=15):
        calls.append(1)
        raise RuntimeError("down")
    from sigma.data import market
    monkeypatch.setattr(market, "_deriv_request", boom)
    deriv_info._fail_until.clear()
    assert deriv_info.economic_calendar() == [] and deriv_info.economic_calendar() == []
    assert len(calls) == 1                                              # second call skipped by the back-off


def test_bot_respects_market_close(hours, monkeypatch):
    from sigma.autobot import bot
    monkeypatch.setattr(time, "time", lambda: T0 + 19.9 * 3600)             # US100 closes in 6 min
    assert bot._closing_soon("US100", bot.NO_ENTRY_BEFORE_CLOSE_MIN)
    assert bot._closing_soon("US100", bot.FLAT_BEFORE_CLOSE_MIN) is None
    monkeypatch.setattr(time, "time", lambda: T0 + 19.95 * 3600)            # 3 min left: close positions
    assert "closes in 3 min" in bot._closing_soon("US100", bot.FLAT_BEFORE_CLOSE_MIN)
    assert bot._closing_soon("EURUSD", 15) is None                           # hours from the test day
