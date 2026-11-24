"""Free extras from Deriv's public API (no login): the economic calendar and official trading hours.

Economic calendar  - Deriv's own calendar, merged with the ForexFactory feed so that one source going
                     down never leaves Sigma blind to news.
Trading times      - the exact open/close times of every Deriv market today, so Sigma knows a market is
                     shut (weekends, holidays, daily breaks) without waiting for candles to go stale, can
                     skip closed markets in scans (saving API calls) and keeps the bot from opening trades
                     just before a market closes.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from .http import cached
from ..symbols import resolve

log = logging.getLogger("sigma.deriv_info")

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BACKOFF = 300                    # after a failure, don't ask Deriv again for 5 minutes
_fail_until: dict[str, float] = {}


def _guarded(key: str, ttl: float, load, empty):
    if time.time() < _fail_until.get(key, 0):
        return empty
    try:
        return cached(key, ttl, load)
    except Exception as e:
        _fail_until[key] = time.time() + BACKOFF
        log.debug("%s: %s", key, e)
        return empty


# ------------------------------------------------------------------------- economic calendar ---
def _num(v):
    """Calendar values come as plain values or as {"display_value": ..., "value": ...}."""
    if isinstance(v, dict):
        v = v.get("display_value", v.get("value"))
    return "" if v is None else str(v)


def _impact(v) -> str:
    """Deriv rates impact 1-5 (5 = highest); some versions send words."""
    if isinstance(v, str) and not v.strip().lstrip("-").isdigit():
        w = v.strip().lower()
        return "High" if w.startswith("high") else "Medium" if w.startswith("med") else "Low"
    try:
        n = int(v)
    except (TypeError, ValueError):
        return "Low"
    return "High" if n >= 4 else "Medium" if n == 3 else "Low"


def parse_calendar(msg: dict) -> list[dict]:
    cal = msg.get("economic_calendar") or {}
    events = cal.get("events", cal) if isinstance(cal, dict) else cal
    out = []
    for e in events or []:
        try:
            ts = e.get("release_date")
            ts = float(ts) if not isinstance(ts, str) or ts.isdigit() else datetime.fromisoformat(ts).timestamp()
        except Exception:
            continue
        out.append({"time": ts, "currency": str(e.get("currency", "")).upper(), "impact": _impact(e.get("impact")),
                    "title": e.get("event_name") or e.get("title") or "", "forecast": _num(e.get("forecast")),
                    "previous": _num(e.get("previous")), "actual": _num(e.get("actual")), "source": "deriv"})
    return out


def economic_calendar(days_back: int = 1, days_ahead: int = 7) -> list[dict]:
    from .market import _deriv_request

    def load():
        now = int(time.time())
        msg = _deriv_request({"economic_calendar": 1, "start_date": now - days_back * 86400,
                              "end_date": now + days_ahead * 86400})
        return parse_calendar(msg)

    return _guarded("deriv_calendar", 1800, load, [])


def merge_calendars(*lists: list[dict]) -> list[dict]:
    """One list without duplicates: the same currency within 10 minutes and a similar title counts as one
    event; the copy with the higher impact (and an actual value, if any) is kept."""
    rank = {"High": 3, "Medium": 2, "Low": 1}

    def words(t):
        return {w for w in "".join(c.lower() if c.isalnum() else " " for c in t).split() if len(w) > 2}

    out: list[dict] = []
    for e in sorted((e for lst in lists for e in lst), key=lambda e: e["time"]):
        dup = None
        for o in out[-40:]:
            if o["currency"] == e["currency"] and abs(o["time"] - e["time"]) <= 600:
                a, b = words(o["title"]), words(e["title"])
                if not a or not b or len(a & b) / min(len(a), len(b)) >= 0.5:
                    dup = o
                    break
        if dup is None:
            out.append(dict(e))
            continue
        if rank.get(e["impact"], 0) > rank.get(dup["impact"], 0):
            dup["impact"] = e["impact"]
        for k in ("actual", "forecast", "previous"):
            if not dup.get(k) and e.get(k):
                dup[k] = e[k]
        dup["source"] = "both" if dup.get("source") != e.get("source") else dup.get("source")
    return out


# ----------------------------------------------------------------------------- trading times ---
def _parse_hms(day: datetime, s: str) -> datetime | None:
    try:
        h, m, sec = (int(x) for x in s.split(":"))
    except Exception:
        return None
    if h >= 24:
        return day + timedelta(days=1)
    return day.replace(hour=h, minute=m, second=sec)


def parse_trading_times(msg: dict, day: datetime) -> dict[str, dict]:
    """{deriv_symbol: {"sessions": [(open_ts, close_ts), ...], "trading_days": [...], "events": [...]}}"""
    out = {}
    tt = msg.get("trading_times") or {}
    day0 = day.replace(hour=0, minute=0, second=0, microsecond=0)
    for mk in tt.get("markets", []):
        for sm in mk.get("submarkets", []):
            for s in sm.get("symbols", []):
                code = s.get("underlying_symbol") or s.get("symbol")
                if not code:
                    continue
                t = s.get("times") or {}
                opens = t.get("open") or []
                closes = t.get("close") or []
                sessions = []
                for o, c in zip(opens, closes):
                    if "--" in str(o) or "--" in str(c):
                        continue          # closed all day
                    a, b = _parse_hms(day0, o), _parse_hms(day0, c)
                    if a and b:
                        if b <= a:
                            b += timedelta(days=1)
                        sessions.append((a.timestamp(), b.timestamp()))
                out[code] = {"name": s.get("name"), "market": mk.get("name"), "submarket": sm.get("name"),
                             "sessions": sessions, "trading_days": s.get("trading_days") or [],
                             "events": s.get("events") or []}
    return out


def trading_times(day: datetime | None = None) -> dict[str, dict]:
    from .market import _deriv_request
    day = day or datetime.now(timezone.utc)
    date = day.strftime("%Y-%m-%d")

    def load():
        return parse_trading_times(_deriv_request({"trading_times": date}), day)

    return _guarded(f"deriv_times:{date}", 6 * 3600, load, {})


def market_hours(name: str, now: float | None = None) -> dict:
    """Is the market open right now, per Deriv's official hours?
    Returns {"open": True/False/None, "closes_in_min", "opens_in_min", "note"}; None = unknown."""
    now = now or time.time()
    inst = resolve(name)
    if inst.asset_class == "synthetic":
        return {"open": True, "closes_in_min": None, "opens_in_min": None, "note": "synthetic: 24/7"}
    code = inst.deriv
    if not code:
        return {"open": None, "note": "not a Deriv market"}
    today = datetime.fromtimestamp(now, timezone.utc)
    info = {}
    for d in (today - timedelta(days=1), today):       # a session that started yesterday may still be running
        x = trading_times(d).get(code)
        if x:
            info.setdefault("sessions", []).extend(x["sessions"])
            info["events"] = x["events"]
    if not info:
        return {"open": None, "note": "hours unavailable"}
    sessions = sorted(set(info["sessions"]))
    for a, b in sessions:
        if a <= now < b:
            return {"open": True, "closes_in_min": round((b - now) / 60, 1), "opens_in_min": None,
                    "events": info.get("events", [])}
    later = [a for a, _ in sessions if a > now]
    return {"open": False, "closes_in_min": None,
            "opens_in_min": round((later[0] - now) / 60, 1) if later else None,
            "note": "closed now" + ("" if later else "; opens on a later day"), "events": info.get("events", [])}
