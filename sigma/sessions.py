"""Session-open scans.

At the open of each major trading session the robot runs a FAST analysis (seconds per pair)
on that session's pairs. Anything that qualifies as a signal is sent to you as an alert with a
'Deep analysis' button - the slow 1-2 minute deep check only runs when you ask for it.

Session opens are defined in each financial centre's own local time, so summer-time changes
are handled automatically; the app shows them in your time zone (SAST).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from . import config, db, engine

log = logging.getLogger("sigma.sessions")

SESSIONS = {
    "Sydney":   {"tz": "Australia/Sydney", "open": (7, 0),
                 "pairs": ["AUDUSD", "NZDUSD", "AUDJPY", "NZDJPY", "AUDNZD", "XAUUSD"]},
    "Tokyo":    {"tz": "Asia/Tokyo", "open": (9, 0),
                 "pairs": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "AUDUSD", "XAUUSD"]},
    "London":   {"tz": "Europe/London", "open": (8, 0),
                 "pairs": ["EURUSD", "GBPUSD", "EURGBP", "USDCHF", "EURJPY", "GBPJPY", "XAUUSD", "USDZAR"]},
    "New York": {"tz": "America/New_York", "open": (8, 0),
                 "pairs": ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "XAUUSD", "USDZAR", "BTCUSD", "ETHUSD"]},
}
SCAN_TIMEFRAMES = ("M15", "H1")
WINDOW_MIN = 20          # a session scan may start up to this many minutes after the open


def settings() -> dict:
    saved = db.kv_get("session_settings") or {}
    out = {}
    for name, s in SESSIONS.items():
        o = saved.get(name, {})
        out[name] = {**s, "pairs": o.get("pairs", s["pairs"]), "enabled": o.get("enabled", True)}
    return out


def update(session: str, pairs: list[str] | None = None, enabled: bool | None = None) -> dict:
    if session not in SESSIONS:
        raise ValueError(f"session must be one of {list(SESSIONS)}")
    saved = db.kv_get("session_settings") or {}
    cur = saved.get(session, {})
    if pairs is not None:
        cur["pairs"] = [p.strip().upper() for p in pairs if p.strip()][:20]
    if enabled is not None:
        cur["enabled"] = bool(enabled)
    saved[session] = cur
    db.kv_set("session_settings", saved)
    return settings()[session]


def open_time(session: str, day_local) -> datetime:
    s = SESSIONS[session]
    tz = ZoneInfo(s["tz"])
    return datetime(day_local.year, day_local.month, day_local.day, *s["open"], tzinfo=tz)


def schedule(now: float | None = None) -> list[dict]:
    """Next open of every session, in your time zone."""
    now = now or time.time()
    out = []
    for name, s in settings().items():
        tz = ZoneInfo(s["tz"])
        local = datetime.fromtimestamp(now, tz)
        nxt = None
        for add in range(0, 8):
            day = (local + timedelta(days=add)).date()
            t = open_time(name, day)
            if t.timestamp() > now - WINDOW_MIN * 60 and t.weekday() < 5:
                nxt = t
                break
        out.append({"session": name, "pairs": s["pairs"], "enabled": s["enabled"],
                    "next_open": nxt.timestamp() if nxt else None,
                    "next_open_local": nxt.astimezone(ZoneInfo(config.DISPLAY_TZ)).strftime("%a %d %b %H:%M")
                    if nxt else None,
                    "last_scan": (db.kv_get("session_last") or {}).get(name)})
    return sorted(out, key=lambda x: x["next_open"] or 9e18)


def due(now: float | None = None) -> list[str]:
    """Sessions whose open was within the last WINDOW_MIN minutes and that were not scanned yet."""
    now = now or time.time()
    last = db.kv_get("session_last") or {}
    out = []
    for name, s in settings().items():
        if not s["enabled"]:
            continue
        tz = ZoneInfo(s["tz"])
        t = open_time(name, datetime.fromtimestamp(now, tz).date())
        if t.weekday() >= 5:                         # no forex session on Saturday/Sunday
            continue
        if 0 <= now - t.timestamp() <= WINDOW_MIN * 60 and (last.get(name) or {}).get("open") != t.timestamp():
            out.append(name)
    return out


def scan(session: str, now: float | None = None, analyze=None) -> dict:
    """Fast-scan the session's pairs; alert any opportunity (deep analysis waits for you)."""
    from . import alerts
    analyze = analyze or (lambda sym, tf: engine.analyze(sym, tf, source="live", journal=False))
    s = settings()[session]
    found, checked, errors = [], 0, []
    for sym in s["pairs"]:
        best = None
        for tf in SCAN_TIMEFRAMES:
            try:
                r = analyze(sym, tf)
            except Exception as e:
                errors.append(f"{sym} {tf}: {e}")
                continue
            checked += 1
            if r.get("market_closed") or not r.get("is_signal"):
                continue
            if best is None or r["expectancy"] > best["expectancy"]:
                best = r
        if best:
            found.append(best)
    found.sort(key=lambda r: -r["expectancy"])
    sent = [alerts.session_opportunity(session, r) for r in found]
    now = now or time.time()
    tz = ZoneInfo(SESSIONS[session]["tz"])
    last = db.kv_get("session_last") or {}
    last[session] = {"at": now, "open": open_time(session, datetime.fromtimestamp(now, tz).date()).timestamp(),
                     "checked": checked, "opportunities": len(found)}
    db.kv_set("session_last", last)
    alerts.session_summary(session, s["pairs"], found)
    return {"session": session, "pairs": s["pairs"], "analyses": checked, "opportunities": len(found),
            "alerts": sent, "errors": errors[:5]}
