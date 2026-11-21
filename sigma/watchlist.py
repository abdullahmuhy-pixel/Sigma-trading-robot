"""Your background watchlist: the ONLY markets the robot scans on its own, on the timeframes you
choose. Signals found on these are the ones you get notified about. Everything else is analysed
only when you ask. Stored in the database, so it is the same on every device and survives updates."""
from __future__ import annotations

from . import db
from .symbols import TIMEFRAMES, resolve

DEFAULT = {"symbols": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "USDZAR", "BTCUSD", "R_75", "BOOM1000"],
           "timeframes": ["M15", "H1"], "enabled": True}
MAX_SYMBOLS = 30
SCANS_PER_DAY = {"M1": 1440, "M5": 288, "M15": 96, "M30": 48, "H1": 24, "H4": 6, "D1": 1}


def get() -> dict:
    w = db.kv_get("watchlist") or {}
    out = {**DEFAULT, **w}
    out["scans_per_day"] = len(out["symbols"]) * sum(SCANS_PER_DAY[t] for t in out["timeframes"])
    return out


def _norm(sym: str) -> str:
    return resolve(sym).name


def _save(w: dict) -> dict:
    db.kv_set("watchlist", {k: w[k] for k in ("symbols", "timeframes", "enabled")})
    return get()


def set_symbols(symbols: list[str]) -> dict:
    w = get()
    seen, out = set(), []
    for s in symbols:
        if not s or not s.strip():
            continue
        n = _norm(s)
        if n not in seen:
            seen.add(n)
            out.append(n)
    if len(out) > MAX_SYMBOLS:
        raise ValueError(f"at most {MAX_SYMBOLS} markets in the background watchlist")
    w["symbols"] = out
    return _save(w)


def add(symbols: list[str]) -> dict:
    return set_symbols(get()["symbols"] + list(symbols))


def remove(symbols: list[str]) -> dict:
    drop = {_norm(s) for s in symbols}
    return set_symbols([s for s in get()["symbols"] if s not in drop])


def set_timeframes(tfs: list[str]) -> dict:
    tfs = [t.upper() for t in tfs if t and t.upper() in TIMEFRAMES]
    if not tfs:
        raise ValueError(f"choose at least one timeframe from {list(TIMEFRAMES)}")
    w = get()
    w["timeframes"] = [t for t in TIMEFRAMES if t in tfs]   # keep a tidy order
    return _save(w)


def set_enabled(on: bool) -> dict:
    w = get()
    w["enabled"] = bool(on)
    return _save(w)
