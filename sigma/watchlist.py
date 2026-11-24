"""Your background watchlist: the ONLY markets the robot scans on its own, on the timeframes you
choose. Signals found on these are the ones you get notified about. Everything else is analysed
only when you ask. Stored in the database, so it is the same on every device and survives updates."""
from __future__ import annotations

from . import db
from .symbols import TIMEFRAMES, resolve

FOCUS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY",
         "EURGBP", "USDZAR", "XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD"]      # forex, gold/silver, crypto
DEFAULT = {"symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD", "USDZAR", "BTCUSD", "ETHUSD"],
           "timeframes": ["M15", "H1"], "enabled": True}
MAX_SYMBOLS = 40
# Deriv's forex list, used only if the live catalogue can't be loaded
DERIV_FX_FALLBACK = ["AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCAD", "EURCHF", "EURGBP",
                     "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNOK", "GBPNZD",
                     "GBPUSD", "NZDJPY", "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "USDMXN", "USDNOK", "USDPLN",
                     "USDSEK"]


def currency_pairs(currencies=("JPY", "USD", "EUR", "GBP"), extra=()) -> list[str]:
    """Every forex pair Deriv offers that contains one of `currencies` (from the live catalogue)."""
    pairs = []
    try:
        from .data.deriv_catalog import catalog
        items = catalog()
        pairs = [i["symbol"][3:] for i in items
                 if i.get("market") == "forex" and str(i.get("symbol", "")).startswith("frx") and len(i["symbol"]) == 9]
        pairs += [i["symbol"] for i in items if i.get("computed")]   # e.g. CHFJPY, CADJPY
    except Exception:
        pairs = []
    pairs = pairs or list(DERIV_FX_FALLBACK) + ["CADJPY", "CHFJPY"]
    pairs += [e for e in extra if e not in pairs]
    want = [p for p in pairs if p[:3] in currencies or p[3:] in currencies]
    major = ["EURUSD", "GBPUSD", "USDJPY"]          # the most liquid first, then alphabetical
    return [p for p in major if p in want] + sorted(p for p in want if p not in major)


SETUP_TIMEFRAMES = ["M15", "H1", "H4"]   # entries, main trading timeframe, bigger swings / direction


def apply_default_setup_once() -> bool:
    """One-time setup at the user's request (24 Nov 2026): every JPY, USD, EUR and GBP pair plus gold
    and silver (v1), scanned on M15, H1 and H4 (v2 - replaced 'every timeframe'). Each step runs once;
    later edits in the app are kept."""
    import time as _t
    changed = False
    if not db.kv_get("setup_fx_metals_v1"):
        preset("fx_metals")
        set_enabled(True)
        db.kv_set("setup_fx_metals_v1", {"at": _t.time()})
        changed = True
    if not db.kv_get("setup_timeframes_v2"):
        set_timeframes(SETUP_TIMEFRAMES)
        db.kv_set("setup_timeframes_v2", {"at": _t.time()})
        changed = True
    return changed


def preset(name: str) -> dict:
    if name == "fx_metals":
        return set_symbols(currency_pairs() + ["XAUUSD", "XAGUSD"])
    if name == "focus":
        return set_symbols(FOCUS)
    raise ValueError(f"unknown preset {name}")
SCANS_PER_DAY = {"M1": 1440, "M5": 288, "M15": 96, "M30": 48, "H1": 24, "H4": 6, "D1": 1}


def get() -> dict:
    w = db.kv_get("watchlist") or {}
    out = {**DEFAULT, **w}
    out["scans_per_day"] = len(out["symbols"]) * sum(SCANS_PER_DAY[t] for t in out["timeframes"])
    out["classes"] = {}
    for s in out["symbols"]:
        try:
            out["classes"][s] = resolve(s).asset_class
        except Exception:
            out["classes"][s] = "?"
    out["focus"] = FOCUS
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
