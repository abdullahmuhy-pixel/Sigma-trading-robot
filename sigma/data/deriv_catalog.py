"""Every instrument Deriv offers (forex, synthetics, crypto, commodities, stock
indices, baskets), read live from Deriv's active_symbols list and cached.

Works with both Deriv APIs: the new one names fields underlying_symbol /
underlying_symbol_name / pip_size, the legacy one symbol / display_name / pip."""
from __future__ import annotations

import logging
import re
import threading
import time

log = logging.getLogger("sigma.deriv_catalog")

MARKET_LABEL = {
    "forex": "Forex", "synthetic_index": "Synthetic indices", "derived": "Synthetic indices",
    "cryptocurrency": "Crypto", "commodities": "Commodities", "indices": "Stock indices",
    "basket_index": "Baskets",
}

_lock = threading.Lock()
_cache: dict = {"t": 0.0, "items": []}
TTL = 6 * 3600


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def aliases(name: str) -> set[str]:
    """Short names traders use: 'V75', 'V75 1s', 'Boom 1000', 'Jump 50', 'Step'."""
    out = {norm(name), norm(re.sub(r"\bindex\b", "", name, flags=re.I))}
    m = re.match(r"Volatility (\d+) \(1s\) Index", name, re.I)
    if m:
        out |= {f"v{m.group(1)}1s", f"vix{m.group(1)}1s", f"volatility{m.group(1)}1s"}
    m = re.match(r"Volatility (\d+) Index", name, re.I)
    if m:
        out |= {f"v{m.group(1)}", f"vix{m.group(1)}", f"volatility{m.group(1)}"}
    m = re.match(r"(Boom|Crash|Jump) (\d+)", name, re.I)
    if m:
        out.add(norm(m.group(1) + m.group(2)))
    return {a for a in out if a}


def _item(raw: dict) -> dict:
    sym = raw.get("underlying_symbol") or raw.get("symbol")
    name = raw.get("underlying_symbol_name") or raw.get("display_name") or sym
    market = raw.get("market", "")
    return {
        "symbol": sym, "name": name, "market": market,
        "market_label": MARKET_LABEL.get(market, market.replace("_", " ").title()),
        "submarket": raw.get("submarket", ""),
        "submarket_label": raw.get("submarket_display_name") or raw.get("submarket", "").replace("_", " ").title(),
        "open": bool(raw.get("exchange_is_open", 1)) and not raw.get("is_trading_suspended", 0),
        "pip": raw.get("pip_size") or raw.get("pip"),
    }


def catalog(force: bool = False) -> list[dict]:
    from .. import db
    from .market import deriv_active_symbols
    with _lock:
        if not force and _cache["items"] and time.time() - _cache["t"] < TTL:
            return _cache["items"]
        try:
            items = [_item(r) for r in deriv_active_symbols()]
            items = [i for i in items if i["symbol"]]
            items.sort(key=lambda i: (i["market_label"], i["submarket"], i["name"]))
            _cache.update(t=time.time(), items=items)
            try:
                db.kv_set("deriv_catalog", items)
            except Exception:
                pass
        except Exception as e:
            log.warning("Deriv symbol list unavailable: %s", e)
            _cache["t"] = time.time() - TTL + 300   # retry in 5 minutes, not on every call
            if not _cache["items"]:
                try:
                    _cache["items"] = db.kv_get("deriv_catalog", []) or []
                except Exception:
                    pass
        return _cache["items"]


def find(query: str) -> dict | None:
    """Match a Deriv code (R_75, frxEURUSD), a display name or a common alias."""
    items = catalog()
    q = query.strip()
    for i in items:
        if i["symbol"].lower() == q.lower():
            return i
    nq = norm(q)
    for i in items:
        if nq in aliases(i["name"]):
            return i
    return None


def search(query: str = "", market: str = "", limit: int = 50) -> list[dict]:
    nq = norm(query)
    out = []
    for i in catalog():
        if market and norm(market) not in (norm(i["market"]), norm(i["market_label"])):
            continue
        if nq and nq not in norm(i["name"]) and nq not in norm(i["symbol"]) and not any(nq in a for a in aliases(i["name"])):
            continue
        out.append(i)
    return out[:limit]
