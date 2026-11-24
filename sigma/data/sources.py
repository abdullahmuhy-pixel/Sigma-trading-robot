"""One report of which data sources work from the server right now."""
from __future__ import annotations

import time

from .. import config


def _try(fn):
    t = time.time()
    try:
        v = fn()
        ok = bool(v) if not isinstance(v, (int, float)) else True
        return {"ok": ok, "ms": round(1000 * (time.time() - t)), **({} if ok else {"error": "no data returned"})}
    except Exception as e:
        return {"ok": False, "ms": round(1000 * (time.time() - t)), "error": str(e)[:160]}


def check_all() -> dict:
    from . import crypto_derivs, deriv_info, market, news, positioning
    out = {
        "deriv_prices": _try(lambda: len(market.deriv_candles("R_75", "M1", 5))),
        "forexfactory_calendar": _try(lambda: news.forexfactory_calendar()),
        "deriv_trading_hours": _try(lambda: deriv_info.trading_times()),
        "binance_spot": _try(lambda: len(market.binance_candles("BTCUSDT", "M1", 5))),
    }
    if not out["binance_spot"]["ok"]:
        out["binance_spot"]["error"] = " | ".join(
            f"{p['host'].split('//')[1]}: " + ("ok" if p["ok"] else p["error"]) for p in market.binance_probe())
    out["twelvedata"] = (_try(lambda: len(market.twelvedata_candles("EUR/USD", "H1", 5)))
                         if config.TWELVEDATA_API_KEY else {"ok": None, "note": "no TWELVEDATA_API_KEY set"})
    conf = positioning.configured()
    positioning._fail_until.clear()
    out["myfxbook"] = (_try(lambda: positioning.myfxbook_outlook()) if conf["myfxbook"]
                       else {"ok": None, "note": "add MYFXBOOK_EMAIL and MYFXBOOK_PASSWORD to switch on"})
    out["ig_sentiment"] = (_try(lambda: positioning.ig_sentiment("EURUSD")) if conf["ig"]
                           else {"ok": None, "note": "add IG_API_KEY, IG_USERNAME and IG_PASSWORD to switch on"})
    for k, v in crypto_derivs.check_sources().items():
        out[k] = v
    out["checked_at"] = time.time()
    return out
