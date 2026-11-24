"""Crypto futures positioning from venues that answer from Render's US servers (Binance blocks them).

  Hyperliquid  - one free request returns funding and open interest for every perpetual (no key).
  Kraken       - Kraken Futures tickers: funding rate and open interest (no key).
  OKX          - optional: share of accounts net long vs net short (may be region-blocked; skipped if so).

What Sigma reads from it:
  Funding      - when longs pay a lot to stay long (high positive funding) the long side is crowded and
                 over-leveraged -> contrarian bearish; strongly negative funding -> contrarian bullish.
  Long/short   - share of accounts long (OKX), read contrarian like the forex crowd data.
  Open interest- remembered over time: rising open interest with price moving = new money behind the
                 move (a note for the explanation); falling = positions closing, a weaker move.
The resulting score is folded into the crowd-positioning factor for crypto, so the learning model
weighs it from Sigma's own results.
"""
from __future__ import annotations

import logging
import time

from .. import db
from ..symbols import resolve
from .http import _session, cached

log = logging.getLogger("sigma.crypto_derivs")

HL_URL = "https://api.hyperliquid.xyz/info"
KRAKEN_URL = "https://futures.kraken.com/derivatives/api/v3/tickers"
OKX_LS_URL = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio"
KRAKEN_CODE = {"BTC": "XBT"}
NORMAL_APR = 0.10          # typical funding, ~10%/year paid by longs: treated as neutral
FULL_APR = 0.60            # 60%/year above (or below) normal = the full contrarian score
BACKOFF = 600
_fail_until: dict[str, float] = {}


def _guard(key: str, fn, empty):
    if time.time() < _fail_until.get(key, 0):
        return empty
    try:
        return fn()
    except Exception as e:
        _fail_until[key] = time.time() + BACKOFF
        log.info("%s unavailable: %s", key, e)
        return empty


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------ sources ---
def parse_hyperliquid(js) -> dict[str, dict]:
    meta, ctxs = js[0], js[1]
    out = {}
    for a, c in zip(meta.get("universe", []), ctxs):
        name, fr, oi, px = a.get("name"), _f(c.get("funding")), _f(c.get("openInterest")), _f(c.get("markPx"))
        if name and fr is not None:
            out[name.upper()] = {"funding_hourly": fr, "open_interest": oi, "open_interest_usd": oi * px if oi and px else None,
                                 "mark": px, "prev_day": _f(c.get("prevDayPx"))}
    return out


def hyperliquid() -> dict[str, dict]:
    def load():
        r = _session.post(HL_URL, json={"type": "metaAndAssetCtxs"}, timeout=15)
        r.raise_for_status()
        return parse_hyperliquid(r.json())
    return _guard("hyperliquid", lambda: cached("hl_ctxs", 300, load), {})


def parse_kraken(js) -> dict[str, dict]:
    out = {}
    for t in js.get("tickers", []):
        sym = str(t.get("symbol", ""))
        if not (sym.startswith("PF_") and sym.endswith("USD")) or t.get("suspended"):
            continue
        coin = sym[3:-3]
        mark = _f(t.get("markPrice")) or _f(t.get("last"))
        rel = _f(t.get("relativeFundingRate"))
        fr = _f(t.get("fundingRate"))
        if rel is None and (fr is None or not mark):
            continue
        out["BTC" if coin == "XBT" else coin] = {"funding_hourly": rel if rel is not None else fr / mark,
                                                 "open_interest": _f(t.get("openInterest")), "mark": mark}
    return out


def kraken() -> dict[str, dict]:
    def load():
        r = _session.get(KRAKEN_URL, timeout=15)
        r.raise_for_status()
        return parse_kraken(r.json())
    return _guard("kraken", lambda: cached("kraken_tickers", 300, load), {})


def okx_long_share(coin: str) -> float | None:
    def load():
        r = _session.get(OKX_LS_URL, params={"ccy": coin, "period": "1H"}, timeout=15)
        r.raise_for_status()
        js = r.json()
        if str(js.get("code")) != "0" or not js.get("data"):
            raise RuntimeError(js.get("msg") or "no data")
        ratio = float(js["data"][0][1])                 # newest first: [timestamp, long/short ratio]
        return 100 * ratio / (1 + ratio)
    return _guard("okx", lambda: cached(f"okx_ls:{coin}", 900, load), None)


# ------------------------------------------------------------------------------ scoring ---
def funding_score(hourly: float) -> float:
    apr = hourly * 24 * 365
    return max(-1.0, min(1.0, -(apr - NORMAL_APR) / FULL_APR))


def _oi_history(coin: str, oi_usd: float | None, now: float) -> dict:
    """Keeps an open-interest snapshot every 30 min (2 days) and returns the % change over ~4h and ~24h."""
    if not oi_usd:
        return {}
    hist = db.kv_get("oi_history") or {}
    h = hist.get(coin, [])
    if not h or now - h[-1][0] >= 1800:
        h.append([now, oi_usd])
        hist[coin] = h[-96:]
        db.kv_set("oi_history", hist)

    def change(hours):
        old = [v for t, v in h if now - t >= hours * 3600 * 0.9]
        return round(100 * (oi_usd / old[-1] - 1), 1) if old else None
    return {"oi_change_4h_pct": change(4), "oi_change_24h_pct": change(24)}


def analyze(name: str, now: float | None = None) -> dict | None:
    inst = resolve(name)
    if inst.asset_class != "crypto" or not inst.base:
        return None
    now = now or time.time()
    coin = inst.base.upper()
    hl = hyperliquid().get(coin)
    kr = kraken().get(coin) if not hl else None              # Kraken only when Hyperliquid has nothing
    venue = hl or kr
    long_share = okx_long_share(coin)
    if not venue and long_share is None:
        return None
    parts, notes, out = [], [], {"coin": coin}
    if venue:
        fr = venue["funding_hourly"]
        fs = funding_score(fr)
        parts.append(fs)
        apr = fr * 24 * 365
        out.update(funding_hourly=fr, funding_apr_pct=round(100 * apr, 1), funding_source="Hyperliquid" if hl else "Kraken")
        notes.append(f"Funding {100 * apr:+.0f}%/yr ({out['funding_source']}): " +
                     ("longs paying heavily - crowded long, squeeze risk down" if fs <= -0.4 else
                      "shorts paying - crowded short, squeeze risk up" if fs >= 0.4 else "normal"))
        oi = _oi_history(coin, venue.get("open_interest_usd") or (
            venue["open_interest"] * venue["mark"] if venue.get("open_interest") and venue.get("mark") else None), now)
        out.update(oi)
        if venue.get("open_interest_usd"):
            out["open_interest_usd"] = round(venue["open_interest_usd"])
        ch = oi.get("oi_change_24h_pct") if oi.get("oi_change_24h_pct") is not None else oi.get("oi_change_4h_pct")
        if ch is not None and venue.get("prev_day") and venue.get("mark"):
            pch = 100 * (venue["mark"] / venue["prev_day"] - 1)
            if abs(ch) >= 3:
                notes.append(f"Open interest {ch:+.1f}% with price {pch:+.1f}% (24h): " +
                             ("new money behind the move" if ch > 0 and abs(pch) >= 0.5 else
                              "leverage building while price stalls - breakout or squeeze likely" if ch > 0 else
                              "positions closing - the move is losing fuel"))
    if long_share is not None:
        from .positioning import contrarian_score
        parts.append(contrarian_score(long_share))
        out["okx_long_pct"] = round(long_share, 1)
        notes.append(f"OKX accounts {long_share:.0f}% net long")
    out["score"] = round(sum(parts) / len(parts), 3)
    out["notes"] = notes
    return out


def check_sources() -> dict:
    """Live reachability check of every crypto source (for the data-sources report)."""
    res = {}
    for key, fn in (("hyperliquid", hyperliquid), ("kraken", kraken)):
        _fail_until.pop(key, None)
        t = time.time()
        d = fn()
        res[key] = {"ok": bool(d), "markets": len(d), "ms": round(1000 * (time.time() - t))}
    _fail_until.pop("okx", None)
    v = okx_long_share("BTC")
    res["okx"] = {"ok": v is not None}
    return res
