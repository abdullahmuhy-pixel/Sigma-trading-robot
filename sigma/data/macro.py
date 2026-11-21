"""Fundamental data from free public sources:
- FRED (St. Louis Fed): policy/short rates, bond yields, real yields, dollar index, VIX
- CFTC Commitments of Traders: speculative positioning in currency & gold futures
- alternative.me Crypto Fear & Greed index
"""
from __future__ import annotations

import logging

import pandas as pd

from .. import config
from .http import cached, get_json

log = logging.getLogger("sigma.macro")

# OECD series on FRED follow a common pattern per country code.
COUNTRY = {"USD": "US", "EUR": "EZ", "GBP": "GB", "JPY": "JP", "AUD": "AU", "NZD": "NZ",
           "CAD": "CA", "CHF": "CH", "ZAR": "ZA", "SEK": "SE", "NOK": "NO", "MXN": "MX"}
LONG_RATE_CC = {**COUNTRY, "EUR": "DE"}  # use the German Bund for the euro long rate

# CFTC contract market codes (Legacy futures-only report)
COT_CODES = {"EUR": "099741", "JPY": "097741", "GBP": "096742", "CHF": "092741",
             "CAD": "090741", "AUD": "232741", "NZD": "112741", "MXN": "095741",
             "ZAR": "122741", "XAU": "088691", "XAG": "084691", "BTC": "133741"}


def fred_series(series_id: str, start: str = "2018-01-01") -> pd.Series:
    if not config.FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY not set")

    def load():
        js = get_json("https://api.stlouisfed.org/fred/series/observations", {
            "series_id": series_id, "api_key": config.FRED_API_KEY,
            "file_type": "json", "observation_start": start,
        })
        obs = [(o["date"], o["value"]) for o in js.get("observations", []) if o["value"] not in (".", "")]
        s = pd.Series({pd.Timestamp(d): float(v) for d, v in obs}, dtype=float).sort_index()
        s.name = series_id
        return s

    return cached(f"fred:{series_id}", 6 * 3600, load)


def _try(series_id: str) -> pd.Series | None:
    try:
        s = fred_series(series_id)
        return s if len(s) else None
    except Exception as e:
        log.debug("FRED %s: %s", series_id, e)
        return None


def short_rate(ccy: str) -> pd.Series | None:
    if ccy == "USD":
        return _try("DFF")
    if ccy == "EUR":
        s = _try("ECBDFR")
        return s if s is not None else _try("IR3TIB01EZM156N")
    cc = COUNTRY.get(ccy)
    return _try(f"IR3TIB01{cc}M156N") if cc else None


def long_rate(ccy: str) -> pd.Series | None:
    if ccy == "USD":
        return _try("DGS10")
    cc = LONG_RATE_CC.get(ccy)
    return _try(f"IRLTLT01{cc}M156N") if cc else None


def change(s: pd.Series | None, days: int) -> float | None:
    if s is None or len(s) < 2:
        return None
    end = s.index[-1]
    past = s[s.index <= end - pd.Timedelta(days=days)]
    if past.empty:
        return None
    return float(s.iloc[-1] - past.iloc[-1])


def latest(s: pd.Series | None) -> float | None:
    return None if s is None or s.empty else float(s.iloc[-1])


def cot(asset: str) -> dict | None:
    code = COT_CODES.get(asset)
    if not code:
        return None

    def load():
        rows = get_json("https://publicreporting.cftc.gov/resource/6dca-aqww.json", {
            "cftc_contract_market_code": code,
            "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": 60,
        })
        if not rows:
            return None
        net = [float(r["noncomm_positions_long_all"]) - float(r["noncomm_positions_short_all"]) for r in rows]
        oi = [float(r.get("open_interest_all") or 1) for r in rows]
        lo, hi = min(net[:52]), max(net[:52])
        idx = 50.0 if hi == lo else 100 * (net[0] - lo) / (hi - lo)
        return {
            "report_date": rows[0]["report_date_as_yyyy_mm_dd"][:10],
            "net_speculative": net[0],
            "net_change_4w": net[0] - net[min(4, len(net) - 1)],
            "net_pct_oi": net[0] / oi[0] if oi[0] else 0,
            "cot_index_52w": idx,  # 0 = most bearish of the year, 100 = most bullish
        }

    try:
        return cached(f"cot:{asset}", 12 * 3600, load)
    except Exception as e:
        log.debug("COT %s: %s", asset, e)
        return None


def crypto_fear_greed() -> dict | None:
    try:
        js = cached("fng", 3600, lambda: get_json("https://api.alternative.me/fng/", {"limit": 7}))
        d = js["data"]
        return {"value": int(d[0]["value"]), "label": d[0]["value_classification"],
                "week_ago": int(d[-1]["value"])}
    except Exception as e:
        log.debug("F&G: %s", e)
        return None


def risk_backdrop() -> dict:
    vix = _try("VIXCLS")
    dxy = _try("DTWEXBGS")
    real10 = _try("DFII10")
    curve = _try("T10Y2Y")
    return {
        "vix": latest(vix), "vix_chg_20d": change(vix, 28),
        "usd_index_chg_30d_pct": (None if dxy is None or change(dxy, 30) is None
                                  else 100 * change(dxy, 30) / float(dxy.iloc[-1])),
        "real_yield_10y": latest(real10), "real_yield_chg_30d": change(real10, 30),
        "curve_10y2y": latest(curve),
    }
