"""Instrument catalogue: one canonical name per instrument, mapped to every
provider's own symbol. Unknown names are inferred where possible."""
from __future__ import annotations

from dataclasses import dataclass

TIMEFRAMES = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}
HIGHER_TF = {"M1": "M15", "M5": "H1", "M15": "H4", "M30": "H4", "H1": "H4", "H4": "D1", "D1": "D1"}
TD_INTERVAL = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1day"}
BINANCE_INTERVAL = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m", "H1": "1h", "H4": "4h", "D1": "1d"}

FX = {"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "ZAR", "MXN", "SEK", "NOK", "SGD", "TRY", "PLN", "HKD"}
METALS = {"XAU", "XAG", "XPT", "XPD"}
CRYPTO = {"BTC", "ETH", "LTC", "XRP", "SOL", "BNB", "ADA", "DOGE", "BCH", "DOT", "LINK", "AVAX"}


@dataclass(frozen=True)
class Instrument:
    name: str
    asset_class: str          # forex | metal | crypto | synthetic | index | stock
    base: str = ""
    quote: str = ""
    deriv: str | None = None
    twelvedata: str | None = None
    binance: str | None = None

    @property
    def has_fundamentals(self) -> bool:
        return self.asset_class in ("forex", "metal", "crypto", "index", "stock")

    @property
    def display(self) -> str:
        return f"{self.base}/{self.quote}" if self.base and self.quote else self.name


SYNTHETICS = {
    "R_10": "Volatility 10 Index", "R_25": "Volatility 25 Index", "R_50": "Volatility 50 Index",
    "R_75": "Volatility 75 Index", "R_100": "Volatility 100 Index",
    "1HZ10V": "Volatility 10 (1s) Index", "1HZ25V": "Volatility 25 (1s) Index",
    "1HZ50V": "Volatility 50 (1s) Index", "1HZ75V": "Volatility 75 (1s) Index",
    "1HZ100V": "Volatility 100 (1s) Index",
    "BOOM300N": "Boom 300 Index", "BOOM500": "Boom 500 Index", "BOOM1000": "Boom 1000 Index",
    "CRASH300N": "Crash 300 Index", "CRASH500": "Crash 500 Index", "CRASH1000": "Crash 1000 Index",
    "stpRNG": "Step Index", "JD10": "Jump 10 Index", "JD25": "Jump 25 Index",
    "JD50": "Jump 50 Index", "JD75": "Jump 75 Index", "JD100": "Jump 100 Index",
}

INDICES = {  # canonical -> (twelvedata symbol, deriv symbol)
    "SPX": ("SPX", "OTC_SPC"), "NDX": ("NDX", "OTC_NDX"), "DJI": ("DJI", "OTC_DJI"),
    "DAX": ("DAX", "OTC_GDAXI"), "FTSE": ("FTSE", "OTC_FTSE"), "NIKKEI": ("N225", "OTC_N225"),
}


def resolve(name: str) -> Instrument:
    raw = name.strip()
    n = raw.upper().replace("/", "").replace("-", "")
    if raw in SYNTHETICS or n in {k.upper() for k in SYNTHETICS}:
        key = next(k for k in SYNTHETICS if k.upper() == n or k == raw)
        return Instrument(key, "synthetic", deriv=key)
    if n in INDICES:
        td, dv = INDICES[n]
        return Instrument(n, "index", deriv=dv, twelvedata=td)
    if len(n) == 6 or (len(n) in (6, 7) and n[:3] in CRYPTO | METALS | FX):
        base, quote = n[:3], n[3:]
        if quote == "USDT":
            quote = "USD"
        if base in FX and quote in FX:
            return Instrument(base + quote, "forex", base, quote, deriv=f"frx{base}{quote}",
                              twelvedata=f"{base}/{quote}")
        if base in METALS:
            return Instrument(base + quote, "metal", base, quote, deriv=f"frx{base}{quote}",
                              twelvedata=f"{base}/{quote}")
        if base in CRYPTO:
            return Instrument(base + quote, "crypto", base, quote, deriv=f"cry{base}{quote}",
                              twelvedata=f"{base}/{quote}", binance=f"{base}USDT")
    # Fallback: treat as a stock / anything Twelve Data knows
    return Instrument(n, "stock", twelvedata=raw.upper())


def tf_seconds(tf: str) -> int:
    if tf not in TIMEFRAMES:
        raise ValueError(f"Unknown timeframe {tf}. Use one of {list(TIMEFRAMES)}")
    return TIMEFRAMES[tf]
