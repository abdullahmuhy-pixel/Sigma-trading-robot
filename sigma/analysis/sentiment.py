"""News sentiment per instrument and the event-risk filter."""
from __future__ import annotations

import math
import re
import time

from .. import config
from ..data import news
from ..symbols import Instrument

KEYWORDS = {
    "USD": ["dollar", "usd", "fed ", "fomc", "powell", "federal reserve", "u.s", "treasury", "nonfarm", "payrolls"],
    "EUR": ["euro", "eur", "ecb", "lagarde", "eurozone", "euro area", "germany", "german"],
    "GBP": ["pound", "sterling", "gbp", "boe", "bank of england", "bailey", "uk ", "britain"],
    "JPY": ["yen", "jpy", "boj", "bank of japan", "ueda", "japan"],
    "AUD": ["aussie", "aud", "rba", "australia"],
    "NZD": ["kiwi", "nzd", "rbnz", "new zealand"],
    "CAD": ["loonie", "cad", "bank of canada", "boc ", "canada", "oil"],
    "CHF": ["franc", "chf", "snb", "swiss"],
    "ZAR": ["rand", "zar", "sarb", "south africa", "kganyago"],
    "XAU": ["gold", "xau", "bullion", "precious metal"],
    "XAG": ["silver", "xag"],
    "BTC": ["bitcoin", "btc", "crypto"],
    "ETH": ["ethereum", "eth ", "ether"],
    "OIL": ["oil", "crude", "brent", "wti", "opec"],
}
HAWKISH = ["hike", "hawkish", "tighten", "higher for longer", "hot inflation", "inflation rises",
           "inflation accelerat", "strong jobs", "beats", "beat expectations", "surge", "rally", "rallies",
           "gains", "jumps", "soars", "strengthens", "upgrade", "record high", "robust", "rebound", "approval", "inflows"]
DOVISH = ["cut", "dovish", "easing", "ease", "slowdown", "recession", "contraction", "weak", "misses",
          "miss expectations", "falls", "slumps", "drops", "plunge", "tumbles", "weakens", "downgrade", "layoffs",
          "unemployment rises", "crisis", "default", "outflows", "hack", "ban", "lawsuit", "selloff", "sell-off"]


def _mentions(text: str, ccy: str) -> bool:
    t = text.lower()
    return any(re.search(r"\b" + re.escape(k.strip()) + r"\b", t) for k in KEYWORDS.get(ccy, [ccy.lower()]))


def _tone(text: str) -> float:
    t = text.lower()
    pos = sum(len(re.findall(r"\b" + re.escape(w), t)) for w in HAWKISH)
    neg = sum(len(re.findall(r"\b" + re.escape(w), t)) for w in DOVISH)
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


def score_headlines(items: list[dict], base: str, quote: str = "", now: float | None = None) -> tuple[float, list[dict]]:
    now = now or time.time()
    total, used = 0.0, []
    for it in items:
        text = f"{it['title']} {it.get('summary', '')}"
        sign = 0
        if base and _mentions(text, base):
            sign += 1
        if quote and _mentions(text, quote):
            sign -= 1
        if sign == 0:
            continue
        tone = _tone(it["title"]) * 0.7 + _tone(it.get("summary", "")) * 0.3
        if tone == 0:
            continue
        age_h = max(0.0, (now - it["published"]) / 3600)
        w = math.exp(-age_h / 12)
        total += sign * tone * w
        used.append({"title": it["title"], "url": it.get("url"), "tone": round(tone, 2),
                     "age_hours": round(age_h, 1), "source": it.get("source")})
    return math.tanh(total / 3), used[:12]


def analyze(inst: Instrument) -> dict:
    if not inst.has_fundamentals:
        return {"score": 0.0, "available": False, "headlines": []}
    base = inst.base or ("OIL" if "oil" in inst.display.lower() else inst.name)
    quote = inst.quote if inst.quote and inst.asset_class == "forex" else ("USD" if inst.asset_class == "metal" else "")
    items = news.headlines()
    score, used = score_headlines(items, base, quote)
    av_ticker = {"forex": f"FOREX:{base}", "crypto": f"CRYPTO:{base}"}.get(inst.asset_class)
    if av_ticker:
        av = news.alphavantage_sentiment(av_ticker)
        if av:
            av_mean = sum(a["score"] for a in av[:20]) / min(20, len(av))
            score = 0.5 * score + 0.5 * max(-1, min(1, av_mean * 3))
    return {"score": score, "available": bool(used), "headlines": used}


def event_risk(inst: Instrument, now: float | None = None) -> dict:
    now = now or time.time()
    ccys = {c for c in (inst.base, inst.quote) if c}
    if inst.asset_class in ("metal", "crypto", "index", "stock", "commodity"):
        ccys.add("USD")
    if inst.asset_class in ("synthetic", "basket"):
        return {"blocked": False, "upcoming": []}
    upcoming = [e for e in news.calendar()
                if e["currency"] in ccys and -1800 <= e["time"] - now <= 24 * 3600]
    high = [e for e in upcoming if e["impact"].lower() == "high"]
    blocked = [e for e in high if abs(e["time"] - now) <= config.NEWS_BLACKOUT_MIN * 60]
    return {"blocked": bool(blocked), "blocking_event": blocked[0] if blocked else None,
            "upcoming": upcoming[:15]}
