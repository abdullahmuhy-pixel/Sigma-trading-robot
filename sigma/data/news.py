"""Headlines (RSS feeds, Finnhub, Alpha Vantage) and the economic calendar
(ForexFactory weekly JSON feed)."""
from __future__ import annotations

import email.utils
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from .. import config
from .http import cached, get_json, get_text

log = logging.getLogger("sigma.news")


def _parse_date(s: str | None) -> float:
    if not s:
        return time.time()
    try:
        return email.utils.parsedate_to_datetime(s).timestamp()
    except Exception:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except Exception:
            return time.time()


def _rss(url: str) -> list[dict]:
    txt = get_text(url)
    root = ET.fromstring(txt.encode("utf-8", "ignore"))
    out = []
    for item in root.iter():
        tag = item.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        get = lambda n: next((c.text or "" for c in item if c.tag.split("}")[-1] == n), "")  # noqa: E731
        link = get("link") or next((c.attrib.get("href", "") for c in item if c.tag.split("}")[-1] == "link"), "")
        desc = re.sub(r"<[^>]+>", " ", get("description") or get("summary"))
        out.append({
            "title": get("title").strip(), "summary": re.sub(r"\s+", " ", desc).strip()[:400],
            "url": link.strip(), "published": _parse_date(get("pubDate") or get("published") or get("updated")),
            "source": url.split("/")[2],
        })
    return out


def _finnhub() -> list[dict]:
    if not config.FINNHUB_API_KEY:
        return []
    out = []
    for cat in ("forex", "crypto", "general"):
        try:
            for n in get_json("https://finnhub.io/api/v1/news",
                              {"category": cat, "token": config.FINNHUB_API_KEY}):
                out.append({"title": n.get("headline", ""), "summary": n.get("summary", "")[:400],
                            "url": n.get("url", ""), "published": float(n.get("datetime", time.time())),
                            "source": "finnhub:" + n.get("source", "")})
        except Exception as e:
            log.debug("finnhub %s: %s", cat, e)
    return out


def headlines(max_age_hours: float = 48) -> list[dict]:
    def load():
        items: list[dict] = []
        for url in config.NEWS_FEEDS:
            try:
                items += _rss(url)
            except Exception as e:
                log.debug("feed %s failed: %s", url, e)
        items += _finnhub()
        seen, uniq = set(), []
        for it in sorted(items, key=lambda x: -x["published"]):
            k = it["title"].lower()[:90]
            if k and k not in seen:
                seen.add(k)
                uniq.append(it)
        return uniq

    cutoff = time.time() - max_age_hours * 3600
    return [i for i in cached("headlines", 600, load) if i["published"] >= cutoff]


def alphavantage_sentiment(tickers: str) -> list[dict]:
    """Scored news for e.g. 'FOREX:EUR' or 'CRYPTO:BTC' (25 calls/day free)."""
    if not config.ALPHAVANTAGE_API_KEY:
        return []

    def load():
        js = get_json("https://www.alphavantage.co/query", {
            "function": "NEWS_SENTIMENT", "tickers": tickers, "limit": 50,
            "apikey": config.ALPHAVANTAGE_API_KEY})
        out = []
        for a in js.get("feed", []):
            ts = [t for t in a.get("ticker_sentiment", []) if t.get("ticker") == tickers]
            score = float(ts[0]["ticker_sentiment_score"]) if ts else float(a.get("overall_sentiment_score", 0))
            out.append({"title": a.get("title", ""), "url": a.get("url", ""),
                        "score": score, "published": _parse_date(
                            datetime.strptime(a["time_published"], "%Y%m%dT%H%M%S")
                            .replace(tzinfo=timezone.utc).isoformat()) if a.get("time_published") else time.time()})
        return out

    try:
        return cached(f"av:{tickers}", 3 * 3600, load)
    except Exception as e:
        log.debug("alphavantage: %s", e)
        return []


def calendar() -> list[dict]:
    """This week's economic events: time (epoch), currency, impact, title, forecast, previous, actual."""
    def load():
        rows = get_json(config.CALENDAR_URL)
        out = []
        for r in rows:
            try:
                ts = datetime.fromisoformat(r["date"]).timestamp()
            except Exception:
                continue
            out.append({"time": ts, "currency": r.get("country", "").upper(), "impact": r.get("impact", ""),
                        "title": r.get("title", ""), "forecast": r.get("forecast", ""),
                        "previous": r.get("previous", ""), "actual": r.get("actual", "")})
        return sorted(out, key=lambda e: e["time"])

    try:
        return cached("calendar", 1800, load)
    except Exception as e:
        log.debug("calendar: %s", e)
        return []
