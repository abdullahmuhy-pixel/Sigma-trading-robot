"""Where the retail crowd is positioned, from two free sources (each optional):

  Myfxbook Community Outlook  - long/short % of thousands of real trading accounts, plus the average
                                price the longs and shorts got in at. Free account; free tier allows
                                100 requests a day, so the whole table is fetched at most every 20 min.
                                Env: MYFXBOOK_EMAIL, MYFXBOOK_PASSWORD
  IG Client Sentiment         - % of IG clients long/short per market, through IG's API.
                                Env: IG_API_KEY, IG_USERNAME, IG_PASSWORD, IG_ACCOUNT_TYPE (live|demo)
                                Sigma only READS sentiment with this login; it never trades on IG.

Retail traders as a group tend to be on the wrong side of moves, so the score is CONTRARIAN:
crowd mostly long -> negative score (bearish), crowd mostly short -> positive. The learning model
decides how much weight this deserves from Sigma's own results.
"""
from __future__ import annotations

import logging
import time

from .. import config, db
from ..symbols import resolve
from .http import _session, cached

log = logging.getLogger("sigma.positioning")

FULL_TILT = 25.0           # 75% / 25% one way = the full contrarian score
BACKOFF = 600
_fail_until: dict[str, float] = {}
_lock_state: dict = {}


def contrarian_score(long_pct: float) -> float:
    return max(-1.0, min(1.0, -(float(long_pct) - 50.0) / FULL_TILT))


def _guard(key: str, fn, empty):
    if time.time() < _fail_until.get(key, 0):
        return empty
    try:
        return fn()
    except Exception as e:
        _fail_until[key] = time.time() + BACKOFF
        log.info("%s unavailable: %s", key, e)
        return empty


# ------------------------------------------------------------------------------- Myfxbook ---
MYFXBOOK = "https://www.myfxbook.com/api"


def _myfxbook_session(renew: bool = False) -> str:
    if not renew and _lock_state.get("mfx_session"):
        return _lock_state["mfx_session"]
    r = _session.get(f"{MYFXBOOK}/login.json", timeout=20,
                     params={"email": config.MYFXBOOK_EMAIL, "password": config.MYFXBOOK_PASSWORD})
    js = r.json()
    if js.get("error") in (True, "true", 1) or not js.get("session"):
        raise RuntimeError(f"Myfxbook login failed: {js.get('message')}")
    _lock_state["mfx_session"] = js["session"]
    return js["session"]


def parse_myfxbook(js: dict) -> dict[str, dict]:
    out = {}
    for s in js.get("symbols") or []:
        name = str(s.get("name", "")).upper().replace("/", "")
        try:
            lp = float(s["longPercentage"])
        except (KeyError, TypeError, ValueError):
            continue
        out[name] = {"long_pct": lp, "short_pct": float(s.get("shortPercentage", 100 - lp)),
                     "avg_long_price": _f(s.get("avgLongPrice")), "avg_short_price": _f(s.get("avgShortPrice")),
                     "long_positions": s.get("longPositions"), "short_positions": s.get("shortPositions")}
    return out


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def myfxbook_outlook() -> dict[str, dict]:
    if not (config.MYFXBOOK_EMAIL and config.MYFXBOOK_PASSWORD):
        return {}

    def load():
        for attempt in range(2):
            sid = _myfxbook_session(renew=attempt > 0)
            js = _session.get(f"{MYFXBOOK}/get-community-outlook.json", params={"session": sid}, timeout=20).json()
            if js.get("error") in (True, "true", 1):
                if "session" in str(js.get("message", "")).lower() and attempt == 0:
                    continue          # session expired: log in again once
                raise RuntimeError(js.get("message"))
            return parse_myfxbook(js)
        return {}

    return _guard("myfxbook", lambda: cached("myfxbook_outlook", 1200, load), {})


# ------------------------------------------------------------------------------------- IG ---
def _ig_base() -> str:
    return "https://demo-api.ig.com/gateway/deal" if config.IG_ACCOUNT_TYPE == "demo" else "https://api.ig.com/gateway/deal"


def _ig_headers(version: str = "1") -> dict:
    h = {"X-IG-API-KEY": config.IG_API_KEY, "Accept": "application/json; charset=UTF-8",
         "Content-Type": "application/json; charset=UTF-8", "Version": version}
    if _lock_state.get("ig_cst"):
        h.update({"CST": _lock_state["ig_cst"], "X-SECURITY-TOKEN": _lock_state["ig_xst"]})
    return h


def _ig_login() -> None:
    _lock_state.pop("ig_cst", None)
    r = _session.post(f"{_ig_base()}/session", headers=_ig_headers("2"), timeout=20,
                      json={"identifier": config.IG_USERNAME, "password": config.IG_PASSWORD})
    if r.status_code != 200 or "CST" not in r.headers:
        raise RuntimeError(f"IG login failed ({r.status_code}): {r.text[:160]}")
    _lock_state.update(ig_cst=r.headers["CST"], ig_xst=r.headers.get("X-SECURITY-TOKEN", ""), ig_at=time.time())


def _ig_get(path: str, version: str = "1", params: dict | None = None) -> dict:
    if not _lock_state.get("ig_cst") or time.time() - _lock_state.get("ig_at", 0) > 5 * 3600:
        _ig_login()
    r = _session.get(f"{_ig_base()}{path}", headers=_ig_headers(version), params=params, timeout=20)
    if r.status_code in (401, 403) and "token" in r.text.lower():
        _ig_login()
        r = _session.get(f"{_ig_base()}{path}", headers=_ig_headers(version), params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"IG {path}: {r.status_code} {r.text[:160]}")
    return r.json()


def _ig_search_terms(inst) -> list[str]:
    terms = []
    if inst.base and inst.quote:
        terms.append(f"{inst.base}/{inst.quote}")
    terms += [inst.display, inst.name]
    return [t for i, t in enumerate(terms) if t and t not in terms[:i]]


def ig_market_id(name: str) -> str | None:
    """IG's sentiment uses its own market ids; find one by searching IG's markets (remembered)."""
    inst = resolve(name)
    ids = db.kv_get("ig_market_ids") or {}
    if inst.name in ids:
        return ids[inst.name]
    found = None
    for term in _ig_search_terms(inst):
        mkts = _ig_get("/markets", "1", {"searchTerm": term}).get("markets") or []
        for m in mkts[:5]:
            epic = m.get("epic")
            if not epic:
                continue
            det = _ig_get(f"/markets/{epic}", "3")
            found = (det.get("instrument") or {}).get("marketId")
            if found:
                break
        if found:
            break
    ids[inst.name] = found            # remember misses too, so we don't search again
    db.kv_set("ig_market_ids", ids)
    return found


def ig_sentiment(name: str) -> dict | None:
    if not (config.IG_API_KEY and config.IG_USERNAME and config.IG_PASSWORD):
        return None

    def load():
        mid = ig_market_id(name)
        if not mid:
            return None
        js = _ig_get(f"/clientsentiment/{mid}")
        lp = float(js["longPositionPercentage"])
        return {"long_pct": lp, "short_pct": float(js.get("shortPositionPercentage", 100 - lp)), "market_id": mid}

    return _guard("ig", lambda: cached(f"ig_sentiment:{resolve(name).name}", 900, load), None)


# ------------------------------------------------------------------------------ combined ---
def analyze(name: str) -> dict:
    """{"score": contrarian -1..1, "sources": {...}, "notes": [...]}; score 0 when nothing is available."""
    inst = resolve(name)
    if inst.asset_class in ("synthetic", "basket"):
        return {"score": 0.0, "sources": {}, "notes": []}
    sources = {}
    mfx = myfxbook_outlook().get(inst.name.replace("/", "").upper())
    if mfx:
        sources["myfxbook"] = mfx
    ig = ig_sentiment(inst.name)
    if ig:
        sources["ig"] = ig
    derivs = None
    if inst.asset_class == "crypto":
        from . import crypto_derivs
        derivs = crypto_derivs.analyze(inst.name)
    if not sources:
        if derivs:
            return {"score": derivs["score"], "sources": {"futures": derivs}, "notes": derivs["notes"]}
        return {"score": 0.0, "sources": {}, "notes": []}
    long_pct = sum(s["long_pct"] for s in sources.values()) / len(sources)
    score = contrarian_score(long_pct)
    if derivs:                                   # crypto: retail crowd and futures positioning, equally
        score = (score + derivs["score"]) / 2
        sources["futures"] = derivs
    notes = [f"Retail crowd {long_pct:.0f}% long / {100 - long_pct:.0f}% short "
             f"({' + '.join({'myfxbook': 'Myfxbook', 'ig': 'IG'}[k] for k in sources)}): "
             + ("crowded long - contrarian bearish" if score <= -0.4 else
                "crowded short - contrarian bullish" if score >= 0.4 else "no strong crowding")]
    if mfx and mfx.get("avg_long_price") and mfx.get("avg_short_price"):
        notes.append(f"Average crowd entries: longs {mfx['avg_long_price']}, shorts {mfx['avg_short_price']} "
                     "(trapped traders' stops often sit beyond these)")
    if derivs:
        notes += derivs["notes"]
    return {"score": round(score, 3), "long_pct": round(long_pct, 1), "sources": sources, "notes": notes}


def configured() -> dict:
    return {"myfxbook": bool(config.MYFXBOOK_EMAIL and config.MYFXBOOK_PASSWORD),
            "ig": bool(config.IG_API_KEY and config.IG_USERNAME and config.IG_PASSWORD)}
