"""Opportunity alerts. When a scheduled scan finds a signal, a deep analysis double-checks it;
only confirmed opportunities are announced - in the chat (so every device sees them) and as an
alert the app shows as a notification."""
from __future__ import annotations

import logging
import time

from . import db, deep

log = logging.getLogger("sigma.alerts")


def _fmt(x):
    return f"{x:,.5f}" if abs(x) < 10 else f"{x:,.2f}"


def message(r: dict) -> tuple[str, str]:
    pl, ex, d = r["plan"], r.get("explanation") or {}, r.get("deep") or {}
    title = f"{pl['order_type']} {r['display']} ({r['timeframe']})"
    look = d.get("lookalike_history") or {}
    hist = (f" Similar past setups on this market: {look['found']}, won {look['win_rate']:.0%}."
            if look.get("found") and look.get("win_rate") is not None else "")
    body = (f"**Opportunity found: {title}**, {pl['setup_type'].lower()}\n\n"
            f"Entry {_fmt(r['entry'])} · Stop loss {_fmt(r['sl'])} · Take profit {_fmt(r['tp'])} · "
            f"P(win) {r['probability']:.0%} · expected {r['expectancy']:+.2f}R"
            f"{' · higher timeframes agree ' + d['higher_tf_alignment'] if d.get('higher_tf_alignment') else ''}\n\n"
            f"{ex.get('summary', '')}{hist}\n\n{ex.get('invalidation', '')}\n\n"
            f"_Paper-traded on the demo account. Ask me \"explain the {r['display']} alert\" for the full breakdown._")
    return title, body


def consider(scan_result: dict) -> dict | None:
    """Called for every scan result. Returns the alert if one was raised."""
    if not scan_result.get("is_signal") or not scan_result.get("analysis_id"):
        return None
    try:   # second opinion: the slower deep analysis must agree
        r = deep.analyze(scan_result["symbol"], scan_result["timeframe"], source="live", journal=False)
    except Exception as e:
        log.warning("deep check failed for %s: %s", scan_result["symbol"], e)
        return None
    if not r["is_signal"] or r["direction"] != scan_result["direction"]:
        log.info("deep analysis did not confirm %s %s", scan_result["symbol"], scan_result["timeframe"])
        return None
    title, body = message(r)
    now = time.time()
    aid = db.insert("alerts", {"created_at": now, "analysis_id": scan_result["analysis_id"], "symbol": r["symbol"],
                               "timeframe": r["timeframe"], "order_type": r["plan"]["order_type"],
                               "title": title, "body": body})
    db.insert("chat", {"session": "default", "created_at": now, "role": "assistant", "content": body})
    return {"id": aid, "title": title, "body": body}


def since(after_id: int = 0, limit: int = 50) -> list[dict]:
    if after_id:
        return db.q("SELECT * FROM alerts WHERE id>? ORDER BY id", (after_id,))
    return db.q("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))[::-1]
