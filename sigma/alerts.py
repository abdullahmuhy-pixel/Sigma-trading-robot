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
            f"Entry {_fmt(r['entry'])} · Stop loss {_fmt(r['sl'])} · "
            + " · ".join(f"{t['name']} {_fmt(t['price'])} ({t['p_reach']:.0%})" for t in r.get("take_profits", []))
            + f" · expected {r['expectancy']:+.2f}R"
            f"{' · higher timeframes agree ' + d['higher_tf_alignment'] if d.get('higher_tf_alignment') else ''}\n\n"
            f"{ex.get('summary', '')}{hist}\n\n{ex.get('invalidation', '')}\n\n"
            f"_Paper-traded on the demo account. Ask me \"explain the {r['display']} alert\" for the full breakdown._")
    return title, body


_queue: "queue.Queue[dict]" = None  # type: ignore
_worker = None


def enqueue(scan_result: dict) -> None:
    """Deep-check a scan signal in the background, one at a time, so scanning never waits for it."""
    import queue
    import threading
    global _queue, _worker
    if _queue is None:
        _queue = queue.Queue()
    if _worker is None or not _worker.is_alive():
        def run():
            while True:
                r = _queue.get()
                try:
                    consider(r)
                except Exception as e:
                    log.warning("deep check failed: %s", e)
        _worker = threading.Thread(target=run, daemon=True, name="deep-checks")
        _worker.start()
    _queue.put(scan_result)


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
    _push("📡 " + title, r, "Deep-confirmed")
    return {"id": aid, "title": title, "body": body}


def _push(title: str, r: dict, kind: str) -> None:
    """Phone notification: the essentials fit in one line."""
    try:
        from . import push
        tps = " ".join(f"{t['name']} {_fmt(t['price'])}" for t in (r.get("take_profits") or [])[:3])
        push.notify(title, f"{kind}. Entry {_fmt(r['entry'])} · SL {_fmt(r['sl'])} · {tps} · "
                           f"P(win) {r['probability']:.0%}. Tap to open.", url="/#markets",
                    tag=f"{r['symbol']}-{r['timeframe']}")
    except Exception as e:
        log.debug("push skipped: %s", e)


def session_opportunity(session: str, r: dict) -> dict:
    """A fast-scan opportunity at a session open. Journalled (so its call is graded) but not traded:
    you decide whether to run the deep analysis."""
    from . import engine
    aid = r.get("analysis_id") or engine.record(r, "session")
    pl = r["plan"]
    title = f"{session} open: {pl['order_type']} {r['display']} ({r['timeframe']})"
    tps = " · ".join(f"{t['name']} {_fmt(t['price'])} ({t['p_reach']:.0%})" for t in r.get("take_profits", []))
    body = (f"**{title}**, {pl['setup_type'].lower()} (fast scan)\n\n"
            f"Entry {_fmt(r['entry'])} · Stop loss {_fmt(r['sl'])} · {tps} · P(win) {r['probability']:.0%} · "
            f"expected {r['expectancy']:+.2f}R\n\n{(r.get('explanation') or {}).get('summary', '')}\n\n"
            f"_Fast scan only - run the deep analysis before acting on it._")
    now = time.time()
    al_id = db.insert("alerts", {"created_at": now, "analysis_id": aid, "symbol": r["symbol"], "timeframe": r["timeframe"],
                                 "order_type": pl["order_type"], "title": title, "body": body, "kind": "session",
                                 "session": session})
    _push("⏰ " + title, r, "Fast scan, deep analysis on request")
    return {"id": al_id, "title": title, "symbol": r["symbol"], "timeframe": r["timeframe"]}


def session_summary(session: str, pairs: list[str], found: list[dict]) -> None:
    """One chat message per session open, so every device sees what the robot found."""
    if found:
        lines = "\n".join(f"- **{r['plan']['order_type']} {r['display']}** ({r['timeframe']}) at {_fmt(r['entry'])}, "
                          f"SL {_fmt(r['sl'])}, P(win) {r['probability']:.0%}, expected {r['expectancy']:+.2f}R"
                          for r in found)
        text = (f"**{session} session open: {len(found)} possible opportunit{'y' if len(found) == 1 else 'ies'}** "
                f"(fast scan of {', '.join(pairs)})\n\n{lines}\n\nSay \"deep analyse\" with the pair to confirm one, "
                f"or tap Deep analysis on the alert.")
    else:
        text = (f"**{session} session open:** fast scan of {', '.join(pairs)} found no clear opportunity. "
                f"I'll keep scanning at every candle close.")
    db.insert("chat", {"session": "default", "created_at": time.time(), "role": "assistant", "content": text})


def since(after_id: int = 0, limit: int = 50) -> list[dict]:
    if after_id:
        return db.q("SELECT * FROM alerts WHERE id>? ORDER BY id", (after_id,))
    return db.q("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))[::-1]
