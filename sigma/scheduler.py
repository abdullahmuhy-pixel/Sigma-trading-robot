"""Background loops: scan the watchlist at every candle close, and resolve
open paper trades. Runs inside the server process."""
from __future__ import annotations

import logging
import threading
import time

from . import config, engine, paper
from .symbols import tf_seconds

log = logging.getLogger("sigma.scheduler")
state = {"last_scan": {}, "last_monitor": None, "errors": [], "running": False, "scanning": None}


def _err(msg: str) -> None:
    log.warning(msg)
    state["errors"] = (state["errors"] + [{"t": time.time(), "msg": msg}])[-50:]


def scan(tf: str, symbols: list[str] | None = None) -> list[dict]:
    out = []
    for s in symbols or config.WATCHLIST:
        state["scanning"] = f"{s} {tf}"
        try:
            r = engine.analyze(s, tf, source="live")
            out.append(r)
            if r.get("is_signal"):
                from . import alerts
                alerts.consider(r)
        except Exception as e:
            _err(f"scan {s} {tf}: {e}")
    state["scanning"] = None
    state["last_scan"][tf] = time.time()
    return out


def _scanner() -> None:
    done: dict[str, int] = {}
    while state["running"]:
        now = time.time()
        for tf in config.SCAN_TIMEFRAMES:
            secs = tf_seconds(tf)
            bar = int(now // secs)
            # wait 20 s after the close so providers have the final candle
            if done.get(tf) != bar and now - bar * secs >= 20:
                if tf in done:  # skip the partial period at start-up
                    scan(tf)
                done[tf] = bar
        time.sleep(5)


def _monitor() -> None:
    while state["running"]:
        try:
            closed = paper.monitor_once()
            if closed:
                log.info("resolved %d paper trades", len(closed))
            from . import journal
            journal.grade_pending()
            if time.time() - state.get("last_prune", 0) > 6 * 3600:
                from . import db
                state["last_prune"] = time.time()
                state["db"] = db.prune()
        except Exception as e:
            _err(f"monitor: {e}")
        state["last_monitor"] = time.time()
        time.sleep(config.MONITOR_SECONDS)


def start() -> None:
    if state["running"]:
        return
    state["running"] = True
    threading.Thread(target=_monitor, daemon=True, name="monitor").start()
    if config.AUTO_SCAN:
        threading.Thread(target=_scanner, daemon=True, name="scanner").start()


def stop() -> None:
    state["running"] = False
