"""Shared HTTP helpers: a TTL cache and a polite per-host rate limiter."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

import requests

UA = {"User-Agent": "Mozilla/5.0 (SigmaTradingRobot; personal research)"}
_session = requests.Session()
_session.headers.update(UA)

_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def cached(key: str, ttl: float, fn: Callable[[], Any]) -> Any:
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _cache_lock:
        _cache[key] = (now, val)
    return val


class RateLimiter:
    """Allows `per_minute` calls in any rolling 60 s window (blocking)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.calls: list[float] = []
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            while True:
                now = time.time()
                self.calls = [t for t in self.calls if now - t < 60]
                if len(self.calls) < self.per_minute:
                    self.calls.append(now)
                    return
                time.sleep(60 - (now - self.calls[0]) + 0.05)


def get_json(url: str, params: dict | None = None, timeout: float = 15, headers: dict | None = None) -> Any:
    r = _session.get(url, params=params, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.json()


def get_text(url: str, timeout: float = 15) -> str:
    r = _session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text
