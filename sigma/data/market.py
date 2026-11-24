"""Price data from free sources, normalised to one DataFrame shape:
index = UTC timestamp, columns = open, high, low, close, volume.

Routing (first that works wins):
  synthetic -> Deriv
  forex / metal -> Deriv -> Twelve Data
  crypto -> Binance -> Deriv -> Twelve Data
  index -> Twelve Data -> Deriv
  stock -> Twelve Data
"""
from __future__ import annotations

import json
import logging
import threading
import time

import pandas as pd

from .. import config
from ..symbols import BINANCE_INTERVAL, TD_INTERVAL, Instrument, resolve, tf_seconds
from .http import RateLimiter, cached, get_json

log = logging.getLogger("sigma.market")

DERIV_PUBLIC_WS = "wss://api.derivws.com/trading/v1/options/ws/public"
DERIV_LEGACY_WS = "wss://ws.derivws.com/websockets/v3?app_id={app_id}"

_td_limit = RateLimiter(8)
_deriv_limit = RateLimiter(180)    # Deriv allows 220 calls/min; stay under it          # Twelve Data free plan: 8 credits / minute
_deriv_slots = threading.BoundedSemaphore(6)  # parallel Deriv connections (chart never waits for a scan)


class DataError(RuntimeError):
    pass


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        raise DataError("no candles returned")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["open", "high", "low", "close"])
    return df[["open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------- Deriv ----
def _deriv_request(payload: dict, timeout: float = 15) -> dict:
    try:
        import websocket  # websocket-client
    except ImportError as e:  # pragma: no cover
        raise DataError("websocket-client not installed") from e
    last_err: Exception | None = None
    for url in (DERIV_PUBLIC_WS, DERIV_LEGACY_WS.format(app_id=config.DERIV_APP_ID)):
        try:
            _deriv_limit.wait()
            with _deriv_slots:
                ws = websocket.create_connection(url, timeout=timeout)
                try:
                    payload = {**payload, "req_id": 1}
                    ws.send(json.dumps(payload))
                    deadline = time.time() + timeout
                    while time.time() < deadline:
                        msg = json.loads(ws.recv())
                        if msg.get("req_id") == 1 or "error" in msg:
                            if "error" in msg:
                                raise DataError(msg["error"].get("message", "Deriv error"))
                            return msg
                finally:
                    ws.close()
        except Exception as e:  # try next endpoint
            last_err = e
            log.debug("Deriv %s failed: %s", url, e)
    raise DataError(f"Deriv unavailable: {last_err}")


def deriv_candles(symbol: str, tf: str, count: int = 500, end: int | None = None) -> pd.DataFrame:
    msg = _deriv_request({
        "ticks_history": symbol, "style": "candles", "granularity": tf_seconds(tf),
        "count": min(count, 5000), "end": end or "latest", "adjust_start_time": 1,
    })
    rows = [{"time": c["epoch"], "open": c.get("open"), "high": c.get("high"),
             "low": c.get("low"), "close": c.get("close"), "volume": 0} for c in msg.get("candles", [])]
    return _frame(rows)


def deriv_active_symbols() -> list[dict]:
    msg = _deriv_request({"active_symbols": "brief"})
    return msg.get("active_symbols", [])


# ---------------------------------------------------------- Twelve Data ----
def twelvedata_candles(symbol: str, tf: str, count: int = 500) -> pd.DataFrame:
    if not config.TWELVEDATA_API_KEY:
        raise DataError("TWELVEDATA_API_KEY not set")
    _td_limit.wait()
    js = get_json("https://api.twelvedata.com/time_series", {
        "symbol": symbol, "interval": TD_INTERVAL[tf], "outputsize": min(count, 5000),
        "apikey": config.TWELVEDATA_API_KEY, "timezone": "UTC", "order": "ASC",
    })
    if js.get("status") == "error":
        raise DataError(js.get("message", "Twelve Data error"))
    rows = [{"time": pd.Timestamp(v["datetime"], tz="UTC").timestamp(), "open": v["open"],
             "high": v["high"], "low": v["low"], "close": v["close"], "volume": v.get("volume", 0)}
            for v in js.get("values", [])]
    return _frame(rows)


# -------------------------------------------------------------- Binance ----
def binance_candles(symbol: str, tf: str, count: int = 500, end_ms: int | None = None) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": BINANCE_INTERVAL[tf], "limit": min(count, 1000)}
    if end_ms:
        params["endTime"] = end_ms
    last = None
    for host in ("https://data-api.binance.vision", "https://api.binance.com"):
        try:
            js = get_json(f"{host}/api/v3/klines", params)
            rows = [{"time": k[0] / 1000, "open": k[1], "high": k[2], "low": k[3],
                     "close": k[4], "volume": k[5]} for k in js]
            return _frame(rows)
        except Exception as e:
            last = e
    raise DataError(f"Binance unavailable: {last}")


# ------------------------------------------------------------- Routing -----
def _routes(inst: Instrument):
    c = inst.asset_class
    if c in ("synthetic", "basket", "commodity"):
        return [("deriv", inst.deriv)]
    if c in ("forex", "metal"):
        return [("deriv", inst.deriv), ("twelvedata", inst.twelvedata)]
    if c == "crypto":
        # Deriv first so prices match what you trade on Deriv; Binance as backup
        return [("deriv", inst.deriv), ("binance", inst.binance), ("twelvedata", inst.twelvedata)]
    if c == "index":
        return [("deriv", inst.deriv), ("twelvedata", inst.twelvedata)]
    return [("twelvedata", inst.twelvedata)]


_FETCH = {"deriv": deriv_candles, "twelvedata": twelvedata_candles, "binance": binance_candles}


def candles(name: str, tf: str, count: int = 500) -> pd.DataFrame:
    """Most recent `count` candles (the last one may still be forming)."""
    inst = resolve(name)
    ttl = max(10, min(tf_seconds(tf) / 6, 300))

    def load():
        errors = []
        for provider, sym in _routes(inst):
            if not sym:
                continue
            try:
                df = _FETCH[provider](sym, tf, count)
                df.attrs["provider"] = provider
                return df
            except Exception as e:
                errors.append(f"{provider}: {e}")
        raise DataError(f"No data for {name} {tf} ({'; '.join(errors) or 'no route'})")

    return cached(f"candles:{inst.name}:{tf}:{count}", ttl, load)


def closed_candles(name: str, tf: str, count: int = 500) -> pd.DataFrame:
    """Candles with the still-forming bar removed, so analysis never repaints."""
    df = candles(name, tf, count + 1)
    if len(df) and df.index[-1].timestamp() + tf_seconds(tf) > time.time():
        df = df.iloc[:-1]
    return df.iloc[-count:]


def last_price(name: str, max_age: float = 2.0) -> float:
    """The latest traded price, at most `max_age` seconds old (used for the final order check)."""
    inst = resolve(name)

    def load():
        for provider, sym in _routes(inst):
            if not sym:
                continue
            try:
                return float(_FETCH[provider](sym, "M1", 2)["close"].iloc[-1])
            except Exception:
                continue
        return float(candles(name, "M1", 3)["close"].iloc[-1])

    return cached(f"last:{inst.name}", max_age, load)


def history(name: str, tf: str, bars: int = 3000) -> pd.DataFrame:
    """Longer history for backtests (pages backwards where the provider allows)."""
    inst = resolve(name)
    if inst.asset_class == "crypto" and inst.binance and not inst.deriv:
        frames, end_ms = [], None
        while sum(len(f) for f in frames) < bars:
            f = binance_candles(inst.binance, tf, 1000, end_ms)
            if f.empty:
                break
            frames.insert(0, f)
            end_ms = int(f.index[0].timestamp() * 1000) - 1
            if len(f) < 1000:
                break
        df = pd.concat(frames)
    elif inst.deriv:
        try:
            df = deriv_candles(inst.deriv, tf, min(bars, 5000))
        except DataError:
            df = twelvedata_candles(inst.twelvedata, tf, min(bars, 5000)) if inst.twelvedata else None
            if df is None:
                raise
    else:
        df = twelvedata_candles(inst.twelvedata, tf, min(bars, 5000))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.iloc[-bars:]
