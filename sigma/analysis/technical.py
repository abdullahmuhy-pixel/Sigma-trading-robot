"""Technical analysis: turns candles into component scores in [-1, 1]
(+ = bullish, - = bearish) plus context the signal builder needs."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import indicators as ta

COMPONENTS = ["trend", "htf_trend", "momentum", "structure", "mean_reversion",
              "levels", "pattern", "divergence"]


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return float(max(lo, min(hi, x)))


def trend_score(df: pd.DataFrame) -> tuple[float, dict]:
    c = df["close"]
    e20, e50, e200 = ta.ema(c, 20), ta.ema(c, 50), ta.ema(c, 200)
    a = ta.atr(df).iloc[-1]
    parts = [np.sign(c.iloc[-1] - e50.iloc[-1]), np.sign(e20.iloc[-1] - e50.iloc[-1])]
    if not np.isnan(e200.iloc[-1]):
        parts.append(np.sign(e50.iloc[-1] - e200.iloc[-1]))
    if len(e50.dropna()) > 10 and a:
        parts.append(math.tanh((e50.iloc[-1] - e50.iloc[-11]) / a))
    st = ta.supertrend(df).iloc[-1]
    if st != 0:
        parts.append(st)
    parts = [p for p in parts if not np.isnan(p)]
    return _clip(float(np.mean(parts)) if parts else 0.0), {
        "ema20": float(e20.iloc[-1]), "ema50": float(e50.iloc[-1]),
        "ema200": None if np.isnan(e200.iloc[-1]) else float(e200.iloc[-1]), "supertrend": int(st)}


def structure_score(df: pd.DataFrame) -> tuple[float, str]:
    highs, lows = ta.swings(df.iloc[-120:])
    if len(highs) < 2 or len(lows) < 2:
        return 0.0, "unclear"
    hh = highs[-1][1] > highs[-2][1]
    hl = lows[-1][1] > lows[-2][1]
    close = df["close"].iloc[-1]
    s = 0.0
    label = "mixed"
    if hh and hl:
        s, label = 0.7, "higher highs & higher lows"
    elif not hh and not hl:
        s, label = -0.7, "lower highs & lower lows"
    if close > highs[-1][1]:
        s += 0.3
        label += ", broke above last swing high"
    elif close < lows[-1][1]:
        s -= 0.3
        label += ", broke below last swing low"
    return _clip(s), label


def analyze(df: pd.DataFrame, htf: pd.DataFrame | None = None) -> dict:
    """df: closed candles of the entry timeframe (>= 60 bars, ideally 300+)."""
    if len(df) < 60:
        raise ValueError("need at least 60 candles")
    c = df["close"]
    price = float(c.iloc[-1])
    a_series = ta.atr(df)
    a = float(a_series.iloc[-1])
    notes: list[str] = []

    trend, tctx = trend_score(df)
    htf_trend = 0.0
    if htf is not None and len(htf) >= 60:
        htf_trend, _ = trend_score(htf)

    r = ta.rsi(c)
    rsi_now = float(r.iloc[-1])
    _, _, hist = ta.macd(c)
    h_now, h_prev = float(hist.iloc[-1]), float(hist.iloc[-2])
    momentum = _clip(0.5 * _clip((rsi_now - 50) / 20) + 0.3 * np.sign(h_now)
                     + 0.2 * math.tanh((h_now - h_prev) / (0.05 * a + 1e-12)))

    structure, slabel = structure_score(df)

    lo, mid, up = ta.bollinger(c)
    width = float(up.iloc[-1] - lo.iloc[-1]) or 1e-12
    pct_b = (price - float(lo.iloc[-1])) / width
    k, _ = ta.stochastic(df)
    k_now = float(k.iloc[-1]) if not np.isnan(k.iloc[-1]) else 50.0
    stretch = 0.6 * _clip((pct_b - 0.5) * 2) + 0.4 * _clip((k_now - 50) / 40)
    mean_reversion = -_clip(stretch) if abs(stretch) > 0.5 else -0.4 * stretch

    lv = ta.levels(df.iloc[-300:], a)
    supports = sorted([l for l in lv if l["price"] < price - 0.1 * a], key=lambda l: price - l["price"])
    resists = sorted([l for l in lv if l["price"] > price + 0.1 * a], key=lambda l: l["price"] - price)
    ds = (price - supports[0]["price"]) / a if supports else 5.0
    dr = (resists[0]["price"] - price) / a if resists else 5.0
    levels_score = _clip((dr - ds) / (dr + ds + 1e-12))

    pats = ta.candle_patterns(df)
    pattern = _clip(sum(v for v in pats.values()) / 2)
    if pats:
        near = ds < 1.0 if pattern > 0 else dr < 1.0
        if near:
            pattern = _clip(pattern * 1.5)
            notes.append("candle pattern printed at a key level")

    divergence = float(ta.rsi_divergence(df, r))

    adx_v, pdi, mdi = ta.adx(df)
    adx_now = float(adx_v.iloc[-1]) if not np.isnan(adx_v.iloc[-1]) else 20.0
    regime = "trending" if adx_now >= 25 else "ranging" if adx_now < 20 else "transition"
    vol_pct = ta.percentile_rank(a_series)

    d1 = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    piv = ta.pivots(*d1.iloc[-2][["high", "low", "close"]]) if len(d1) >= 2 else {}

    highs, lows = ta.swings(df.iloc[-150:])
    fib = None
    if highs and lows:
        hi_i, hi_p = highs[-1]
        lo_i, lo_p = lows[-1]
        rng = hi_p - lo_p
        if rng > 0:
            up_leg = lo_i < hi_i
            fib = {"from": lo_p if up_leg else hi_p, "to": hi_p if up_leg else lo_p,
                   "levels": {str(f): (hi_p - f * rng) if up_leg else (lo_p + f * rng)
                              for f in (0.382, 0.5, 0.618, 0.786)}}

    if divergence:
        notes.append(f"{'bullish' if divergence > 0 else 'bearish'} RSI divergence")
    if vol_pct > 90:
        notes.append("volatility in the top 10% of the last 200 bars")
    if vol_pct < 10:
        notes.append("volatility unusually compressed - breakout risk")

    return {
        "price": price, "atr": a, "regime": regime, "adx": adx_now, "vol_percentile": vol_pct,
        "components": {"trend": trend, "htf_trend": htf_trend, "momentum": momentum,
                       "structure": structure, "mean_reversion": _clip(mean_reversion),
                       "levels": levels_score, "pattern": pattern, "divergence": divergence},
        "context": {
            **tctx, "rsi": rsi_now, "macd_hist": h_now, "bb_pct_b": pct_b, "stoch_k": k_now,
            "plus_di": float(pdi.iloc[-1]), "minus_di": float(mdi.iloc[-1]),
            "structure": slabel, "patterns": pats, "pivots": piv, "fib": fib,
            "nearest_support": supports[0]["price"] if supports else None,
            "nearest_resistance": resists[0]["price"] if resists else None,
            "levels": [{"price": l["price"], "touches": l["touches"]} for l in lv],
            "last_swing_high": highs[-1][1] if highs else None,
            "last_swing_low": lows[-1][1] if lows else None,
        },
        "notes": notes,
    }
