"""Technical indicators on OHLC DataFrames (pure pandas / numpy).
Conventions match the common charting-platform definitions (Wilder smoothing
for RSI, ATR and ADX)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder's RMA, seeded with the simple mean of the first n values."""
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    valid = np.where(~np.isnan(vals))[0]
    if len(valid) < n:
        return pd.Series(out, index=s.index)
    start = valid[0]
    seed_end = start + n
    out[seed_end - 1] = np.nanmean(vals[start:seed_end])
    tail = vals[seed_end:]
    if not np.isnan(tail).any():
        # fast path: after the seed, Wilder's RMA is an EMA with alpha = 1/n
        x = np.concatenate([[out[seed_end - 1]], tail])
        out[seed_end - 1:] = pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()
        return pd.Series(out, index=s.index)
    for i in range(seed_end, len(vals)):
        v = vals[i]
        out[i] = out[i - 1] if np.isnan(v) else (out[i - 1] * (n - 1) + v) / n
    return pd.Series(out, index=s.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = wilder(d.clip(lower=0), n)
    loss = wilder((-d).clip(lower=0), n)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    """max(high-low, |high-prev close|, |low-prev close|); the first bar has no previous close."""
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    pc = np.empty_like(c)
    pc[0] = np.nan
    pc[1:] = c[:-1]
    with np.errstate(invalid="ignore"):
        tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr, index=df.index)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = true_range(df)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return wilder(tr, n)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def stochastic(df: pd.DataFrame, n: int = 14, d: int = 3):
    lo = df["low"].rolling(n, min_periods=n).min()
    hi = df["high"].rolling(n, min_periods=n).max()
    k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(d, min_periods=d).mean()


def adx(df: pd.DataFrame, n: int = 14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = true_range(df)
    tr_s = wilder(tr.iloc[1:], n).reindex(df.index)
    pdi = 100 * wilder(plus_dm.iloc[1:], n).reindex(df.index) / tr_s
    mdi = 100 * wilder(minus_dm.iloc[1:], n).reindex(df.index) / tr_s
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder(dx, n), pdi, mdi


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.Series:
    """+1 when price is above the SuperTrend line, -1 below."""
    a = atr(df, n)
    hl2 = (df["high"] + df["low"]) / 2
    upper, lower = (hl2 + mult * a).to_numpy(), (hl2 - mult * a).to_numpy()
    close = df["close"].to_numpy()
    fu, fl = upper.copy(), lower.copy()
    direction = np.zeros(len(df))  # 0 until price first breaks a band: no default side
    for i in range(1, len(df)):
        if np.isnan(upper[i]) or np.isnan(fu[i - 1]):
            continue
        fu[i] = upper[i] if upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1] else fu[i - 1]
        fl[i] = lower[i] if lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1] else fl[i - 1]
        if close[i] > fu[i - 1]:
            direction[i] = 1
        elif close[i] < fl[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return pd.Series(direction, index=df.index)


def swings(df: pd.DataFrame, left: int = 3, right: int = 3):
    """Fractal swing highs/lows. A swing at bar i is only known at bar i+right,
    so callers must not use the last `right` bars as confirmed swings."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    highs, lows = [], []
    for i in range(left, len(df) - right):
        if h[i] == max(h[i - left:i + right + 1]) and h[i] > max(h[i - left:i]):
            highs.append((i, h[i]))
        if l[i] == min(l[i - left:i + right + 1]) and l[i] < min(l[i - left:i]):
            lows.append((i, l[i]))
    return highs, lows


def levels(df: pd.DataFrame, atr_now: float, max_levels: int = 8) -> list[dict]:
    """Support/resistance zones: swing points clustered within 0.5 ATR,
    ranked by number of touches and recency."""
    highs, lows = swings(df)
    pts = [(i, p) for i, p in highs + lows]
    if not pts or not atr_now or np.isnan(atr_now):
        return []
    pts.sort(key=lambda x: x[1])
    clusters: list[list[tuple[int, float]]] = []
    for i, p in pts:
        if clusters and p - clusters[-1][-1][1] <= 0.5 * atr_now:
            clusters[-1].append((i, p))
        else:
            clusters.append([(i, p)])
    n = len(df)
    out = []
    for c in clusters:
        price = float(np.mean([p for _, p in c]))
        last = max(i for i, _ in c)
        out.append({"price": price, "touches": len(c), "last_index": last,
                    "strength": len(c) + 2 * (last / n)})
    out.sort(key=lambda x: -x["strength"])
    return out[:max_levels]


def pivots(prev_high: float, prev_low: float, prev_close: float) -> dict:
    p = (prev_high + prev_low + prev_close) / 3
    return {"P": p, "R1": 2 * p - prev_low, "S1": 2 * p - prev_high,
            "R2": p + (prev_high - prev_low), "S2": p - (prev_high - prev_low)}


def candle_patterns(df: pd.DataFrame) -> dict:
    """Patterns on the last closed bar: +1 bullish, -1 bearish, 0 none."""
    if len(df) < 3:
        return {}
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    i = len(df) - 1
    body = abs(c[i] - o[i])
    rng = max(h[i] - l[i], 1e-12)
    upper = h[i] - max(c[i], o[i])
    lower = min(c[i], o[i]) - l[i]
    out = {}
    if c[i] > o[i] and c[i - 1] < o[i - 1] and c[i] >= o[i - 1] and o[i] <= c[i - 1]:
        out["engulfing"] = 1
    elif c[i] < o[i] and c[i - 1] > o[i - 1] and c[i] <= o[i - 1] and o[i] >= c[i - 1]:
        out["engulfing"] = -1
    if lower >= 2 * body and upper <= 0.35 * rng:
        out["pin_bar"] = 1
    elif upper >= 2 * body and lower <= 0.35 * rng:
        out["pin_bar"] = -1
    if h[i] < h[i - 1] and l[i] > l[i - 1]:
        out["inside_bar"] = 0
    if body / rng > 0.8:
        out["marubozu"] = 1 if c[i] > o[i] else -1
    return out


def rsi_divergence(df: pd.DataFrame, r: pd.Series, lookback: int = 40) -> int:
    """+1 bullish divergence (lower low in price, higher low in RSI),
    -1 bearish divergence, 0 none. Uses confirmed swings only."""
    sub = df.iloc[-lookback:]
    rs = r.iloc[-lookback:].to_numpy()
    highs, lows = swings(sub, 3, 2)
    bull = bear = False
    if len(lows) >= 2:
        (i1, p1), (i2, p2) = lows[-2], lows[-1]
        bull = p2 < p1 and rs[i2] > rs[i1] and len(sub) - i2 <= 8
    if len(highs) >= 2:
        (i1, p1), (i2, p2) = highs[-2], highs[-1]
        bear = p2 > p1 and rs[i2] < rs[i1] and len(sub) - i2 <= 8
    return int(bull) - int(bear)  # both at once cancel out instead of favouring one side


def percentile_rank(s: pd.Series, window: int = 200) -> float:
    w = s.dropna().iloc[-window:]
    if len(w) < 10:
        return 50.0
    return float((w < w.iloc[-1]).mean() * 100)
