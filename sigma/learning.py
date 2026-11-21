"""Self-learning layer.

1. Outcome model - a regularised logistic regression that predicts P(win)
   for a trade from the component scores (aligned with the trade direction),
   with separate effects in trending and ranging regimes. It starts from
   prior weights (classic technical-analysis beliefs) and is refitted on
   every resolved paper trade, so components that keep failing lose weight
   and ones that keep working gain it.
2. Setup statistics - win rate and expectancy per (symbol, timeframe,
   regime, direction) with Bayesian shrinkage; setups that keep losing
   are switched off automatically.
3. Calibration - predicted vs realised win rate, so the shown confidence
   is honest.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np

from . import config, db

BASE = ["trend", "htf_trend", "momentum", "structure", "mean_reversion",
        "levels", "pattern", "divergence", "fundamental", "news"]
FEATURES = BASE + [f"{b}@trending" for b in BASE] + [f"{b}@ranging" for b in BASE]

PRIOR_BASE = {"trend": 0.6, "htf_trend": 0.6, "momentum": 0.4, "structure": 0.5,
              "mean_reversion": 0.3, "levels": 0.4, "pattern": 0.2, "divergence": 0.2,
              "fundamental": 0.4, "news": 0.25}
PRIOR_REGIME = {"trend@trending": 0.3, "htf_trend@trending": 0.2, "mean_reversion@trending": -0.3,
                "mean_reversion@ranging": 0.4, "levels@ranging": 0.3, "trend@ranging": -0.3,
                "divergence@ranging": 0.2}
PRIOR_STRENGTH = 10.0      # the prior is worth about this many trades
LIVE_WEIGHT = 3.0          # a live paper trade counts this much more than a backtest one
BACKTEST_WEIGHT = 0.3      # backtest trades overlap heavily, so each counts for less


def breakeven_prob(reward_r: float = None) -> float:
    r = reward_r or config.REWARD_R
    return 1.0 / (1.0 + r)


def prior_theta() -> np.ndarray:
    th = np.zeros(len(FEATURES))
    for i, f in enumerate(FEATURES):
        th[i] = PRIOR_BASE.get(f, PRIOR_REGIME.get(f, 0.0))
    return th


def prior_bias() -> float:
    p = breakeven_prob()
    return math.log(p / (1 - p))


def featurize(components: dict, regime: str, direction: int) -> np.ndarray:
    s = np.array([float(components.get(b, 0.0) or 0.0) for b in BASE]) * direction
    t = s if regime == "trending" else np.zeros_like(s)
    r = s if regime == "ranging" else np.zeros_like(s)
    return np.concatenate([s, t, r])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Model:
    def __init__(self, theta=None, bias=None, meta=None):
        self.theta = np.array(theta) if theta is not None else prior_theta()
        self.bias = prior_bias() if bias is None else float(bias)
        self.meta = meta or {"n_live": 0, "n_backtest": 0, "fitted_at": None,
                             "avg_win_r": config.REWARD_R * 0.95, "avg_loss_r": -1.0,
                             "calib": calib_prior()}
        self.meta.setdefault("calib", calib_prior())

    # ---------------------------------------------------------------- use --
    def prob(self, components: dict, regime: str, direction: int) -> float:
        return float(sigmoid(self.bias + self.theta @ featurize(components, regime, direction)))

    def calibrate(self, p: float) -> float:
        """Platt scaling learned from the model's own out-of-sample record:
        if past 70% calls only won 50% of the time, today's 70% becomes ~50%."""
        a, c = self.meta.get("calib") or calib_prior()
        p = min(max(p, 1e-6), 1 - 1e-6)
        return float(sigmoid(a + c * math.log(p / (1 - p))))

    def expectancy(self, p: float) -> float:
        return p * self.meta["avg_win_r"] + (1 - p) * self.meta["avg_loss_r"]

    def weights(self) -> dict:
        return {f: round(float(w), 4) for f, w in zip(FEATURES, self.theta)}

    # ---------------------------------------------------------------- fit --
    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, iters: int = 400, lr: float = 0.5) -> None:
        if len(y) == 0:
            return
        th0, b0 = prior_theta(), prior_bias()
        th, b = self.theta.copy(), self.bias
        W = w.sum()
        lam = PRIOR_STRENGTH / max(W, 1.0)
        for _ in range(iters):
            p = sigmoid(b + X @ th)
            g = (p - y) * w
            grad_th = X.T @ g / W + 2 * lam * (th - th0)
            grad_b = g.sum() / W + 2 * lam * (b - b0) * 0.1
            th -= lr * grad_th
            b -= lr * grad_b
        self.theta, self.bias = th, b

    def to_json(self) -> dict:
        return {"theta": self.theta.tolist(), "bias": self.bias, "meta": self.meta, "features": FEATURES}

    @classmethod
    def from_json(cls, js: dict | None) -> "Model":
        if not js or js.get("features") != FEATURES:
            return cls()
        return cls(js["theta"], js["bias"], js["meta"])


_model: Model | None = None


def model() -> Model:
    global _model
    if _model is None:
        _model = Model.from_json(db.kv_get("model"))
    return _model


def reset_model() -> None:
    global _model
    _model = Model()
    db.kv_set("model", _model.to_json())


# --------------------------------------------------------------- dataset ---
def training_rows(before: float | None = None) -> list[dict]:
    sql = ("SELECT t.direction, t.r_multiple, t.source, t.closed_at, a.components, a.regime, a.raw_probability "
           "FROM trades t JOIN analyses a ON a.id = t.analysis_id "
           "WHERE t.status != 'open' AND t.r_multiple IS NOT NULL AND t.origin = 'robot'")
    params: list = []
    if before is not None:
        sql += " AND t.closed_at < ?"
        params.append(before)
    return [db.decode(r) for r in db.q(sql, params)]


def refit(before: float | None = None, source_filter: str | None = None,
          extra_rows: list[dict] | None = None, use_db: bool = True) -> Model:
    rows = training_rows(before) if use_db else []
    rows += extra_rows or []
    if source_filter:
        rows = [r for r in rows if r["source"] == source_filter]
    m = Model()
    if rows:
        X = np.array([featurize(r["components"], r["regime"], r["direction"]) for r in rows])
        y = np.array([1.0 if r["r_multiple"] > 0 else 0.0 for r in rows])
        w = np.array([LIVE_WEIGHT if r["source"] == "live" else BACKTEST_WEIGHT for r in rows])
        m.fit(X, y, w)
        wins = [r["r_multiple"] for r in rows if r["r_multiple"] > 0]
        losses = [r["r_multiple"] for r in rows if r["r_multiple"] <= 0]
        k = 20  # shrink average win/loss toward defaults until there is enough data
        m.meta["avg_win_r"] = (sum(wins) + k * config.REWARD_R * 0.95) / (len(wins) + k)
        m.meta["avg_loss_r"] = (sum(losses) + k * -1.0) / (len(losses) + k)
        m.meta["calib"] = fit_calibration(
            [r.get("raw_probability") for r in rows], y, w)
        m.meta["n_live"] = sum(1 for r in rows if r["source"] == "live")
        m.meta["n_backtest"] = sum(1 for r in rows if r["source"] == "backtest")
    m.meta["fitted_at"] = time.time()
    return m


CALIB_PRIOR = 30.0   # calibration needs evidence to move away from its prior
CALIB_SLOPE0 = 0.6   # start sceptical: assume only 60% of the model's claimed edge is real


def calib_prior() -> list[float]:
    """Shrinks the model's log-odds toward break-even (not toward 50%)."""
    return [prior_bias() * (1 - CALIB_SLOPE0), CALIB_SLOPE0]


def fit_calibration(raw_probs, y, w, iters: int = 500, lr: float = 0.5) -> list[float]:
    idx = [i for i, p in enumerate(raw_probs) if p is not None]
    a0, c0 = calib_prior()
    if len(idx) < 30:
        return [a0, c0]
    p = np.clip(np.array([raw_probs[i] for i in idx], dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    yy, ww = np.asarray(y)[idx], np.asarray(w)[idx]
    W = ww.sum()
    lam = CALIB_PRIOR / max(W, 1.0)
    a, c = a0, c0
    for _ in range(iters):
        q = sigmoid(a + c * z)
        g = (q - yy) * ww
        ga = g.sum() / W + 2 * lam * (a - a0)
        gc = (g * z).sum() / W + 2 * lam * (c - c0)
        a -= lr * ga
        c -= lr * gc
    c = max(c, 0.0)  # never invert the model's ranking
    return [float(a), float(c)]


def refit_and_save() -> Model:
    global _model
    _model = refit()
    db.kv_set("model", _model.to_json())
    history = db.kv_get("weight_history", [])
    history.append({"t": time.time(), "n": _model.meta["n_live"] + _model.meta["n_backtest"],
                    "weights": {b: _model.weights()[b] for b in BASE}})
    db.kv_set("weight_history", history[-200:])
    return _model


# ----------------------------------------------------------- setup stats ---
def shrunk_expectancy(sum_r: float, n: int, k: float = 10.0, prior: float = 0.0) -> float:
    return (sum_r + k * prior) / (n + k)


def setup_stats(signals_only: bool = True) -> dict[str, dict]:
    sql = ("SELECT t.setup_key, t.r_multiple, t.source FROM trades t "
           "JOIN analyses a ON a.id = t.analysis_id WHERE t.status != 'open' AND t.r_multiple IS NOT NULL AND t.origin = 'robot'")
    if signals_only:
        sql += " AND a.is_signal = 1"
    agg: dict[str, dict] = defaultdict(lambda: {"n": 0, "wins": 0, "sum_r": 0.0})
    for r in db.q(sql):
        a = agg[r["setup_key"]]
        a["n"] += 1
        a["wins"] += r["r_multiple"] > 0
        a["sum_r"] += r["r_multiple"]
    out = {}
    for k, a in agg.items():
        n = a["n"]
        out[k] = {**a, "win_rate": a["wins"] / n, "expectancy": a["sum_r"] / n,
                  "shrunk_expectancy": shrunk_expectancy(a["sum_r"], n),
                  # Beta(1,1) posterior mean for win rate
                  "win_rate_post": (a["wins"] + 1) / (n + 2)}
    return out


def gated(setup_key: str, stats: dict | None = None) -> dict | None:
    s = (stats if stats is not None else setup_stats()).get(setup_key)
    if s and s["n"] >= config.GATE_MIN_TRADES and s["shrunk_expectancy"] < config.GATE_MAX_EXPECTANCY:
        return s
    return None


def calibration(source: str | None = None, bins: int = 10) -> list[dict]:
    sql = ("SELECT a.probability p, t.r_multiple r FROM trades t JOIN analyses a ON a.id=t.analysis_id "
           "WHERE t.status!='open' AND t.r_multiple IS NOT NULL AND t.origin='robot'")
    params = []
    if source:
        sql += " AND t.source=?"
        params.append(source)
    rows = db.q(sql, params)
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = [r for r in rows if lo <= (r["p"] or 0) < hi or (i == bins - 1 and r["p"] == 1)]
        if sel:
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(sel),
                        "predicted": sum(r["p"] for r in sel) / len(sel),
                        "actual": sum(1 for r in sel if r["r"] > 0) / len(sel)})
    return out


def performance(source: str | None = "live", signals_only: bool = True, symbol: str | None = None,
                timeframe: str | None = None, origin: str = "robot") -> dict:
    sql = ("SELECT t.*, a.probability FROM trades t JOIN analyses a ON a.id=t.analysis_id "
           "WHERE t.status!='open' AND t.r_multiple IS NOT NULL AND t.origin=?")
    params: list = [origin]
    if origin == "user":
        signals_only = False
    for col, val in (("t.source", source), ("t.symbol", symbol), ("t.timeframe", timeframe)):
        if val:
            sql += f" AND {col}=?"
            params.append(val)
    if signals_only:
        sql += " AND a.is_signal=1"
    rows = db.q(sql + " ORDER BY t.closed_at", params)
    return summarize([r["r_multiple"] for r in rows]) | {
        "open": db.q1("SELECT COUNT(*) n FROM trades WHERE status='open' AND origin=?" +
                      (" AND source=?" if source else ""), [origin] + ([source] if source else []))["n"]}


def summarize(rs: list[float]) -> dict:
    n = len(rs)
    if n == 0:
        return {"trades": 0, "win_rate": None, "expectancy_r": None, "total_r": 0.0,
                "profit_factor": None, "max_drawdown_r": 0.0}
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    eq, peak, dd = 0.0, 0.0, 0.0
    for r in rs:
        eq += r
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    gl = -sum(losses)
    return {"trades": n, "win_rate": len(wins) / n, "expectancy_r": sum(rs) / n, "total_r": sum(rs),
            "profit_factor": (sum(wins) / gl) if gl > 0 else None, "max_drawdown_r": dd,
            "avg_win_r": (sum(wins) / len(wins)) if wins else None,
            "avg_loss_r": (sum(losses) / len(losses)) if losses else None}


def insights(limit: int = 8) -> list[str]:
    """Plain-language findings the chat can quote."""
    m = model()
    out = []
    w = m.weights()
    p = dict(zip(FEATURES, prior_theta()))
    moved = sorted(BASE, key=lambda b: -abs(w[b] - p[b]))
    for b in moved[:3]:
        if abs(w[b] - p[b]) > 0.05:
            out.append(f"'{b}' weight moved {p[b]:+.2f} -> {w[b]:+.2f} from results "
                       f"({'more' if w[b] > p[b] else 'less'} predictive than assumed).")
    stats = setup_stats()
    for k, s in sorted(stats.items(), key=lambda kv: kv[1]["shrunk_expectancy"]):
        if s["n"] >= 8:
            out.append(f"Setup {k}: {s['n']} signals, win {s['win_rate']:.0%}, "
                       f"expectancy {s['expectancy']:+.2f}R"
                       + (" - SWITCHED OFF" if gated(k, stats) else ""))
        if len(out) >= limit:
            break
    return out
