"""Demo account: paper money that the robot trades with.

Which trades it takes
  - every robot SIGNAL found by the scanner, and
  - every instrument YOU ask it to analyse (chat or the Analyze button), signal or not.
Shadow trades (scanner analyses that were not signals) are still paper-traded for learning,
but do not touch the demo balance.

Sizing: each trade risks `risk_pct` of the balance at the moment it is placed, so
profit/loss in money = R multiple x risk amount (a 2R win at 1% risk = +2% of balance).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from . import config, db, learning

DEFAULTS = {"start_balance": 10_000.0, "currency": "USD", "risk_pct": 1.0}


def settings() -> dict:
    s = db.kv_get("demo_account") or {}
    out = {**DEFAULTS, **s}
    out.setdefault("started_at", time.time())
    if "started_at" not in s:
        db.kv_set("demo_account", out)
    return out


def update_settings(start_balance: float | None = None, currency: str | None = None,
                    risk_pct: float | None = None, reset: bool = False) -> dict:
    s = settings()
    if start_balance is not None:
        if start_balance <= 0:
            raise ValueError("starting balance must be positive")
        s["start_balance"] = float(start_balance)
    if currency:
        s["currency"] = currency.upper()[:3]
    if risk_pct is not None:
        if not 0.05 <= risk_pct <= 5:
            raise ValueError("risk per trade must be between 0.05% and 5%")
        s["risk_pct"] = float(risk_pct)
    if reset:   # only trades placed after this point count (by trade number, not by clock time)
        s["started_at"] = time.time()
        last = db.q1("SELECT COALESCE(MAX(id), 0) m FROM trades")
        s["start_after_id"] = int(last["m"] or 0)
    db.kv_set("demo_account", s)
    return s


def _closed_pnl(after_id: int) -> float:
    r = db.q1("SELECT COALESCE(SUM(r_multiple * risk_amount), 0) v FROM trades "
              "WHERE demo=1 AND id > ? AND r_multiple IS NOT NULL AND risk_amount IS NOT NULL", (after_id,))
    return float(r["v"] or 0)


def balance() -> float:
    s = settings()
    return s["start_balance"] + _closed_pnl(int(s.get("start_after_id", 0)))


def size_trade(symbol_quote: str, risk_distance: float) -> dict:
    """Money at risk for a new trade and the position size in units (when a rate is available)."""
    s = settings()
    risk_amount = balance() * s["risk_pct"] / 100
    units = None
    try:
        from .agent import fx_rate
        conv = fx_rate(symbol_quote or "USD", s["currency"])
        units = risk_amount / (risk_distance * conv) if risk_distance > 0 else None
    except Exception:
        pass
    return {"risk_amount": round(risk_amount, 2), "units": units}


def _live_r(t: dict, prices: dict) -> float | None:
    px = prices.get(t["symbol"])
    risk = abs(t["entry"] - t["sl"])
    if px is None or risk <= 0:
        return None
    return t["direction"] * (px - t["entry"]) / risk


def summary(with_prices: bool = True) -> dict:
    s = settings()
    since = s["started_at"]
    rows = db.q("SELECT t.*, a.is_signal, a.source analysis_source, a.probability FROM trades t "
                "JOIN analyses a ON a.id=t.analysis_id WHERE t.demo=1 AND t.id > ? "
                "ORDER BY t.opened_at DESC", (int(s.get("start_after_id", 0)),))
    open_ = [r for r in rows if r["status"] == "open"]
    pending = [r for r in rows if r["status"] == "pending"]
    closed = [r for r in rows if r["r_multiple"] is not None and r["status"] not in ("open", "pending")]
    prices: dict = {}
    if with_prices and open_:
        from .data import market
        for sym in {r["symbol"] for r in open_}:
            try:
                prices[sym] = market.last_price(sym)
            except Exception:
                pass
    unreal = 0.0
    for r in open_:
        lr = _live_r(r, prices)
        r["current_price"] = prices.get(r["symbol"])
        r["unrealized_r"] = lr
        r["unrealized_pnl"] = None if lr is None or r["risk_amount"] is None else lr * r["risk_amount"]
        unreal += r["unrealized_pnl"] or 0.0
    for r in closed:
        r["pnl"] = (r["r_multiple"] or 0) * (r["risk_amount"] or 0)
    bal = s["start_balance"] + sum(r["pnl"] for r in closed)

    tz = ZoneInfo(config.DISPLAY_TZ)
    now = datetime.now(tz)
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    starts = {"today": day0.timestamp(), "this_week": (day0 - timedelta(days=day0.weekday())).timestamp(),
              "this_month": day0.replace(day=1).timestamp()}
    periods = {k: round(sum(r["pnl"] for r in closed if (r["closed_at"] or 0) >= v), 2) for k, v in starts.items()}

    eq, curve = s["start_balance"], [{"t": since, "balance": s["start_balance"]}]
    for r in sorted(closed, key=lambda r: r["closed_at"] or 0):
        eq += r["pnl"]
        curve.append({"t": r["closed_at"], "balance": round(eq, 2)})
    stats = learning.summarize([r["r_multiple"] for r in sorted(closed, key=lambda r: r["closed_at"] or 0)])
    return {
        "settings": s, "currency": s["currency"], "start_balance": s["start_balance"],
        "balance": round(bal, 2), "equity": round(bal + unreal, 2), "unrealized_pnl": round(unreal, 2),
        "total_pnl": round(bal - s["start_balance"], 2),
        "return_pct": round(100 * (bal - s["start_balance"]) / s["start_balance"], 2),
        "pnl": periods, "stats": stats, "open": open_, "pending": pending, "closed": closed[:200],
        "equity_curve": curve,
    }
