"""The bot grading itself.

After every closed trade the numbers are updated; at most every 30 minutes the bot re-learns:
  - Setups that lose: any (mode, market) with 20+ trades and negative expectancy (shrunk toward
    zero so a short unlucky run is not enough) is paused automatically.
  - Take-profit size for scalps: if many trades reached 60% of their target and then reversed,
    targets are lowered 10% (down to 60% of normal); if most targets are hit quickly, they are
    raised 5% (up to 150%). This is exactly the 'was in profit, held too long, reversed' problem.
  - A daily self-review summary is posted to the chat (and as a notification) after the day ends.
"""
from __future__ import annotations

import time
from collections import defaultdict

from .. import db

MIN_TRADES_TO_PAUSE = 20
SHRINK = 10


def learning() -> dict:
    return db.kv_get("bot_learning") or {"scalp_tp_factor": 1.0, "paused": {}}


def stats(account: str | None = None, since: float | None = None) -> dict:
    sql = "SELECT * FROM bot_trades WHERE status='closed'"
    params: list = []
    if account:
        sql += " AND account=?"
        params.append(account)
    if since:
        sql += " AND closed_at>=?"
        params.append(since)
    rows = db.q(sql + " ORDER BY closed_at", params)

    def summ(rs):
        n = len(rs)
        if not n:
            return {"trades": 0}
        wins = [r["pnl"] for r in rs if r["pnl"] > 0]
        losses = [r["pnl"] for r in rs if r["pnl"] <= 0]
        gl = -sum(losses)
        giveback = sum(1 for r in rs if (r["max_profit"] or 0) >= 0.6 * r["tp_usd"] and r["pnl"] <= 0)
        return {"trades": n, "win_rate": len(wins) / n, "net": round(sum(r["pnl"] for r in rs), 2),
                "avg_win": round(sum(wins) / len(wins), 3) if wins else None,
                "avg_loss": round(sum(losses) / len(losses), 3) if losses else None,
                "expectancy": round(sum(r["pnl"] for r in rs) / n, 4),
                "profit_factor": round(sum(wins) / gl, 2) if gl > 0 else None,
                "gave_back_after_profit": giveback,
                "avg_minutes": round(sum((r["closed_at"] - r["opened_at"]) for r in rs) / n / 60, 1)}

    by = defaultdict(list)
    for r in rows:
        by[f"{r['mode']}|{r['symbol']}"].append(r)
    reasons = defaultdict(int)
    for r in rows:
        reasons[r["close_reason"] or "?"] += 1
    return {"overall": summ(rows), "scalp": summ([r for r in rows if r["mode"] == "scalp"]),
            "short": summ([r for r in rows if r["mode"] == "short"]),
            "by_setup": {k: summ(v) for k, v in sorted(by.items())}, "exit_reasons": dict(reasons)}


def relearn(account: str | None = None) -> dict:
    L = learning()
    rows = db.q("SELECT * FROM bot_trades WHERE status='closed'" + (" AND account=?" if account else ""),
                [account] if account else [])
    # 1. pause losing setups
    by = defaultdict(list)
    for r in rows:
        by[f"{r['mode']}|{r['symbol']}"].append(r["pnl"])
    paused = {}
    for k, pnls in by.items():
        n = len(pnls)
        shrunk = sum(pnls) / (n + SHRINK)
        if n >= MIN_TRADES_TO_PAUSE and shrunk < 0:
            paused[k] = {"trades": n, "net": round(sum(pnls), 2), "since": time.time()}
    L["paused"] = paused
    # 2. scalp take-profit size from the last 40 scalps
    sc = [r for r in rows if r["mode"] == "scalp"][-40:]
    if len(sc) >= 15:
        reversed_ = sum(1 for r in sc if (r["max_profit"] or 0) >= 0.6 * r["tp_usd"] and r["close_reason"] != "take profit")
        quick = sum(1 for r in sc if r["close_reason"] == "take profit" and (r["closed_at"] - r["opened_at"]) < 180)
        f = L.get("scalp_tp_factor", 1.0)
        if reversed_ / len(sc) > 0.35:
            f = max(0.6, f * 0.9)
        elif quick / len(sc) > 0.6:
            f = min(1.5, f * 1.05)
        L["scalp_tp_factor"] = round(f, 3)
        L["last_scalp_review"] = {"scalps": len(sc), "reversed_after_60pct": reversed_, "quick_targets": quick}
    L["updated_at"] = time.time()
    db.kv_set("bot_learning", L)
    return L


def maybe_learn() -> None:
    L = learning()
    if time.time() - (L.get("updated_at") or 0) > 1800:
        relearn()
