"""Alerts for the trades the robot takes on its own (background scans) on the demo account, so you can
take them too. Each alert carries the lot size for YOUR real account.

  Opened      - a market trade was taken, or a limit/stop order was placed (with the price now)
  Filled      - a pending order was filled
  Breakeven   - the trade is +1R: the paper trade moves its stop to the entry; do the same
  Closed      - the result (won / lost / breakeven / expired), or an order that was cancelled or missed
  Deep check  - (from alerts.py) the slower deep analysis confirms or disagrees, shortly after opening

Trades from analyses you asked for in the chat are not announced (you already have them).
Lot size: your real balance x risk % / (stop distance x contract size), converted to your account
currency. Real balance, currency and risk come from these settings, or from your goal in the chat.
"""
from __future__ import annotations

import logging
import threading
import time

from . import config, db
from .symbols import resolve

log = logging.getLogger("sigma.trade_alerts")

DEFAULTS = {"enabled": True, "opened": True, "filled": True, "breakeven": True, "closed": True,
            "real_balance": None, "currency": None, "risk_pct": None}
FALLBACK_RISK = 1.0


def settings() -> dict:
    return {**DEFAULTS, **(db.kv_get("trade_alerts") or {})}


def update_settings(**kw) -> dict:
    s = settings()
    for k, v in kw.items():
        if k not in DEFAULTS:
            continue
        if k == "real_balance" and v is not None:
            v = float(v)
            if v <= 0:
                raise ValueError("real balance must be positive")
        if k == "risk_pct" and v is not None:
            v = float(v)
            if not 0.1 <= v <= 5:
                raise ValueError("risk must be between 0.1% and 5%")
        if k == "currency" and v:
            v = str(v).upper()[:3]
        s[k] = v
    db.kv_set("trade_alerts", {k: s[k] for k in DEFAULTS})
    return s


def real_account() -> dict:
    """Balance, currency and risk % used for lot sizes (settings first, then your chat goal)."""
    s = settings()
    g = {}
    try:
        from . import goals
        g = goals.get() or {}
    except Exception:
        pass
    return {"balance": s["real_balance"] or g.get("balance"), "currency": s["currency"] or g.get("currency") or "USD",
            "risk_pct": s["risk_pct"] or g.get("risk_pct") or FALLBACK_RISK}


def lot_size(symbol: str, entry: float, sl: float) -> dict | None:
    acc = real_account()
    if not acc["balance"]:
        return None
    try:
        from .agent import position_size
        r = position_size(symbol, acc["balance"], acc["risk_pct"], entry, sl, acc["currency"])
    except Exception as e:
        log.debug("lot size: %s", e)
        return None
    if r.get("error") or r.get("lots") is None:
        return None
    exact = r["units"] / r["contract_size_assumed"]
    lots = int(exact * 100 + 1e-9) / 100                   # round DOWN to 0.01 so risk never exceeds your %
    out = {"lots": lots, "risk_amount": r["risk_amount"], "currency": acc["currency"], "risk_pct": acc["risk_pct"]}
    if lots < 0.01:
        out["lots"] = 0.01
        out["too_small"] = True
        out["actual_risk"] = round(r["risk_amount"] * 0.01 / max(exact, 1e-12), 2)
    return out


def _fmt(x) -> str:
    return "-" if x is None else (f"{x:,.5f}" if abs(x) < 10 else f"{x:,.3f}" if abs(x) < 1000 else f"{x:,.2f}")


def _lots_text(l: dict | None) -> str:
    if not l:
        return "Tell me your real balance (Paper trading → Trade alerts) for a lot size."
    if l.get("too_small"):
        return (f"Your lot size: 0.01 (smallest) - that risks about {l['actual_risk']:,.2f} {l['currency']}, "
                f"more than your {l['risk_pct']:g}%.")
    return f"Your lot size: {l['lots']:.2f} (risks {l['risk_amount']:,.2f} {l['currency']} = {l['risk_pct']:g}%)."


def _announce(t: dict, title: str, body: str, kind: str, chat: bool = False) -> None:
    now = time.time()
    try:
        db.insert("alerts", {"created_at": now, "analysis_id": t.get("analysis_id"), "symbol": t["symbol"],
                             "timeframe": t["timeframe"], "order_type": t.get("order_type"), "title": title,
                             "body": body, "kind": f"trade_{kind}"})
        if chat:
            db.insert("chat", {"session": "default", "created_at": now, "role": "assistant", "content": body})
        from . import push
        push.notify(title, push.plain(body)[:230], url="/#paper", tag=f"trade-{t['id']}")
    except Exception as e:
        log.warning("trade alert failed: %s", e)


def _side(t: dict) -> str:
    return t.get("order_type") or ("BUY" if t["direction"] > 0 else "SELL")


def _name(t: dict) -> str:
    try:
        return resolve(t["symbol"]).display
    except Exception:
        return t["symbol"]


def _wanted(t: dict, kind: str) -> bool:
    s = settings()
    if not (s["enabled"] and s.get(kind) and t.get("demo")):
        return False
    a = db.q1("SELECT source FROM analyses WHERE id=?", (t.get("analysis_id"),))
    return not (a and a.get("source") == "chat") and t.get("origin") != "user"


# ----------------------------------------------------------------------------------- events ---
def opened(trade_id: int, res: dict | None = None) -> None:
    t = db.q1("SELECT * FROM trades WHERE id=?", (trade_id,))
    if not t or not _wanted(t, "opened"):
        return
    pending = t["status"] == "pending"
    l = lot_size(t["symbol"], t["entry"], t["sl"])
    p = (res or {}).get("probability")
    tps = (res or {}).get("take_profits") or []
    tp_txt = " · ".join(f"{x['name']} {_fmt(x['price'])} ({x['p_reach']:.0%})" for x in tps[:4]) or f"TP {_fmt(t['tp'])}"
    now_px = (res or {}).get("price")
    head = (f"⏳ Order placed: {_side(t)} {_name(t)} ({t['timeframe']})" if pending
            else f"📈 Trade taken: {_side(t)} {_name(t)} ({t['timeframe']})")
    body = (f"**{head}**\n\n"
            f"Entry {_fmt(t['entry'])}" + (f" (price now {_fmt(now_px)})" if pending and now_px else "") +
            f" · Stop loss {_fmt(t['sl'])} · {tp_txt}" + (f" · P(win) {p:.0%}" if p else "") + "\n\n"
            f"{_lots_text(l)}\n\n"
            + ("_The demo account fills this order only if price reaches the entry; you'll get an alert when it does._"
               if pending else "_Taken on the demo account. Sigma moves its stop to entry at +1R and will alert you._")
            + " A deep check is running - you'll hear if it disagrees.")
    _announce(t, head, body, "opened", chat=True)


def filled(t: dict) -> None:
    if not _wanted(t, "filled"):
        return
    t = db.q1("SELECT * FROM trades WHERE id=?", (t["id"],)) or t
    l = lot_size(t["symbol"], t["entry"], t["sl"])
    head = f"✅ Filled: {_side(t)} {_name(t)} ({t['timeframe']})"
    _announce(t, head, f"**{head}** at {_fmt(t['entry'])} · SL {_fmt(t['sl'])} · TP {_fmt(t['tp'])}\n\n{_lots_text(l)}",
              "filled")


def breakeven(t: dict) -> None:
    if not _wanted(t, "breakeven"):
        return
    head = f"🛡 {_name(t)} is +{config.BREAKEVEN_R:g}R: move your stop to entry"
    _announce(t, head, f"**{head}** ({_fmt(t['entry'])}). {_side(t)} {t['timeframe']} - the demo trade now has its "
                       f"stop at the entry, so it can no longer lose (apart from costs).", "breakeven")


def closed(t: dict) -> None:
    if not _wanted(t, "closed"):
        return
    st, r = t["status"], t.get("r_multiple")
    words = {"won": "🎯 Take profit hit", "lost": "❌ Stop loss hit", "breakeven": "⚖️ Closed at breakeven",
             "expired": "⌛ Closed at its time limit", "missed": "↗️ Order missed", "cancelled": "🚫 Order cancelled"}
    head = f"{words.get(st, st)}: {_side(t)} {_name(t)} ({t['timeframe']})"
    detail = (f" {r:+.2f}R" if r is not None and st not in ("cancelled", "missed") else "")
    extra = (f" Price went the right way by {t.get('missed_r', 0):.1f}R without reaching the entry."
             if st == "missed" else " The entry was never reached." if st == "cancelled" else "")
    _announce(t, head, f"**{head}**{detail}.{extra}", "closed")


def deep_disagrees(scan_result: dict) -> None:
    """The background deep check didn't confirm a trade that was already announced."""
    t = db.q1("SELECT * FROM trades WHERE analysis_id=? AND demo=1", (scan_result.get("analysis_id"),))
    if not t or not _wanted(t, "opened"):
        return
    head = f"⚠️ Deep check doesn't confirm {_side(t)} {_name(t)} ({t['timeframe']})"
    _announce(t, head, f"**{head}** - the slower, fuller analysis came out differently. The demo account keeps the "
                       f"trade so it can learn from it, but consider skipping it.", "deep")


# --------------------------------------------------------------------- breakeven watcher ---
def check_breakeven(open_trades: list[dict], bars_by_trade: dict[int, "object"]) -> None:
    """Called by the paper monitor with each open trade's bars since it opened."""
    if not config.BREAKEVEN_R:
        return
    done = set(db.kv_get("be_alerted") or [])
    new = []
    for t in open_trades:
        sub = bars_by_trade.get(t["id"])
        if t["id"] in done or sub is None or not len(sub):
            continue
        risk = abs(t["entry"] - t["sl"])
        if risk <= 0:
            continue
        best = (float(sub["high"].max()) - t["entry"]) if t["direction"] > 0 else (t["entry"] - float(sub["low"].min()))
        if best / risk >= config.BREAKEVEN_R:
            new.append(t["id"])
            breakeven(t)
    if new:
        db.kv_set("be_alerted", sorted(done | set(new))[-500:])


def run(fn, *a) -> None:
    """Alerts never slow down trading or scanning."""
    threading.Thread(target=lambda: _safe(fn, *a), daemon=True).start()


def _safe(fn, *a):
    try:
        fn(*a)
    except Exception as e:
        log.warning("trade alert %s: %s", getattr(fn, "__name__", fn), e)
