"""PDF report of the demo account's closed paper trades, for reviewing and improving the robot.

Contents:
  1. Summary - trades, win rate, net P&L, average R, profit factor, expectancy.
  2. Where it wins and loses - the same numbers split by market, timeframe, order type, direction,
     market regime, setup, asset class and the probability Sigma gave the trade.
  3. Warning signs - trades that were +1R or better and still lost, and which factors pointed the
     wrong way most often in losing trades.
  4. Missed entries - limit/stop orders that never filled while price went the predicted way.
  5. Every trade - levels, result, probability, regime, setup, how far price went for/against it,
     the factor scores, Sigma's analysis notes at the time, and the lesson written after it closed.
"""
from __future__ import annotations

import io
import json
import time
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from . import account, config, db
from .symbols import resolve

ASCII = {"→": "->", "←": "<-", "≥": ">=", "≤": "<=", "−": "-", "–": "-", "—": "-", "…": "...", "×": "x",
         "’": "'", "‘": "'", "“": '"', "”": '"', "•": "-", "≈": "~", "±": "+/-", "°": " deg", "·": "-",
         "↑": "up", "↓": "down", "✓": "ok", "✗": "x", "⚠": "!", " ": " "}


def clean(s) -> str:
    """Built-in PDF fonts only have Latin characters: swap symbols for plain text, escape markup."""
    s = "" if s is None else str(s)
    for k, v in ASCII.items():
        s = s.replace(k, v)
    s = s.encode("latin-1", "replace").decode("latin-1")
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _when(ts) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, ZoneInfo(config.DISPLAY_TZ)).strftime("%d %b %H:%M")


def _load(days: int | None):
    s = account.settings()
    rows = db.q("SELECT t.*, a.probability, a.regime, a.setup_key a_setup, a.components, a.notes, a.source a_source, "
                "a.is_signal, a.price a_price, a.blocked_reason FROM trades t JOIN analyses a ON a.id=t.analysis_id "
                "WHERE t.demo=1 AND t.id > ? ORDER BY t.closed_at", (int(s.get("start_after_id", 0)),))
    cut = time.time() - days * 86400 if days else 0
    rows = [db.decode(r) for r in rows if (r.get("closed_at") or 0) >= cut]
    for r in rows:
        r["pnl"] = (r["r_multiple"] or 0) * (r["risk_amount"] or 0)
        for k in ("components", "notes"):
            if isinstance(r.get(k), str):
                try:
                    r[k] = json.loads(r[k])
                except Exception:
                    pass
    closed = [r for r in rows if r["r_multiple"] is not None and r["status"] not in ("open", "pending", "missed", "cancelled")]
    missed = [r for r in rows if r["status"] == "missed"]
    return s, closed, missed


def _stats(rs: list[dict]) -> dict:
    n = len(rs)
    if not n:
        return {"n": 0}
    rr = [r["r_multiple"] for r in rs]
    wins = [x for x in rr if x > 0]
    losses = [x for x in rr if x <= 0]
    return {"n": n, "win": len([r for r in rs if r["status"] == "won"]) / n, "avg_r": sum(rr) / n,
            "net": sum(r["pnl"] for r in rs),
            "pf": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else None}


def _prob_bucket(p) -> str:
    if p is None:
        return "unknown"
    lo = int(p * 20) * 5
    return f"{lo}-{lo + 5}%"


def build(days: int | None = None) -> bytes:
    s, closed, missed = _load(days)
    cur = s.get("currency", "USD")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title="Sigma paper trades report",
                            author="Sigma trading robot")
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, spaceAfter=4)
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4)
    P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9, leading=11.5, alignment=TA_LEFT)
    S = ParagraphStyle("s", parent=P, fontSize=8, leading=10)
    M = ParagraphStyle("m", parent=S, textColor=colors.HexColor("#555555"))
    grid = TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9edf3")),
                       ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8ced8")),
                       ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                       ("BOTTOMPADDING", (0, 0), (-1, -1), 2)])

    def money(v):
        return "" if v is None else f"{v:,.2f} {cur}"

    def pct(v):
        return "-" if v is None else f"{100 * v:.0f}%"

    def r2(v):
        return "-" if v is None else f"{v:+.2f}"

    out = [Paragraph("Sigma - closed paper trades report", H1),
           Paragraph(clean(f"Demo account, {'last ' + str(days) + ' days' if days else 'all trades since the account started'}. "
                           f"Generated {datetime.now(ZoneInfo(config.DISPLAY_TZ)).strftime('%d %b %Y %H:%M')} "
                           f"({config.DISPLAY_TZ}). R = the trade's stop distance: -1R is a full stop-loss, "
                           f"+{config.REWARD_R:g}R the take profit."), M), Spacer(1, 6)]

    # 1. summary
    a = _stats(closed)
    if not a["n"]:
        out.append(Paragraph("No closed trades in this period yet.", P))
        doc.build(out)
        return buf.getvalue()
    results = defaultdict(int)
    for r in closed:
        results[r["status"]] += 1
    exp = a["avg_r"]
    out.append(Paragraph("1. Summary", H2))
    out.append(Table([["Trades", "Won / lost / breakeven / expired", "Win rate", "Average R", "Profit factor", "Net P&L",
                       "Avg winner (R)", "Avg loser (R)"],
                      [a["n"], f"{results['won']} / {results['lost']} / {results['breakeven']} / {results['expired']}", pct(a["win"]),
                       r2(exp), "-" if a["pf"] is None else f"{a['pf']:.2f}", money(a["net"]),
                       r2(_avg([r["r_multiple"] for r in closed if r["r_multiple"] > 0])),
                       r2(_avg([r["r_multiple"] for r in closed if r["r_multiple"] <= 0]))]], style=grid))
    be = 1 / (1 + config.REWARD_R)
    out.append(Paragraph(clean(f"With a {config.REWARD_R:g}:1 target the break-even win rate is about {100 * be:.0f}% "
                               f"before costs; average R above 0 means the robot is profitable on these trades."), M))

    # 2. breakdowns
    out.append(Paragraph("2. Where it wins and where it loses", H2))
    splits = [
        ("Market", lambda r: r["symbol"]), ("Timeframe", lambda r: r["timeframe"]),
        ("Order type", lambda r: r.get("order_type") or "?"),
        ("Direction", lambda r: "BUY" if r["direction"] > 0 else "SELL"),
        ("Market regime", lambda r: r.get("regime") or "?"), ("Setup", lambda r: r.get("a_setup") or r.get("setup_key") or "?"),
        ("Asset class", lambda r: _asset(r["symbol"])), ("Sigma's probability", lambda r: _prob_bucket(r.get("probability"))),
        ("Who chose it", lambda r: "your chat analysis" if r.get("a_source") == "chat" else "robot signal"),
    ]
    for title, key in splits:
        g = defaultdict(list)
        for r in closed:
            g[key(r)].append(r)
        body = [[title, "Trades", "Win rate", "Avg R", "Net P&L", "Profit factor"]]
        for k, rs in sorted(g.items(), key=lambda kv: -len(kv[1]))[:25]:
            x = _stats(rs)
            body.append([clean(k), x["n"], pct(x["win"]), r2(x["avg_r"]), money(x["net"]),
                         "-" if x["pf"] is None else f"{x['pf']:.2f}"])
        t = Table(body, colWidths=[70 * mm, 20 * mm, 22 * mm, 22 * mm, 35 * mm, 25 * mm], style=grid)
        out.append(KeepTogether([t, Spacer(1, 5)]))

    # 3. warning signs
    out.append(Paragraph("3. Warning signs", H2))
    gave = [r for r in closed if (r.get("mfe_r") or 0) >= 1 and r["r_multiple"] < 0]
    out.append(Paragraph(clean(f"Trades that were +1R or more in profit and still ended as a loss: {len(gave)} of "
                               f"{len([r for r in closed if r['r_multiple'] < 0])} losers. A high share means trades "
                               f"should be protected earlier (breakeven / trailing stop) or targets are too far."), P))
    if gave:
        out.append(Table([["Closed", "Market", "Order", "Best (R)", "Result (R)"]] +
                         [[_when(r["closed_at"]), clean(f"{r['symbol']} {r['timeframe']}"), clean(r.get("order_type")),
                           r2(r.get("mfe_r")), r2(r["r_multiple"])] for r in gave[-30:]], style=grid))
    # factors that pointed the wrong way
    agree = defaultdict(lambda: [0, 0, 0, 0])     # factor -> [agreed&won, agreed&lost, against&won, against&lost]
    for r in closed:
        comps = r.get("components") if isinstance(r.get("components"), dict) else {}
        won = r["r_multiple"] > 0
        for f, v in comps.items():
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if abs(v) < 0.15:
                continue
            with_trade = v * r["direction"] > 0
            agree[f][(0 if with_trade else 2) + (0 if won else 1)] += 1
    if agree:
        body = [["Factor", "Agreed with trade: win rate (n)", "Against the trade: win rate (n)", "Reading"]]
        for f, (aw, al, xw, xl) in sorted(agree.items()):
            wa = aw / (aw + al) if aw + al else None
            wx = xw / (xw + xl) if xw + xl else None
            note = ""
            if wa is not None and wx is not None and aw + al >= 5 and xw + xl >= 5:
                note = ("helps" if wa > wx + 0.08 else "misleads" if wx > wa + 0.08 else "little effect")
            body.append([clean(f), f"{pct(wa)} ({aw + al})", f"{pct(wx)} ({xw + xl})", note])
        out.append(Spacer(1, 4))
        out.append(Paragraph("How often each factor was right: win rate when it agreed with the trade vs when it "
                             "pointed the other way (only clear readings, |score| >= 0.15).", S))
        out.append(Table(body, colWidths=[50 * mm, 55 * mm, 55 * mm, 35 * mm], style=grid))

    # 4. missed entries
    out.append(Paragraph("4. Missed entries", H2))
    if missed:
        out.append(Paragraph(clean(f"{len(missed)} limit/stop orders never filled although price moved the predicted way "
                                   f"by 1R or more - the direction was right but the entry was too far away."), P))
        out.append(Table([["Created", "Market", "Order", "Planned entry", "Moved (R)", "Note"]] +
                         [[_when(r["opened_at"]), clean(f"{r['symbol']} {r['timeframe']}"), clean(r.get("order_type")),
                           f"{r['entry']:.5g}", r2(r.get("missed_r")), Paragraph(clean(r.get("lesson") or ""), S)]
                          for r in missed[-40:]], colWidths=[24 * mm, 32 * mm, 24 * mm, 24 * mm, 18 * mm, 150 * mm],
                         style=grid))
    else:
        out.append(Paragraph("None in this period.", P))

    # 5. every trade
    out.append(PageBreak())
    out.append(Paragraph("5. Every closed trade, with Sigma's reasoning", H2))
    for i, r in enumerate(reversed(closed), 1):
        comps = r.get("components") if isinstance(r.get("components"), dict) else {}
        top = sorted(((k, float(v)) for k, v in comps.items() if _num(v)), key=lambda kv: -abs(kv[1]))[:8]
        notes = r.get("notes") if isinstance(r.get("notes"), list) else []
        head = Table([[clean(f"#{i}  {r['symbol']} {r['timeframe']}  {r.get('order_type') or ''}"),
                       clean(f"{r['status'].upper()}  {r2(r['r_multiple'])}R  {money(r['pnl'])}"),
                       clean(f"opened {_when(r['opened_at'])} - closed {_when(r['closed_at'])}")]],
                     colWidths=[85 * mm, 75 * mm, 113 * mm],
                     style=TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica-Bold", 9),
                                       ("BACKGROUND", (0, 0), (-1, -1),
                                        colors.HexColor("#e3f4e8" if r["r_multiple"] > 0 else "#fbe4e4")),
                                       ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        facts = (f"Entry {r['entry']:.5g} | SL {r['sl']:.5g} | TP {r['tp']:.5g} | exit {_fmt(r.get('exit_price'))} | "
                 f"probability {pct(r.get('probability'))} | regime {r.get('regime') or '?'} | "
                 f"setup {r.get('a_setup') or r.get('setup_key') or '?'} | best {r2(r.get('mfe_r'))}R, "
                 f"worst {r2(r.get('mae_r'))}R | {'your chat analysis' if r.get('a_source') == 'chat' else 'robot signal'}")
        block = [head, Paragraph(clean(facts), S)]
        if top:
            block.append(Paragraph(clean("Factors (+ favours buy, - favours sell): " +
                                         ", ".join(f"{k} {v:+.2f}" for k, v in top)), S))
        if notes:
            block.append(Paragraph("<b>Analysis at the time:</b> " + clean(" / ".join(str(n) for n in notes[:8])), S))
        block.append(Paragraph("<b>Lesson after it closed:</b> " + clean(r.get("lesson") or "(none written)"), S))
        block.append(Spacer(1, 6))
        out.append(KeepTogether(block))

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(landscape(A4)[0] - 12 * mm, 7 * mm, f"Sigma paper trades report - page {d.page}")
        canvas.restoreState()

    doc.build(out, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def _avg(xs):
    return sum(xs) / len(xs) if xs else None


def _num(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.5g}"


def _asset(sym: str) -> str:
    try:
        return resolve(sym).asset_class
    except Exception:
        return "?"
