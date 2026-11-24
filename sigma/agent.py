"""Chat analyst: Claude with tools that call the robot's own engine, data
sources, journal and learning state. Claude explains and challenges; the
numbers always come from tools."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from . import backtest, config, db, deep, engine, learning, paper, review
from .analysis import fundamental, sentiment
from .data import market, news
from .symbols import TIMEFRAMES, resolve

log = logging.getLogger("sigma.agent")

SYSTEM = """You are Sigma, a professional trading analyst robot. You analyse forex, metals, crypto,
indices and Deriv synthetic indices using technical analysis, fundamentals (FRED macro data, CFTC COT
positioning), news sentiment and the economic calendar, and you learn from every paper trade you take.

How you work:
- ALWAYS call tools for prices, levels and statistics. Never invent or estimate a number yourself.
- Be open-minded: the robot builds a full buy case AND a full sell case (both_sides) and picks the one the facts
  support. Always show both briefly (order, P(win), expected R, why one was preferred). Never start from an assumed
  direction - even if the user asks "find me a sell", report honestly if the buy case is stronger.
- For any question about one instrument, use deep_analysis (thorough, 1-2 minutes) - quality over speed.
  Use analyze_market only for quick checks or many instruments. Quote the deep result's top-down alignment and
  how look-alike setups from history actually turned out. The engine's
  calibrated probability and expectancy are the robot's honest view; you may veto a signal with reasons
  (e.g. conflicting higher timeframe, news risk) but never upgrade one.
- Quote the robot's track record (performance / learning_report) when giving a signal, so the user sees
  how reliable this kind of setup has been. If a setup has few results, call it "unproven".
- Present every analysis in this order, in plain language a trader can act on:
  1. Headline: instrument, timeframe, ORDER TYPE (BUY, SELL, BUY LIMIT, SELL LIMIT, BUY STOP or SELL STOP),
     setup type, and whether it is a confirmed SIGNAL or only a lean.
  2. Levels: entry, stop loss, and EVERY take profit (TP1, TP2, TP3, TP4) as a small table with its price, R multiple,
     the % chance of reaching it before the stop (p_reach), and the expected R of closing there. Mention that the
     user can choose where to close, and which TP has the best expected value. Also P(win) vs break-even.
  3. Why this direction: the top reasons from explanation.for, each with its actual reading (EMA levels, RSI,
     structure, the higher timeframe, fundamentals, news) - explain what each means for price.
  4. What argues against it: explanation.against, honestly.
  5. Why this order type: order_reason (e.g. why wait for a pullback with a limit, or a breakout with a stop),
     and that an unfilled pending order is cancelled after the stated number of candles.
  6. Invalidation and upcoming high-impact events.
- If there is no signal, say so plainly and what would change your mind.
- When the user describes a trade they are considering ("I want to buy EURUSD at 1.0850, SL 1.0820, TP 1.0920"),
  call review_trade with their exact numbers and give a clear verdict, the probability vs break-even win rate,
  the issues found, and a better entry/stop/target if the numbers suggest one. If a number is missing, use the
  current price for entry and the robot's own levels, and say so. Ask for the timeframe only if you cannot infer it
  (default H1 for intraday, H4 for swing).
- When the user sends a chart screenshot, read the instrument, timeframe and any drawn levels from it, say what you
  see, then verify everything with live data through tools before giving a view.
- Users can review their own past trades with my_trades. For "were your signals right?" use signal_journal; be
  honest about wrong calls and missed entries (direction right, entry never reached), and what was changed because of them.
- Background scanning only covers the user's background watchlist (background_watchlist tool) - that is what costs
  data calls and what triggers notifications. When they ask to watch, add, drop or stop watching a market, change it
  there and tell them the new list and roughly how many scans a day it means.
- Session opens: at each Sydney, Tokyo, London and New York open the robot fast-scans that session's pairs and
  alerts opportunities (marked "fast scan"). When the user wants to act on one, run deep_analysis on it first and
  say whether the deep check confirms or rejects the fast-scan idea.
- Goals: when the user tells you their real balance and a daily/weekly/monthly target, call set_account_goal,
  then explain goal_plan honestly: required return, wins needed, the simulated chance of success and of a
  drawdown, and every warning. When they ask for trades to reach the goal, call find_trades_for_goal and give
  each trade with order type, levels, lot size, money won at TP / lost at SL, and why. Never raise risk above
  their setting to "catch up", never promise a target will be hit, and say clearly when a target is unrealistic.
  When they report a real profit or loss, call log_trade_result.
- You can analyse every instrument Deriv lists. If the user names one loosely ("V75", "Jump 50", "Boom 500",
  "oil"), pass it through or call list_symbols to find its code.
- Deriv synthetic indices are generated by a random number generator: no news or fundamentals apply and
  technical edges are doubtful; say this when relevant.
- If market_closed is true (weekend/holiday), say the market is closed, give the levels as a plan for the
  reopen only, and warn about gap risk at the open.
- Show times in {tz} (the user's local time).
- Every analysis is automatically journalled and paper-traded; mention that the result will feed learning.
- This is analysis for the user's own decisions, not financial advice; keep that to one short line, at most.
Be concise and direct. Current time: {now}.
"""

TOOLS = [
    {"name": "deep_analysis", "description": "DEFAULT for analysing one instrument (takes 1-2 minutes): top-down across "
     "timeframes, look-alike setups from this instrument's own history and their real outcomes, fundamentals, news, "
     "events, order type and full explanation. Journals and paper-trades it.",
     "input_schema": {"type": "object", "properties": {
         "symbol": {"type": "string", "description": "any Deriv instrument or alias (V75, Boom 1000, XAUUSD...)"},
         "timeframe": {"type": "string", "enum": list(TIMEFRAMES)}}, "required": ["symbol", "timeframe"]}},
    {"name": "analyze_market", "description": "QUICK (seconds) analysis of one instrument on one timeframe: technical "
     "components, fundamentals, news, event risk, learned probability, entry/SL/TP. Journals and paper-trades it.",
     "input_schema": {"type": "object", "properties": {
         "symbol": {"type": "string", "description": "Any Deriv instrument: EURUSD, XAUUSD, BTCUSD, R_75 / 'Volatility 75 Index', 1HZ100V, BOOM1000, JD25, stpRNG, OTC_NDX... (use list_symbols if unsure)"},
         "timeframe": {"type": "string", "enum": list(TIMEFRAMES)}}, "required": ["symbol", "timeframe"]}},
    {"name": "multi_timeframe", "description": "Analyse an instrument on M15, H1, H4 and D1 at once.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
    {"name": "scan_watchlist", "description": "Analyse every watchlist instrument on a timeframe and rank the best setups.",
     "input_schema": {"type": "object", "properties": {"timeframe": {"type": "string", "enum": list(TIMEFRAMES)},
                                                       "symbols": {"type": "array", "items": {"type": "string"}}},
                      "required": ["timeframe"]}},
    {"name": "list_symbols", "description": "Search every instrument Deriv offers (forex, synthetic indices such as "
     "Volatility/Boom/Crash/Jump/Step, crypto, commodities, stock indices, baskets). Use it to find the right code.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "e.g. 'volatility 75', 'jump', 'gold'"},
                                                       "market": {"type": "string", "description": "e.g. forex, synthetic_index, cryptocurrency, commodities, indices"}}}},
    {"name": "get_fundamentals", "description": "Macro/fundamental picture for an instrument (rates, yields, COT, risk backdrop).",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
    {"name": "get_news", "description": "Recent headlines relevant to an instrument with sentiment scores.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
    {"name": "economic_calendar", "description": "This week's economic events, optionally for some currencies.",
     "input_schema": {"type": "object", "properties": {"currencies": {"type": "array", "items": {"type": "string"}},
                                                       "impact": {"type": "string", "enum": ["High", "Medium", "Low", "any"]}}}},
    {"name": "crowd_positioning", "description": "Where retail traders are positioned in an instrument "
     "(Myfxbook community outlook and IG client sentiment): % long/short, average crowd entry prices, and the "
     "contrarian reading. Works for forex, metals and IG markets; empty if no source is connected.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
    {"name": "market_hours", "description": "Official Deriv trading hours for an instrument: open now or not, "
     "minutes until it closes or opens, and any special events today (early closes, holidays).",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
    {"name": "get_price", "description": "Latest price of an instrument.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
    {"name": "performance", "description": "Paper-trading track record (signals only by default).",
     "input_schema": {"type": "object", "properties": {
         "symbol": {"type": "string"}, "timeframe": {"type": "string"},
         "source": {"type": "string", "enum": ["live", "backtest", "all"]},
         "include_shadow": {"type": "boolean", "description": "include non-signal shadow trades"}}}},
    {"name": "learning_report", "description": "What the robot has learned: component weights, calibration, best/worst setups, lessons.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}}}},
    {"name": "review_trade", "description": "Grade a trade the USER is considering (their entry/SL/TP) against live "
     "analysis: verdict, P(win) vs break-even, expected R, issues (tight stop, levels in the way, news, robot disagrees). "
     "Tracks it as a paper trade.",
     "input_schema": {"type": "object", "properties": {
         "symbol": {"type": "string"}, "timeframe": {"type": "string", "enum": list(TIMEFRAMES)},
         "direction": {"type": "string", "enum": ["buy", "sell"]},
         "entry": {"type": "number"}, "stop_loss": {"type": "number"},
         "take_profit": {"type": "number", "description": "optional; defaults to 2R"},
         "track": {"type": "boolean", "description": "paper-track it (default true)"},
         "note": {"type": "string"}},
         "required": ["symbol", "timeframe", "direction", "entry", "stop_loss"]}},
    {"name": "my_trades", "description": "The user's reviewed trades and how they turned out, with their win rate and expectancy.",
     "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
    {"name": "demo_account", "description": "The robot's demo account (paper money): balance, equity, profit/loss today, "
     "this week, this month, open positions with live P&L, pending orders, recent closed trades. Can also change its "
     "settings (starting balance, currency, risk per trade) or reset it.",
     "input_schema": {"type": "object", "properties": {
         "start_balance": {"type": "number"}, "currency": {"type": "string"}, "risk_pct": {"type": "number"},
         "reset": {"type": "boolean", "description": "start the demo account again from the starting balance"}}}},
    {"name": "set_account_goal", "description": "Save the user's REAL account balance, currency and profit target "
     "for a day, week or month (and optionally risk % per trade and how many trades a day they can take).",
     "input_schema": {"type": "object", "properties": {
         "balance": {"type": "number"}, "currency": {"type": "string", "description": "e.g. ZAR, USD"},
         "target": {"type": "number", "description": "profit wanted, in the account currency"},
         "period": {"type": "string", "enum": ["day", "week", "month"]},
         "risk_pct": {"type": "number", "description": "default 1"}, "trades_per_day": {"type": "integer"}},
         "required": ["balance", "currency", "target", "period"]}},
    {"name": "goal_plan", "description": "Honest plan for the saved goal: required return, wins needed, Monte Carlo "
     "chance of reaching it from the robot's real track record, drawdown risk, progress this period, warnings.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "find_trades_for_goal", "description": "Scan for the best current setups, deep-check the top ones, and "
     "size each for the user's account and goal (lots, money at stake, contribution to the goal). Takes a few minutes.",
     "input_schema": {"type": "object", "properties": {
         "timeframe": {"type": "string", "enum": list(TIMEFRAMES)}, "max_trades": {"type": "integer"},
         "symbols": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "log_trade_result", "description": "Record profit (+) or loss (-) the user made on their REAL account, "
     "so goal progress and their balance stay up to date.",
     "input_schema": {"type": "object", "properties": {"amount": {"type": "number"}, "note": {"type": "string"}},
                      "required": ["amount"]}},
    {"name": "signal_journal", "description": "The journal of every signal and analysis with its grade (was the call "
     "right or wrong by how the market moved), order outcomes including MISSED entries (price went the right way but "
     "never reached the limit/stop), and accuracy by side, timeframe, market and order type.",
     "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
    {"name": "background_watchlist", "description": "The ONLY markets the robot scans on its own at every candle "
     "close (and the only ones that trigger opportunity notifications), plus their timeframes. Show, add, remove or "
     "replace markets, change timeframes, or switch background scanning on/off.",
     "input_schema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["show", "add", "remove", "set", "timeframes", "enable", "disable"]},
         "symbols": {"type": "array", "items": {"type": "string"}, "description": "for add/remove/set: any Deriv market or alias"},
         "timeframes": {"type": "array", "items": {"type": "string", "enum": list(TIMEFRAMES)}}},
         "required": ["action"]}},
    {"name": "trading_sessions", "description": "Session-open scanning: show the schedule (next opens in the user's "
     "time), the pairs each session scans, run a fast scan of a session now, or change a session's pairs / switch it off.",
     "input_schema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["schedule", "scan_now", "set_pairs", "enable", "disable"]},
         "session": {"type": "string", "enum": ["Sydney", "Tokyo", "London", "New York"]},
         "pairs": {"type": "array", "items": {"type": "string"}}}, "required": ["action"]}},
    {"name": "auto_bot", "description": "The automatic scalping/short-trade bot. 'status' shows balance, equity, "
     "today's P&L vs target, open positions, risk state and recent log; 'stats' shows its graded track record "
     "(by mode/market, exit reasons, trades that gave back profit) and what it learned; 'start'/'stop' run it; "
     "'set' changes daily_target, risk limits or markets. Switching to a LIVE account is only possible on the Bot "
     "page with the user's typed confirmation - never claim you switched it to live.",
     "input_schema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["status", "stats", "start", "stop", "set"]},
         "daily_target": {"type": "number"}, "daily_loss_pct": {"type": "number"},
         "max_drawdown_pct": {"type": "number"}, "symbols": {"type": "array", "items": {"type": "string"}}},
         "required": ["action"]}},
    {"name": "open_paper_trades", "description": "Currently open paper trades.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_backtest", "description": "Walk-forward backtest on history; also trains the learning model. Can take a minute.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"},
                                                       "timeframe": {"type": "string", "enum": list(TIMEFRAMES)},
                                                       "bars": {"type": "integer"}}, "required": ["symbol", "timeframe"]}},
    {"name": "position_size", "description": "Units/lots for a given account balance, risk % and stop distance.",
     "input_schema": {"type": "object", "properties": {
         "symbol": {"type": "string"}, "balance": {"type": "number"}, "risk_percent": {"type": "number"},
         "entry": {"type": "number"}, "stop_loss": {"type": "number"},
         "account_currency": {"type": "string", "description": "e.g. ZAR, USD"}},
         "required": ["symbol", "balance", "risk_percent", "entry", "stop_loss", "account_currency"]}},
]


def _local(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, ZoneInfo(config.DISPLAY_TZ)).strftime("%Y-%m-%d %H:%M %Z")


def _r(x, n=5):
    return None if x is None else round(float(x), n)


def compact(res: dict) -> dict:
    ctx = res["context"]
    fund = res.get("fundamental") or {}
    nws = res.get("news") or {}
    ev = res.get("events") or {}
    return {
        "symbol": res["display"], "timeframe": res["timeframe"], "price": res["price"],
        "bar_closed": _local(res["bar_time"]), "data_source": res.get("provider"),
        "regime": res["regime"], "adx": _r(res["adx"], 1), "atr": _r(res["atr"]),
        "direction": "LONG" if res["direction"] > 0 else "SHORT",
        "order_type": res["plan"]["order_type"], "setup_type": res["plan"]["setup_type"],
        "order_reason": res["plan"]["order_reason"],
        "live_price": _r(res.get("live_price")), "live_adjustment": res["plan"].get("live_adjustment"),
        "pending_order_expires_after_candles": res["plan"]["pending_bars"] or None,
        "explanation": res.get("explanation"),
        "take_profits": [{"name": t["name"], "price": _r(t["price"]), "r": t["r"], "p_reach": _r(t["p_reach"], 3),
                          "expected_r_if_closed_here": _r(t["expected_r_if_closed_here"], 3),
                          "history_hits": t.get("history_hits"), "main": t["main"]} for t in res.get("take_profits", [])],
        "both_sides": {k: {**{kk: (_r(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "take_profits"},
                           "tp_odds": {t["name"]: _r(t["p_reach"], 3) for t in v.get("take_profits", [])}}
                       for k, v in (res.get("both_sides") or {}).items()},
        "is_signal": res["is_signal"], "blocked_reason": res["blocked_reason"],
        "market_closed": res.get("market_closed", False),
        "p_win_calibrated": _r(res["probability"], 3), "p_long": _r(res["p_long"], 3), "p_short": _r(res["p_short"], 3),
        "expected_R_after_costs": _r(res["expectancy"], 3),
        "entry": _r(res["entry"]), "stop_loss": _r(res["sl"]), "take_profit": _r(res["tp"]), "reward_R": res["reward_r"],
        "component_scores": {k: _r(v, 2) for k, v in res["components"].items()},
        "technical": {k: (_r(v) if isinstance(v, float) else v) for k, v in ctx.items() if k != "levels"},
        "key_levels": [_r(l["price"]) for l in ctx.get("levels", [])],
        "fundamental": {"score": _r(fund.get("score"), 2), "parts": fund.get("parts"), "notes": fund.get("notes")},
        "news": {"score": _r(nws.get("score"), 2), "headlines": (nws.get("headlines") or [])[:6]},
        "upcoming_events": [{**e, "time": _local(e["time"])} for e in (ev.get("upcoming") or [])[:8]],
        "notes": res["notes"], "setup_key": res["setup_key"],
        "setup_track_record": learning.setup_stats().get(res["setup_key"]),
        "journal_id": res.get("analysis_id"), "paper_trade_id": res.get("trade_id"),
    }


def run_tool(name: str, args: dict) -> object:
    if name == "deep_analysis":
        r = deep.analyze(args["symbol"], args["timeframe"], source="chat")
        return {**compact(r), "deep": r["deep"]}
    if name == "analyze_market":
        return compact(engine.analyze(args["symbol"], args["timeframe"], source="chat"))
    if name == "multi_timeframe":
        out = {}
        for tf in ("M15", "H1", "H4", "D1"):
            try:
                r = engine.analyze(args["symbol"], tf, source="chat")
                out[tf] = {k: v for k, v in compact(r).items() if k not in ("upcoming_events", "news", "fundamental")}
            except Exception as e:
                out[tf] = {"error": str(e)}
        return out
    if name == "scan_watchlist":
        rows = []
        from . import watchlist
        for s in args.get("symbols") or watchlist.get()["symbols"]:
            try:
                r = engine.analyze(s, args["timeframe"], source="chat")
                rows.append({"symbol": r["display"], "direction": "LONG" if r["direction"] > 0 else "SHORT",
                             "p_win": _r(r["probability"], 3), "expected_R": _r(r["expectancy"], 3),
                             "is_signal": r["is_signal"], "regime": r["regime"], "blocked": r["blocked_reason"],
                             "entry": _r(r["entry"]), "sl": _r(r["sl"]), "tp": _r(r["tp"])})
            except Exception as e:
                rows.append({"symbol": s, "error": str(e)})
        return sorted(rows, key=lambda r: -(r.get("expected_R") or -9))
    if name == "list_symbols":
        from .data import deriv_catalog
        return [{"code": i["symbol"], "name": i["name"], "market": i["market_label"],
                 "submarket": i.get("submarket_label"), "open": i["open"]}
                for i in deriv_catalog.search(args.get("query", ""), args.get("market", ""), limit=60)]
    if name == "get_fundamentals":
        return fundamental.analyze(resolve(args["symbol"]))
    if name == "get_news":
        return sentiment.analyze(resolve(args["symbol"]))
    if name == "economic_calendar":
        ccys = {c.upper() for c in args.get("currencies") or []}
        imp = args.get("impact", "any")
        return [{**e, "time": _local(e["time"])} for e in news.calendar()
                if (not ccys or e["currency"] in ccys) and (imp in (None, "any") or e["impact"] == imp)][:60]
    if name == "crowd_positioning":
        from .data import positioning
        out = positioning.analyze(args["symbol"])
        out["connected"] = positioning.configured()
        return out
    if name == "market_hours":
        from .data.deriv_info import market_hours
        return market_hours(args["symbol"])
    if name == "get_price":
        return {"symbol": args["symbol"], "price": market.last_price(args["symbol"]), "time": _local(time.time())}
    if name == "performance":
        src = args.get("source", "live")
        return learning.performance(None if src == "all" else src, not args.get("include_shadow", False),
                                    resolve(args["symbol"]).name if args.get("symbol") else None,
                                    args.get("timeframe"))
    if name == "learning_report":
        return learning_report(args.get("symbol"))
    if name == "review_trade":
        return review.review_trade(**args)
    if name == "my_trades":
        rows = db.q("SELECT id, symbol, timeframe, direction, entry, sl, tp, status, r_multiple, opened_at, "
                    "closed_at, lesson, note FROM trades WHERE origin='user' ORDER BY id DESC LIMIT ?",
                    (int(args.get("limit", 20)),))
        return {"summary": learning.performance("live", origin="user"),
                "trades": [{**r, "opened_at": _local(r["opened_at"]), "closed_at": _local(r["closed_at"])} for r in rows]}
    if name == "demo_account":
        from . import account
        if any(args.get(k) is not None for k in ("start_balance", "currency", "risk_pct")) or args.get("reset"):
            account.update_settings(args.get("start_balance"), args.get("currency"), args.get("risk_pct"),
                                    bool(args.get("reset")))
        sm = account.summary()
        keep = ("symbol", "timeframe", "direction", "order_type", "entry", "sl", "tp", "status", "r_multiple",
                "pnl", "unrealized_pnl", "risk_amount", "opened_at", "closed_at")
        trim = lambda rows: [{k: (_local(v) if k.endswith("_at") else v) for k, v in r.items() if k in keep} for r in rows]  # noqa: E731
        return {**{k: v for k, v in sm.items() if k not in ("open", "pending", "closed", "equity_curve")},
                "open": trim(sm["open"]), "pending": trim(sm["pending"]), "recent_closed": trim(sm["closed"][:15])}
    if name == "set_account_goal":
        from . import goals
        goals.set_goal(**args)
        return goals.plan()
    if name == "goal_plan":
        from . import goals
        return goals.plan()
    if name == "find_trades_for_goal":
        from . import goals
        return goals.find_trades(args.get("timeframe", "H1"), args.get("symbols"), int(args.get("max_trades", 3)))
    if name == "log_trade_result":
        from . import goals
        goals.log_result(args["amount"], args.get("note", ""))
        return goals.plan()
    if name == "signal_journal":
        from . import journal
        return {"accuracy": journal.call_accuracy(), "signals_only": journal.call_accuracy(signals_only=True),
                "orders": journal.order_stats(cache_seconds=0),
                "recent": [{**r, "created_at": _local(r["created_at"])} for r in journal.recent(int(args.get("limit", 25)))]}
    if name == "background_watchlist":
        from . import watchlist
        act = args["action"]
        if act == "add":
            watchlist.add(args.get("symbols") or [])
        elif act == "remove":
            watchlist.remove(args.get("symbols") or [])
        elif act == "set":
            watchlist.set_symbols(args.get("symbols") or [])
        elif act == "timeframes":
            watchlist.set_timeframes(args.get("timeframes") or [])
        elif act in ("enable", "disable"):
            watchlist.set_enabled(act == "enable")
        return watchlist.get()
    if name == "trading_sessions":
        from . import sessions
        act, ses = args["action"], args.get("session")
        if act == "scan_now":
            return sessions.scan(ses)
        if act == "set_pairs":
            sessions.update(ses, pairs=args.get("pairs") or [])
        elif act in ("enable", "disable"):
            sessions.update(ses, enabled=act == "enable")
        return [{k: v for k, v in x.items() if k != "next_open"} for x in sessions.schedule()]
    if name == "auto_bot":
        from .autobot import bot, review
        act = args["action"]
        if act == "stats":
            return {"stats": review.stats(bot.settings()["account"]), "learning": review.learning()}
        if act == "set":
            bot.update_settings(**{k: args[k] for k in ("daily_target", "daily_loss_pct", "max_drawdown_pct", "symbols")
                                   if args.get(k) is not None})
        elif act == "start":
            bot.start()
        elif act == "stop":
            bot.stop()
        st = bot.status()
        st["positions"] = [{k: t.get(k) for k in ("mode", "symbol", "direction", "entry", "tp_usd", "sl_usd", "pnl_now")}
                           for t in st.get("positions", [])]
        st["log"] = [x["msg"] for x in st["log"][-12:]]
        return st
    if name == "open_paper_trades":
        return [{**t, "opened_at": _local(t["opened_at"]), "expires_at": _local(t["expires_at"])}
                for t in paper.open_positions()[:40]]
    if name == "run_backtest":
        return backtest.run(args["symbol"], args["timeframe"], int(args.get("bars", 2000)))
    if name == "position_size":
        return position_size(**args)
    raise ValueError(f"unknown tool {name}")


def learning_report(symbol: str | None = None) -> dict:
    m = learning.model()
    lessons_sql = "SELECT symbol, timeframe, r_multiple, lesson, closed_at FROM trades WHERE lesson IS NOT NULL"
    params = []
    if symbol:
        lessons_sql += " AND symbol=?"
        params.append(resolve(symbol).name)
    lessons = db.q(lessons_sql + " ORDER BY closed_at DESC LIMIT 8", params)
    stats = learning.setup_stats()
    ranked = sorted(stats.items(), key=lambda kv: -kv[1]["shrunk_expectancy"])
    return {
        "trained_on": {"live_trades": m.meta["n_live"], "backtest_trades": m.meta["n_backtest"]},
        "weights": {b: m.weights()[b] for b in learning.BASE},
        "regime_adjustments": {k: v for k, v in m.weights().items() if "@" in k and abs(v) > 0.05},
        "calibration_slope": round(m.meta["calib"][1], 3),
        "calibration_meaning": "1.0 = model confidence matches results; <1 = model was overconfident and is discounted",
        "calibration_table": learning.calibration(),
        "best_setups": [{"setup": k, **v} for k, v in ranked[:5] if v["n"] >= 5],
        "worst_setups": [{"setup": k, **v} for k, v in ranked[-5:] if v["n"] >= 5],
        "insights": learning.insights(),
        "recent_lessons": [{**l, "closed_at": _local(l["closed_at"])} for l in lessons],
        "live_signal_performance": learning.performance("live"),
        "long_vs_short": learning.direction_balance(),
        "call_accuracy": __import__("sigma.journal", fromlist=["x"]).call_accuracy(),
        "order_fill_stats": __import__("sigma.journal", fromlist=["x"]).order_stats(),
    }


def fx_rate(a: str, b: str) -> float:
    if a == b:
        return 1.0
    for pair, inv in ((a + b, False), (b + a, True)):
        try:
            px = market.last_price(pair)
            return 1 / px if inv else px
        except Exception:
            pass
    if "USD" not in (a, b):
        return fx_rate(a, "USD") * fx_rate("USD", b)
    raise ValueError(f"no rate {a}{b}")


def position_size(symbol, balance, risk_percent, entry, stop_loss, account_currency):
    inst = resolve(symbol)
    risk_amount = balance * risk_percent / 100
    dist = abs(entry - stop_loss)
    if dist <= 0:
        return {"error": "stop loss must differ from entry"}
    acct = account_currency.upper()
    quote = inst.quote or "USD"
    try:
        conv = fx_rate(quote, acct)   # 1 unit of quote currency = conv account currency
    except Exception:
        return {"error": f"could not convert {quote} to {acct}"}
    units = risk_amount / (dist * conv)
    contract = {"forex": 100_000, "metal": 100 if inst.base == "XAU" else 5000, "crypto": 1}.get(inst.asset_class, 1)
    return {"risk_amount": round(risk_amount, 2), "account_currency": acct, "units": round(units, 4),
            "lots": round(units / contract, 3), "contract_size_assumed": contract,
            "note": "Check your broker's contract size and minimum lot; synthetic indices use the broker's own lot rules."}


# ------------------------------------------------------------------ chat ---
def _client():
    import anthropic
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def chat(message: str, session: str = "default", max_turns: int = 12, images: list[dict] | None = None) -> dict:
    if not config.ANTHROPIC_API_KEY:
        return {"reply": "Set ANTHROPIC_API_KEY in .env to enable the chat. The robot's scanning, paper "
                         "trading and learning work without it.", "tools_used": []}
    hist = db.q("SELECT role, content FROM chat WHERE session=? ORDER BY id DESC LIMIT 20", (session,))[::-1]
    messages = [{"role": h["role"], "content": h["content"]} for h in hist]
    if images:
        blocks = [{"type": "image", "source": {"type": "base64", "media_type": im["media_type"], "data": im["data"]}}
                  for im in images[:4]]
        messages.append({"role": "user", "content": blocks + [{"type": "text", "text": message or "Analyse this chart."}]})
    else:
        messages.append({"role": "user", "content": message})
    stored = message + (f"\n\n[{len(images)} chart image(s) attached]" if images else "")
    db.insert("chat", {"session": session, "created_at": time.time(), "role": "user", "content": stored})

    system = SYSTEM.format(tz=config.DISPLAY_TZ, now=_local(time.time()))
    tools = list(TOOLS)
    if config.CLAUDE_WEB_SEARCH:
        tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 5})
    client = _client()
    used = []
    reply = ""
    thinking = config.CLAUDE_THINKING
    for _ in range(max_turns):
        try:
            extra = {"thinking": {"type": "enabled", "budget_tokens": 6000}} if thinking else {}
            resp = client.messages.create(model=config.CLAUDE_MODEL, max_tokens=12000 if thinking else 4000,
                                          system=system, tools=tools, messages=messages, **extra)
        except Exception as e:
            if thinking:
                log.warning("retrying without extended thinking: %s", e)
                thinking = False
                continue
            if any(t.get("name") == "web_search" for t in tools):
                log.warning("retrying without web search: %s", e)
                tools = [t for t in tools if t.get("name") != "web_search"]
                continue
            raise
        messages.append({"role": "assistant", "content": resp.content})
        reply = "".join(b.text for b in resp.content if b.type == "text")
        calls = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not calls:
            break
        results = []
        for c in calls:
            used.append({"tool": c.name, "input": c.input})
            try:
                out = run_tool(c.name, c.input)
                content = json.dumps(out, default=str)[:60000]
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": content})
            except Exception as e:
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": f"Error: {e}", "is_error": True})
        messages.append({"role": "user", "content": results})
    db.insert("chat", {"session": session, "created_at": time.time(), "role": "assistant", "content": reply})
    return {"reply": reply, "tools_used": used}


def post_mortem(analysis: dict, trade: dict) -> str:
    prompt = (f"A paper trade closed. Write ONE or TWO sentences: the most likely reason it "
              f"{'worked' if trade['r_multiple'] > 0 else 'failed'} and a concrete rule to apply next time. "
              f"No preamble.\nSetup: {analysis['setup_key']}\nComponents: {json.dumps(analysis['components'])}\n"
              f"Notes: {analysis.get('notes')}\nResult: {trade['status']} {trade['r_multiple']:+.2f}R, "
              f"max favourable {trade.get('mfe_r', 0):.2f}R, max adverse {trade.get('mae_r', 0):.2f}R")
    resp = _client().messages.create(model=config.CLAUDE_MODEL, max_tokens=200,
                                     messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in resp.content if b.type == "text").strip()
