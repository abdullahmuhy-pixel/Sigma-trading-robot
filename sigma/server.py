"""HTTP API + web UI.  Run:  python -m sigma  (then open http://localhost:8000)"""
from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from . import agent, backtest, config, db, engine, learning, paper, scheduler
from .data import market, news
from .symbols import resolve

log = logging.getLogger("sigma")
app = FastAPI(title="Sigma Trading Robot")
STATIC = __import__("pathlib").Path(__file__).parent / "static"


@app.on_event("startup")
def _startup():
    db.connect()
    scheduler.start()


# ------------------------------------------------------------------ login --
PUBLIC = {"/login", "/api/login", "/manifest.webmanifest", "/sw.js", "/icon-192.png", "/icon-512.png",
          "/apple-touch-icon.png", "/api/health"}
COOKIE = "sigma_auth"


def auth_token() -> str:
    return hmac.new(config.APP_PASSWORD.encode(), b"sigma-device-login", hashlib.sha256).hexdigest()


def is_authorised(cookie: str | None, header: str | None = None) -> bool:
    if not config.APP_PASSWORD:
        return True
    tok = auth_token()
    return any(v and hmac.compare_digest(v, tok) for v in (cookie, header))


@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    if path in PUBLIC or is_authorised(request.cookies.get(COOKIE), request.headers.get("x-sigma-token")):
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse({"detail": "login required"}, status_code=401)
    return RedirectResponse("/login")


class LoginIn(BaseModel):
    password: str


@app.post("/api/login")
def login(body: LoginIn):
    if not config.APP_PASSWORD or hmac.compare_digest(body.password, config.APP_PASSWORD):
        r = JSONResponse({"ok": True})
        if config.APP_PASSWORD:
            r.set_cookie(COOKIE, auth_token(), max_age=400 * 86400, httponly=True, samesite="lax",
                         secure=False)
        return r
    time.sleep(1)  # slow down guessing
    raise HTTPException(401, "wrong password")


@app.get("/login")
def login_page():
    return FileResponse(STATIC / "login.html")


@app.get("/api/health")
def health():
    return {"ok": True, "db": db.backend()}


# ------------------------------------------------------------- app shell --
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    return FileResponse(STATIC / "sw.js", media_type="application/javascript")


@app.get("/icon-192.png")
def icon192():
    return FileResponse(STATIC / "icon-192.png")


@app.get("/icon-512.png")
def icon512():
    return FileResponse(STATIC / "icon-512.png")


@app.get("/apple-touch-icon.png")
def apple_icon():
    return FileResponse(STATIC / "icon-192.png")


# ------------------------------------------------------------------- chat --
class ImageIn(BaseModel):
    media_type: str
    data: str  # base64


class ChatIn(BaseModel):
    message: str = ""
    session: str = "default"
    images: list[ImageIn] = []


@app.post("/api/chat")
def chat(body: ChatIn):
    imgs = [i.model_dump() for i in body.images if i.media_type in ("image/png", "image/jpeg", "image/webp", "image/gif")]
    if sum(len(i["data"]) for i in imgs) > 20_000_000:
        raise HTTPException(413, "images too large")
    try:
        return agent.chat(body.message, body.session, images=imgs or None)
    except Exception as e:
        log.exception("chat")
        raise HTTPException(500, str(e))


@app.get("/api/chat/history")
def chat_history(session: str = "default", limit: int = 60, after_id: int = 0):
    """Shared by every device: poll with after_id to receive messages sent elsewhere."""
    if after_id:
        return db.q("SELECT id, role, content, created_at FROM chat WHERE session=? AND id>? ORDER BY id",
                    (session, after_id))
    return db.q("SELECT id, role, content, created_at FROM chat WHERE session=? ORDER BY id DESC LIMIT ?",
                (session, limit))[::-1]


@app.get("/api/my-trades")
def my_trades(limit: int = 100):
    return {"summary": learning.performance("live", origin="user"),
            "trades": db.q("SELECT * FROM trades WHERE origin='user' ORDER BY id DESC LIMIT ?", (limit,))}


class ReviewIn(BaseModel):
    symbol: str
    timeframe: str = "H1"
    direction: str
    entry: float
    stop_loss: float
    take_profit: float | None = None
    note: str = ""


@app.post("/api/review")
def review_trade(body: ReviewIn):
    from . import review
    try:
        return review.review_trade(**body.model_dump())
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/analyze")
def analyze(symbol: str, timeframe: str = "H1", deep: bool = True):
    try:
        if deep:
            from . import deep as deep_mod
            r = deep_mod.analyze(symbol, timeframe, source="chat")
            return {**agent.compact(r), "deep": r["deep"]}
        return agent.compact(engine.analyze(symbol, timeframe, source="chat"))
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/symbols")
def symbols(q: str = "", market: str = ""):
    """Every Deriv instrument, grouped in the UI by market."""
    from .data import deriv_catalog
    items = deriv_catalog.search(q, market, limit=2000)
    if not items:  # Deriv unreachable: fall back to the built-in list
        from .symbols import SYNTHETICS
        items = [{"symbol": s, "name": resolve(s).display, "market_label": "Watchlist", "open": True}
                 for s in config.WATCHLIST] + \
                [{"symbol": k, "name": v, "market_label": "Synthetic indices", "open": True}
                 for k, v in SYNTHETICS.items() if k not in config.WATCHLIST]
    return items


@app.get("/api/candles")
def candles(symbol: str, timeframe: str = "H1", count: int = 300):
    try:
        df = market.candles(symbol, timeframe, count)
    except Exception as e:
        raise HTTPException(400, str(e))
    return [{"time": int(t.timestamp()), "open": r.open, "high": r.high, "low": r.low, "close": r.close}
            for t, r in df.iterrows()]


@app.get("/api/signals")
def signals(limit: int = 50, all: bool = False):
    sql = ("SELECT a.*, t.status, t.r_multiple, t.id trade_id FROM analyses a "
           "LEFT JOIN trades t ON t.analysis_id=a.id WHERE a.source!='backtest'")
    if not all:
        sql += " AND a.is_signal=1"
    return [db.decode(r) for r in db.q(sql + " ORDER BY a.created_at DESC LIMIT ?", (limit,))]


@app.get("/api/trades")
def trades(status: str | None = None, signals_only: bool = False, limit: int = 100):
    sql = ("SELECT t.*, a.is_signal, a.probability FROM trades t JOIN analyses a ON a.id=t.analysis_id "
           "WHERE t.source='live' AND t.origin='robot'")
    params: list = []
    if status:
        sql += " AND t.status=?"
        params.append(status)
    if signals_only:
        sql += " AND a.is_signal=1"
    return db.q(sql + " ORDER BY t.opened_at DESC LIMIT ?", (*params, limit))


@app.get("/api/performance")
def performance():
    return {
        "live_signals": learning.performance("live", True),
        "your_trades": learning.performance("live", origin="user"),
        "live_all": learning.performance("live", False),
        "backtest_signals": learning.performance("backtest", True),
        "equity_live_signals": _equity("live", True),
        "by_symbol": db.q("SELECT t.symbol, COUNT(*) n, AVG(t.r_multiple) exp_r, SUM(CASE WHEN t.r_multiple>0 THEN 1 ELSE 0 END)*1.0/COUNT(*) win "
                          "FROM trades t JOIN analyses a ON a.id=t.analysis_id WHERE t.source='live' AND t.origin='robot' "
                          "AND t.status!='open' AND t.r_multiple IS NOT NULL AND a.is_signal=1 GROUP BY t.symbol ORDER BY exp_r DESC"),
    }


def _equity(source: str, signals_only: bool):
    rows = db.q("SELECT t.closed_at, t.r_multiple FROM trades t JOIN analyses a ON a.id=t.analysis_id "
                "WHERE t.source=? AND t.origin='robot' AND t.r_multiple IS NOT NULL AND t.status!='open'" + (" AND a.is_signal=1" if signals_only else "") +
                " ORDER BY t.closed_at", (source,))
    eq, out = 0.0, []
    for r in rows:
        eq += r["r_multiple"]
        out.append({"t": r["closed_at"], "equity_r": eq})
    return out


@app.get("/api/learning")
def learning_state():
    rep = agent.learning_report()
    rep["weight_history"] = db.kv_get("weight_history", [])[-60:]
    return rep


@app.post("/api/learning/reset")
def learning_reset():
    learning.reset_model()
    return {"ok": True}


class BacktestIn(BaseModel):
    symbol: str
    timeframe: str = "H1"
    bars: int = 3000


@app.post("/api/backtest")
def run_backtest(body: BacktestIn):
    def job():
        try:
            backtest.run(body.symbol, body.timeframe, body.bars)
        except Exception as e:
            backtest.progress()[f"{resolve(body.symbol).name}:{body.timeframe.upper()}"] = {"status": f"error: {e}"}
    threading.Thread(target=job, daemon=True).start()
    return {"started": True}


@app.get("/api/backtest/progress")
def backtest_progress():
    return backtest.progress()


class ScanIn(BaseModel):
    timeframe: str = "H1"


@app.post("/api/scan")
def scan_now(body: ScanIn):
    threading.Thread(target=scheduler.scan, args=(body.timeframe.upper(),), daemon=True).start()
    return {"started": True}


@app.get("/api/account")
def demo_account():
    from . import account
    return account.summary()


class AccountIn(BaseModel):
    start_balance: float | None = None
    currency: str | None = None
    risk_pct: float | None = None
    reset: bool = False


@app.post("/api/account")
def demo_account_settings(body: AccountIn):
    from . import account
    try:
        account.update_settings(**body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return account.summary(with_prices=False)


@app.get("/api/calendar")
def calendar():
    return news.calendar()


@app.get("/api/status")
def status():
    return {"time": time.time(), "tz": config.DISPLAY_TZ, "database": db.backend(),
            "login_enabled": bool(config.APP_PASSWORD), "watchlist": config.WATCHLIST,
            "scan_timeframes": config.SCAN_TIMEFRAMES, "scheduler": scheduler.state,
            "open_trades": len(paper.open_positions()),
            "keys": {"anthropic": bool(config.ANTHROPIC_API_KEY), "twelvedata": bool(config.TWELVEDATA_API_KEY),
                     "fred": bool(config.FRED_API_KEY), "finnhub": bool(config.FINNHUB_API_KEY),
                     "alphavantage": bool(config.ALPHAVANTAGE_API_KEY)},
            "model": learning.model().meta}
