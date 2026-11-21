"""Central configuration. Every value can be overridden with an environment
variable or a .env file in the project root."""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = re.split(r"\s+#", v, maxsplit=1)[0]  # allow "KEY=value   # comment"
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _get(name: str, default: str = "") -> str:
    v = os.environ.get(name, default).strip()
    # hosting forms sometimes force a value: treat placeholders as "not set"
    return default if v.lower() in ("none", "-", "n/a", "na", "null", "empty", "skip") else v


def _float(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError:
        return default


# --- Keys (all optional except ANTHROPIC_API_KEY for chat) -------------------
ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = _get("CLAUDE_MODEL", "claude-sonnet-5-5")
CLAUDE_WEB_SEARCH = _get("CLAUDE_WEB_SEARCH", "1") == "1"
CLAUDE_THINKING = _get("CLAUDE_THINKING", "1") == "1"   # let Claude reason step by step before answering
TWELVEDATA_API_KEY = _get("TWELVEDATA_API_KEY")
FRED_API_KEY = _get("FRED_API_KEY")
FINNHUB_API_KEY = _get("FINNHUB_API_KEY")
ALPHAVANTAGE_API_KEY = _get("ALPHAVANTAGE_API_KEY")
DERIV_APP_ID = _get("DERIV_APP_ID", "1089")  # 1089 = Deriv's public test id

# --- Storage -----------------------------------------------------------------
DB_PATH = Path(_get("SIGMA_DB", str(ROOT / "sigma.db")))
DATABASE_URL = _get("DATABASE_URL") if _get("DATABASE_URL").startswith("postgres") else ""   # postgres://... (e.g. Supabase) -> shared across devices
APP_PASSWORD = _get("APP_PASSWORD")   # protects the app when it is on the internet

# --- Display -----------------------------------------------------------------
DISPLAY_TZ = _get("DISPLAY_TZ", "Africa/Johannesburg")

# --- Robot behaviour ---------------------------------------------------------
WATCHLIST = [s.strip() for s in _get(
    "WATCHLIST",
    "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF,NZDUSD,EURJPY,GBPJPY,XAUUSD,"
    "BTCUSD,ETHUSD,USDZAR,R_10,R_25,R_50,R_75,R_100,1HZ75V,1HZ100V,BOOM500,BOOM1000,CRASH500,CRASH1000,"
    "JD25,JD75,stpRNG",
).split(",") if s.strip()]
SCAN_TIMEFRAMES = [s.strip() for s in _get("SCAN_TIMEFRAMES", "M15,H1,H4").split(",") if s.strip()]
AUTO_SCAN = _get("AUTO_SCAN", "1") == "1"
AUTO_PAPER_TRADE = _get("AUTO_PAPER_TRADE", "1") == "1"
MIN_PROBABILITY = _float("MIN_PROBABILITY", 0.45)   # min calibrated P(win) to signal
MIN_EXPECTANCY_R = _float("MIN_EXPECTANCY_R", 0.10)  # min expected R per trade
REWARD_R = _float("REWARD_R", 2.0)                   # main TP (used for paper trades and learning), in R
TP_LEVELS = [float(x) for x in _get("TP_LEVELS", "1,2,3,4").split(",") if x.strip()]  # TP1..TPn, in R
SL_ATR_MULT = _float("SL_ATR_MULT", 1.5)
MAX_HOLD_BARS = _int("MAX_HOLD_BARS", 30)            # expiry, in entry-timeframe bars
NEWS_BLACKOUT_MIN = _int("NEWS_BLACKOUT_MIN", 30)    # no new signals this close to high-impact events
MONITOR_SECONDS = _int("MONITOR_SECONDS", 30)
GATE_MIN_TRADES = _int("GATE_MIN_TRADES", 20)        # samples before a setup can be switched off
GATE_MAX_EXPECTANCY = _float("GATE_MAX_EXPECTANCY", -0.10)

# --- News feeds (RSS). Edit freely; dead feeds are skipped. -----------------
NEWS_FEEDS = [s.strip() for s in _get(
    "NEWS_FEEDS",
    "https://www.investing.com/rss/news_1.rss,"
    "https://www.investing.com/rss/news_95.rss,"
    "https://www.fxstreet.com/rss/news,"
    "https://www.federalreserve.gov/feeds/press_all.xml,"
    "https://www.ecb.europa.eu/rss/press.html,"
    "https://cointelegraph.com/rss",
).split(",") if s.strip()]
CALENDAR_URL = _get("CALENDAR_URL", "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
