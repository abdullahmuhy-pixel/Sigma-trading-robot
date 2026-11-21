"""Journal: every analysis, every paper trade, the learning model, chat.

Two backends with the same API:
- SQLite (default, a file next to the app) for running on one computer.
- PostgreSQL when DATABASE_URL is set (e.g. a free Supabase database). Use this when the
  robot is hosted, so every device - phone, laptop, tablet - sees the same data and the
  data survives restarts and redeploys."""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable

from . import config

_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    bar_time REAL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    source TEXT NOT NULL,           -- live | chat | backtest
    price REAL,
    direction INTEGER,              -- 1 long, -1 short, 0 none
    raw_score REAL,
    raw_probability REAL,           -- model output before calibration
    probability REAL,               -- calibrated P(win)
    expectancy REAL,
    is_signal INTEGER DEFAULT 0,
    entry REAL, sl REAL, tp REAL,
    regime TEXT,
    setup_key TEXT,
    components TEXT,                -- json {name: score}
    features TEXT,                  -- json {name: signed feature}
    notes TEXT,                     -- json list of strings
    blocked_reason TEXT,
    call_result TEXT,               -- journal grade: right | wrong | flat | unclear
    call_move_r REAL,               -- furthest move in the called direction (R) before the grade
    call_adverse_r REAL,            -- furthest move against it (R)
    graded_at REAL
);
CREATE INDEX IF NOT EXISTS ix_an_sym ON analyses(symbol, timeframe, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS ux_an_bar ON analyses(symbol, timeframe, source, bar_time)
    WHERE source != 'chat';

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER REFERENCES analyses(id),
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    source TEXT NOT NULL,           -- live | backtest
    direction INTEGER NOT NULL,
    entry REAL NOT NULL, sl REAL NOT NULL, tp REAL NOT NULL,
    opened_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',   -- pending | open | won | lost | expired | cancelled
    origin TEXT NOT NULL DEFAULT 'robot',  -- robot | user (a trade you asked Sigma to review)
    order_type TEXT DEFAULT 'MARKET',      -- BUY, SELL, BUY LIMIT, SELL LIMIT, BUY STOP, SELL STOP
    fill_by REAL,                          -- a pending order not filled by then is cancelled
    demo INTEGER DEFAULT 0,                -- 1 = traded on the demo account (paper money)
    risk_amount REAL,                      -- money at risk on the demo account
    units REAL,                            -- position size on the demo account
    missed_r REAL,                         -- unfilled order: how far price moved the predicted way anyway
    note TEXT,
    exit_price REAL,
    closed_at REAL,
    r_multiple REAL,
    mfe_r REAL DEFAULT 0, mae_r REAL DEFAULT 0,
    setup_key TEXT,
    learned INTEGER DEFAULT 0,
    lesson TEXT
);
CREATE INDEX IF NOT EXISTS ix_tr_status ON trades(status);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    analysis_id INTEGER,
    symbol TEXT, timeframe TEXT, order_type TEXT,
    title TEXT NOT NULL,
    body TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    created_at REAL NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);
"""

log = logging.getLogger("sigma.db")
_conn = None
_backend = "sqlite"


def pg_schema() -> str:
    s = SCHEMA.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
    s = s.replace("INTEGER REFERENCES", "BIGINT REFERENCES")
    return re.sub(r"\bREAL\b", "DOUBLE PRECISION", s)


def to_pg(sql: str) -> str:
    """Translate the small SQLite dialect used in this project to PostgreSQL."""
    sql = sql.replace("?", "%s")
    if sql.lstrip().upper().startswith("INSERT OR IGNORE"):
        sql = re.sub(r"(?i)^\s*INSERT OR IGNORE", "INSERT", sql)
        if " RETURNING " in sql:
            body, ret = sql.split(" RETURNING ", 1)
            sql = f"{body} ON CONFLICT DO NOTHING RETURNING {ret}"
        else:
            sql += " ON CONFLICT DO NOTHING"
    return sql


# columns added after the first release: added to existing databases on start-up
MIGRATIONS = [("analyses", "raw_probability", "REAL"), ("trades", "origin", "TEXT NOT NULL DEFAULT 'robot'"),
              ("trades", "note", "TEXT"), ("trades", "order_type", "TEXT DEFAULT 'MARKET'"),
              ("trades", "fill_by", "REAL"), ("trades", "demo", "INTEGER DEFAULT 0"),
              ("trades", "risk_amount", "REAL"), ("trades", "units", "REAL"), ("trades", "missed_r", "REAL"),
              ("analyses", "call_result", "TEXT"), ("analyses", "call_move_r", "REAL"),
              ("analyses", "call_adverse_r", "REAL"), ("analyses", "graded_at", "REAL")]


def _migrate(execute_fn, pg: bool) -> None:
    for table, col, decl in MIGRATIONS:
        if pg:
            decl = re.sub(r"\bREAL\b", "DOUBLE PRECISION", decl)
            execute_fn(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {decl}")
        else:
            try:
                execute_fn(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
            except Exception:
                pass  # already there


def backend() -> str:
    return _backend


def connect(path: str | None = None):
    global _conn, _backend
    with _lock:
        if _conn is not None and path is None:
            return _conn
        if config.DATABASE_URL and path is None:
            import psycopg
            from psycopg.rows import dict_row
            _conn = psycopg.connect(config.DATABASE_URL, row_factory=dict_row, autocommit=False,
                                    prepare_threshold=None)  # pooled Supabase URLs dislike prepared statements
            _backend = "postgres"
            with _conn.cursor() as cur:
                # keep everything in its own schema so Sigma can share a database with another app
                sch = config.DB_SCHEMA
                if not re.fullmatch(r"[a-z_][a-z0-9_]*", sch):
                    raise ValueError("DB_SCHEMA must be a simple lower-case name")
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {sch}")
                cur.execute(f"SET search_path TO {sch}")
                for stmt in pg_schema().split(";"):
                    if stmt.strip():
                        cur.execute(stmt)
                _migrate(cur.execute, True)
            _conn.commit()
        else:
            _conn = sqlite3.connect(str(path or config.DB_PATH), check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.executescript(SCHEMA)
            _migrate(_conn.execute, False)
            _conn.commit()
            _backend = "sqlite"
        return _conn


def _reconnect() -> None:
    global _conn
    try:
        _conn.close()
    except Exception:
        pass
    _conn = None
    connect()


def _run(sql: str, params: tuple, fetch: bool):
    """Execute one statement; reconnect once if a hosted database dropped the connection."""
    for attempt in (1, 2):
        c = connect()
        try:
            with _lock:
                if _backend == "postgres":
                    with c.cursor() as cur:
                        cur.execute(to_pg(sql), params)
                        rows = [dict(r) for r in cur.fetchall()] if fetch and cur.description else []
                        c.commit()
                        return rows, cur.rowcount, None
                cur = c.execute(sql, params)
                rows = [dict(r) for r in cur.fetchall()] if fetch else []
                c.commit()
                return rows, cur.rowcount if not fetch else len(rows), cur.lastrowid
        except Exception as e:
            try:
                c.rollback()
            except Exception:
                pass
            lost = e.__class__.__name__ in ("OperationalError", "InterfaceError") or "closed" in str(e).lower()
            if _backend == "postgres" and attempt == 1 and lost:
                log.warning("database connection lost, reconnecting: %s", e)
                _reconnect()
                continue
            raise


def execute(sql: str, params: Iterable[Any] = ()) -> None:
    _run(sql, tuple(params), fetch=False)


@contextmanager
def tx():
    """Kept for compatibility: yields an object with .execute(sql, params)."""
    class _X:
        @staticmethod
        def execute(sql, params=()):
            execute(sql, params)
    yield _X()


def q(sql: str, params: Iterable[Any] = ()) -> list[dict]:
    return _run(sql, tuple(params), fetch=True)[0]


def q1(sql: str, params: Iterable[Any] = ()) -> dict | None:
    rows = q(sql, params)
    return rows[0] if rows else None


def insert(table: str, row: dict) -> int:
    row = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()}
    cols = ",".join(row)
    marks = ",".join("?" for _ in row)
    sql = f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({marks})"
    if connect() and _backend == "postgres":
        rows, _, _ = _run(sql + " RETURNING id", tuple(row.values()), fetch=True)
        return rows[0]["id"] if rows else 0
    _, rowcount, lastrowid = _run(sql, tuple(row.values()), fetch=False)
    return lastrowid if rowcount else 0


def update(table: str, row_id: int, fields: dict) -> None:
    fields = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in fields.items()}
    sets = ",".join(f"{k}=?" for k in fields)
    execute(f"UPDATE {table} SET {sets} WHERE id=?", (*fields.values(), row_id))


def kv_get(key: str, default: Any = None) -> Any:
    r = q1("SELECT value FROM kv WHERE key=?", (key,))
    return json.loads(r["value"]) if r else default


def kv_set(key: str, value: Any) -> None:
    execute("INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value), time.time()))


def size_mb() -> float | None:
    try:
        if connect() and _backend == "postgres":
            r = q1("SELECT COALESCE(SUM(pg_total_relation_size(c.oid)),0) b FROM pg_class c "
                   "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=? AND c.relkind='r'",
                   (config.DB_SCHEMA,))
            return round(float(r["b"]) / 1e6, 1)
        p = config.DB_PATH
        return round(p.stat().st_size / 1e6, 1) if p.exists() else 0.0
    except Exception:
        return None


def prune(shadow_days: int | None = None, chat_days: int = 120, alert_days: int = 90) -> dict:
    """Keeps the database small (Supabase free tier: 500 MB shared with your other app).
    Deleted: scanner analyses that were NOT signals (and their shadow trades) older than shadow_days,
    old chat messages and alerts. Signals, your own analyses, your trades and backtests are kept."""
    days = shadow_days or config.KEEP_SHADOW_DAYS
    cut = time.time() - days * 86400
    old = "SELECT id FROM analyses WHERE source='live' AND is_signal=0 AND created_at < ?"
    execute(f"DELETE FROM trades WHERE origin='robot' AND status NOT IN ('open','pending') AND analysis_id IN ({old})", (cut,))
    execute(f"DELETE FROM analyses WHERE id IN ({old}) AND id NOT IN (SELECT analysis_id FROM trades WHERE analysis_id IS NOT NULL)", (cut,))
    execute("DELETE FROM chat WHERE created_at < ?", (time.time() - chat_days * 86400,))
    execute("DELETE FROM alerts WHERE created_at < ?", (time.time() - alert_days * 86400,))
    size = size_mb()
    if size is not None and size > config.MAX_DB_MB and days > 3:   # still too big: prune harder
        return prune(max(3, days // 2), chat_days, alert_days)
    return {"size_mb": size, "kept_shadow_days": days}


def decode(row: dict) -> dict:
    for k in ("components", "features", "notes"):
        if isinstance(row.get(k), str):
            try:
                row[k] = json.loads(row[k])
            except ValueError:
                pass
    return row
