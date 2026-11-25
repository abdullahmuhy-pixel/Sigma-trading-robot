import importlib
import json
import os
import sys
import tempfile
import time
import types

import pytest

from sigma import agent, db


def stub_server(monkeypatch):
    fa = types.ModuleType("fastapi")

    class App:
        def __init__(self, **k): pass
        def _d(self, *a, **k): return lambda f: f
        get = post = on_event = middleware = _d

    class HTTPException(Exception):
        def __init__(self, code, detail=""):
            super().__init__(detail)
            self.status_code = code
    fa.FastAPI, fa.HTTPException, fa.Request = App, HTTPException, object
    resp = types.ModuleType("fastapi.responses")
    resp.FileResponse, resp.JSONResponse, resp.RedirectResponse = (lambda p, **k: p), dict, (lambda url: url)
    monkeypatch.setitem(sys.modules, "fastapi", fa)
    monkeypatch.setitem(sys.modules, "fastapi.responses", resp)
    sys.modules.pop("sigma.server", None)
    return importlib.import_module("sigma.server"), HTTPException


def test_saved_and_reopened(monkeypatch):
    db.connect(os.path.join(tempfile.mkdtemp(), "sv.db"))
    server, HTTPException = stub_server(monkeypatch)
    aid = db.insert("analyses", {"created_at": time.time(), "bar_time": time.time(), "symbol": "EURJPY", "timeframe": "H1"})
    tid = db.insert("trades", {"analysis_id": aid, "symbol": "EURJPY", "timeframe": "H1", "source": "live", "direction": -1,
                               "entry": 174.56, "sl": 174.85, "tp": 173.97, "opened_at": time.time(),
                               "expires_at": time.time() + 99, "status": "open", "setup_key": "k"})
    full = {"symbol": "EUR/JPY", "timeframe": "H1", "direction": "SHORT", "order_type": "SELL", "is_signal": False,
            "probability": 0.376, "entry": 174.56, "stop_loss": 174.85, "journal_id": aid, "paper_trade_id": tid,
            "take_profits": [{"name": "TP1", "price": 174.27, "p_reach": 0.52}], "deep": {"steps": ["a", "b"]},
            "explanation": {"summary": "Mixed-signal setup"}}
    out = agent.save_result(dict(full), "frxEURJPY", "chart")
    other = agent.save_result({**full, "symbol": "GBP/JPY", "order_type": "BUY LIMIT", "direction": "LONG",
                               "paper_trade_id": None}, "frxGBPJPY", "chat")
    items = server.saved_list(24)["items"]
    assert [i["symbol"] for i in items] == ["GBP/JPY", "EUR/JPY"]           # newest first
    assert items[1]["trade_status"] == "open" and items[1]["code"] == "frxEURJPY"
    again = server.saved_get(out["saved_id"])
    assert again["take_profits"] == full["take_profits"] and again["deep"] == full["deep"]   # exactly as shown
    assert again["code"] == "frxEURJPY" and again["trade_now"]["status"] == "open"
    db.update("trades", tid, {"status": "won", "r_multiple": 1.95})
    assert server.saved_get(out["saved_id"])["trade_now"]["r_multiple"] == 1.95             # shows what happened since
    # older than 24 h drops off the strip; older than 7 days is removed
    db.execute("UPDATE saved_analyses SET created_at=? WHERE id=?", (time.time() - 30 * 3600, other["saved_id"]))
    assert [i["symbol"] for i in server.saved_list(24)["items"]] == ["EUR/JPY"]
    db.execute("UPDATE saved_analyses SET created_at=? WHERE id=?", (time.time() - 8 * 86400, other["saved_id"]))
    db.prune()
    with pytest.raises(HTTPException):
        server.saved_get(other["saved_id"])
