import json
import os
import random
import tempfile
import time

from sigma import account, db, report_pdf


def make_trades(n=40, seed=3):
    rnd = random.Random(seed)
    now = time.time()
    syms = [("R_75", "H1"), ("BTCUSD", "M15"), ("EURUSD", "H1"), ("1HZ100V", "M15"), ("XAUUSD", "H1")]
    for i in range(n):
        sym, tf = syms[i % len(syms)]
        d = 1 if rnd.random() < 0.5 else -1
        aid = db.insert("analyses", {
            "created_at": now - 86400 * 3 + i * 600, "bar_time": now - 86400 * 3 + i * 600, "symbol": sym,
            "timeframe": tf, "source": "chat" if i % 7 == 0 else "live", "price": 100.0, "direction": d,
            "probability": 0.5 + rnd.random() * 0.2, "is_signal": 1, "entry": 100, "sl": 99, "tp": 102,
            "regime": rnd.choice(["trending", "ranging", "volatile"]), "setup_key": rnd.choice(["Trend pullback", "Breakout"]),
            "components": json.dumps({"trend": rnd.uniform(-1, 1), "momentum": rnd.uniform(-1, 1),
                                      "crowd": rnd.uniform(-1, 1), "news": 0.05}),
            "notes": json.dumps(["H1 trend up → pullback to EMA20", "RSI ≥ 70 – stretched", "Crowd 72% long – contrarian bearish"])})
        status = rnd.choice(["won", "lost", "lost", "expired", "missed"])
        r = {"won": 1.95, "lost": -1.03, "expired": rnd.uniform(-0.5, 0.8), "missed": None}[status]
        db.insert("trades", {
            "analysis_id": aid, "symbol": sym, "timeframe": tf, "source": "live", "direction": d, "entry": 100.0,
            "sl": 99.0, "tp": 102.0, "opened_at": now - 86400 * 3 + i * 600, "closed_at": now - 86400 * 2 + i * 600,
            "expires_at": now, "status": status, "setup_key": "x", "order_type": rnd.choice(["BUY", "SELL LIMIT", "BUY STOP"]),
            "demo": 1, "risk_amount": 100.0, "r_multiple": r, "exit_price": 101.0,
            "mfe_r": rnd.uniform(0, 2.2), "mae_r": -rnd.uniform(0, 1.1), "missed_r": 1.4 if status == "missed" else None,
            "lesson": "The short failed because the H1 trend was up – it sold into support “too early” → avoid shorts when the higher timeframe is bullish."})


def test_report_builds_with_all_sections(tmp_path):
    db.connect(os.path.join(tempfile.mkdtemp(), "r.db"))
    account.settings()
    make_trades()
    pdf = report_pdf.build()
    assert pdf[:5] == b"%PDF-" and len(pdf) > 8000
    (tmp_path / "r.pdf").write_bytes(pdf)
    from reportlab.lib.utils import ImageReader  # noqa: F401  (reportlab is installed)
    assert report_pdf.build(days=1)[:5] == b"%PDF-"          # a period with no trades still builds


def test_empty_report():
    db.connect(os.path.join(tempfile.mkdtemp(), "e.db"))
    assert report_pdf.build()[:5] == b"%PDF-"


def test_clean_text():
    assert report_pdf.clean("a → b ≥ c “q” <x> & y") == "a -&gt; b &gt;= c \"q\" &lt;x&gt; &amp; y"
