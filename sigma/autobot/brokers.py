"""Brokers the bot can trade through. Every broker offers the same small interface:

  balance() -> float                      account balance in the account currency
  open(symbol, direction, size, tp_price, sl_price) -> dict(ref, entry)
  close(ref, position) -> dict(exit, pnl)
  mark(position) -> dict(price, pnl, closed?, exit?)   current state of an open position
  multipliers(symbol) -> list[int] ; min_stake(symbol) -> float

PaperBroker  - the bot's own demo account (default). Fills at the live Deriv price, charges costs.
DerivBroker  - a real Deriv account (or a Deriv demo account) through Deriv's API, using
               Multipliers contracts with take profit and stop loss set on Deriv's side too,
               so positions stay protected even if the bot is offline.
"""
from __future__ import annotations

import json
import logging
import threading
import time

from .. import config, db
from ..data import market
from ..engine import COST_FRACTION
from ..symbols import resolve
from .sizing import DEFAULT_MULTIPLIERS, pnl

log = logging.getLogger("sigma.bot.broker")


class BrokerError(RuntimeError):
    pass


# --------------------------------------------------------------------------------- paper ---
class PaperBroker:
    name = "paper"
    live = False

    def __init__(self, start_balance: float = 100.0, currency: str = "USD"):
        acct = db.kv_get("bot_paper")
        if not acct:
            acct = {"balance": float(start_balance), "start_balance": float(start_balance),
                    "currency": currency, "created_at": time.time()}
            db.kv_set("bot_paper", acct)
        self.acct = acct

    def reset(self, start_balance: float, currency: str = "USD") -> None:
        self.acct = {"balance": float(start_balance), "start_balance": float(start_balance),
                     "currency": currency, "created_at": time.time()}
        db.kv_set("bot_paper", self.acct)

    def balance(self) -> float:
        return float((db.kv_get("bot_paper") or self.acct)["balance"])

    def _add(self, amount: float) -> None:
        a = db.kv_get("bot_paper") or self.acct
        a["balance"] = round(float(a["balance"]) + amount, 6)
        db.kv_set("bot_paper", a)
        self.acct = a

    def price(self, symbol: str) -> float:
        return market.last_price(symbol)

    def multipliers(self, symbol: str) -> list[int]:
        return DEFAULT_MULTIPLIERS.get(resolve(symbol).asset_class, [10, 20, 50, 100])

    def min_stake(self, symbol: str) -> float:
        return 1.0

    def cost_fraction(self, symbol: str) -> float:
        return COST_FRACTION.get(resolve(symbol).asset_class, 0.0003)

    def open(self, symbol: str, direction: int, size, tp_price: float, sl_price: float) -> dict:
        px = self.price(symbol)
        if self.balance() < size.stake:
            raise BrokerError("not enough balance for the stake")
        return {"ref": f"paper-{int(time.time() * 1000)}", "entry": px}

    def mark(self, pos: dict) -> dict:
        px = self.price(pos["symbol"])
        return {"price": px, "pnl": pnl(pos["exposure"], pos["direction"], pos["entry"], px, pos["cost"]),
                "closed": False}

    def close(self, ref: str, pos: dict) -> dict:
        px = self.price(pos["symbol"])
        p = pnl(pos["exposure"], pos["direction"], pos["entry"], px, pos["cost"])
        p = max(p, -pos["stake"])            # a multiplier position can never lose more than its stake
        self._add(p)
        return {"exit": px, "pnl": p}


# --------------------------------------------------------------------------------- Deriv ---
DERIV_LEGACY_WS = "wss://ws.derivws.com/websockets/v3?app_id={app_id}"
DERIV_OTP_URL = "https://api.derivws.com/trading/v1/options/accounts/{account}/otp"


class DerivBroker:
    """Deriv Multipliers through the API.

    Credentials (Render -> Environment, never in the code):
      DERIV_API_TOKEN   a personal access/API token with the 'trade' scope (Deriv -> account settings -> API token)
      DERIV_APP_ID      your registered app id (developers.deriv.com)
      DERIV_ACCOUNT_ID  optional: the options account id for Deriv's new API; without it the legacy
                        API is used (authorize with the token). Use a DEMO account first.
    """
    name = "deriv"
    live = True

    def __init__(self):
        if not config.DERIV_API_TOKEN:
            raise BrokerError("DERIV_API_TOKEN is not set")
        self._lock = threading.Lock()
        self._ws = None
        self._req = 0
        self._mult_cache: dict[str, list[int]] = {}
        self.currency = "USD"

    # ---- connection -----------------------------------------------------------------------
    def _connect(self):
        import websocket
        if config.DERIV_ACCOUNT_ID:     # Deriv's new API: one-time-password websocket URL
            import requests
            r = requests.post(DERIV_OTP_URL.format(account=config.DERIV_ACCOUNT_ID), timeout=15,
                              headers={"Authorization": f"Bearer {config.DERIV_API_TOKEN}",
                                       "Deriv-App-ID": str(config.DERIV_APP_ID)})
            r.raise_for_status()
            url = r.json()["data"]["url"]
            ws = websocket.create_connection(url, timeout=20)
        else:                           # legacy API: authorize with the token
            ws = websocket.create_connection(DERIV_LEGACY_WS.format(app_id=config.DERIV_APP_ID), timeout=20)
            ws.send(json.dumps({"authorize": config.DERIV_API_TOKEN, "req_id": 1}))
            msg = json.loads(ws.recv())
            if "error" in msg:
                raise BrokerError("Deriv login failed: " + msg["error"].get("message", ""))
            self.currency = msg.get("authorize", {}).get("currency") or "USD"
        self._ws = ws

    def _call(self, payload: dict, timeout: float = 20) -> dict:
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._ws is None:
                        self._connect()
                    self._req += 1
                    rid = self._req
                    self._ws.send(json.dumps({**payload, "req_id": rid}))
                    end = time.time() + timeout
                    while time.time() < end:
                        msg = json.loads(self._ws.recv())
                        if msg.get("req_id") == rid:
                            if "error" in msg:
                                raise BrokerError(msg["error"].get("message", "Deriv error"))
                            return msg
                    raise BrokerError("Deriv did not answer in time")
                except BrokerError:
                    raise
                except Exception as e:          # dropped connection: reconnect once
                    log.warning("Deriv connection problem (%s), reconnecting", e)
                    try:
                        self._ws.close()
                    except Exception:
                        pass
                    self._ws = None
                    if attempt == 2:
                        raise BrokerError(f"Deriv unreachable: {e}")
        raise BrokerError("Deriv call failed")

    # ---- account & contract info ----------------------------------------------------------
    def balance(self) -> float:
        m = self._call({"balance": 1})
        b = m.get("balance", {})
        self.currency = b.get("currency") or self.currency
        return float(b.get("balance", 0))

    def price(self, symbol: str) -> float:
        return market.last_price(symbol)

    def multipliers(self, symbol: str) -> list[int]:
        code = resolve(symbol).deriv or symbol
        if code not in self._mult_cache:
            try:
                m = self._call({"contracts_for": code, "currency": self.currency})
                ms = set()
                for c in m.get("contracts_for", {}).get("available", []):
                    if c.get("contract_type") in ("MULTUP", "MULTDOWN"):
                        ms.update(int(x) for x in c.get("multiplier_range", []))
                self._mult_cache[code] = sorted(ms) or DEFAULT_MULTIPLIERS.get(resolve(symbol).asset_class, [10, 20])
            except BrokerError as e:
                log.warning("multipliers for %s: %s", code, e)
                self._mult_cache[code] = DEFAULT_MULTIPLIERS.get(resolve(symbol).asset_class, [10, 20])
        return self._mult_cache[code]

    def min_stake(self, symbol: str) -> float:
        return float(config.DERIV_MIN_STAKE)

    def cost_fraction(self, symbol: str) -> float:
        return COST_FRACTION.get(resolve(symbol).asset_class, 0.0003)

    # ---- trading --------------------------------------------------------------------------
    def open(self, symbol: str, direction: int, size, tp_price: float, sl_price: float) -> dict:
        code = resolve(symbol).deriv or symbol
        prop = self._call({
            "proposal": 1, "amount": size.stake, "basis": "stake", "currency": self.currency,
            "contract_type": "MULTUP" if direction > 0 else "MULTDOWN", "symbol": code,
            "multiplier": size.multiplier,
            # protective orders live on Deriv's side as well (in money), in case the bot is offline
            "limit_order": {"take_profit": round(max(size.tp_usd, 0.1), 2),
                            "stop_loss": round(max(size.sl_usd, 0.1), 2)},
        })
        pid = prop["proposal"]["id"]
        buy = self._call({"buy": pid, "price": size.stake})["buy"]
        spot = None
        try:
            poc = self._call({"proposal_open_contract": 1, "contract_id": buy["contract_id"]})
            spot = float(poc["proposal_open_contract"].get("entry_spot") or 0) or None
        except BrokerError:
            pass
        return {"ref": str(buy["contract_id"]), "entry": spot or self.price(symbol)}

    def mark(self, pos: dict) -> dict:
        poc = self._call({"proposal_open_contract": 1, "contract_id": int(pos["broker_ref"])})["proposal_open_contract"]
        closed = bool(poc.get("is_sold")) or poc.get("status") in ("sold", "won", "lost")
        return {"price": float(poc.get("current_spot") or 0), "pnl": float(poc.get("profit") or 0),
                "closed": closed, "exit": float(poc.get("exit_tick") or poc.get("current_spot") or 0)}

    def close(self, ref: str, pos: dict) -> dict:
        sold = self._call({"sell": int(ref), "price": 0})["sell"]
        st = self.mark(pos)
        return {"exit": st["exit"] or st["price"], "pnl": float(st["pnl"]),
                "sold_for": float(sold.get("sold_for", 0))}


def get(name: str):
    if name == "deriv":
        return DerivBroker()
    s = db.kv_get("bot_settings") or {}
    return PaperBroker(float(s.get("paper_start_balance", 100.0)), s.get("currency", "USD"))
