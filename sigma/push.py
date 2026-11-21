"""Web push: opportunity alerts reach your phone's notification bar even when Sigma is closed.

- The server's VAPID key pair is created once and kept in the database.
- Each device that allows notifications registers a push subscription.
- Every alert is sent to every registered device; devices that no longer exist
  (the browser answers 404 or 410) are removed automatically.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time

from . import config, db

log = logging.getLogger("sigma.push")
_lock = threading.Lock()


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def keys() -> dict:
    """{'public': <base64url uncompressed P-256 point for the browser>, 'private_der': <base64url DER>}"""
    with _lock:
        k = db.kv_get("vapid")
        if k:
            return k
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        priv = ec.generate_private_key(ec.SECP256R1())
        pub = priv.public_key().public_bytes(serialization.Encoding.X962,
                                             serialization.PublicFormat.UncompressedPoint)
        der = priv.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
                                 serialization.NoEncryption())
        k = {"public": _b64(pub), "private_der": _b64(der), "created_at": time.time()}
        db.kv_set("vapid", k)
        return k


def public_key() -> str:
    return keys()["public"]


def subscribe(sub: dict, device: str = "") -> int:
    if not sub.get("endpoint") or not (sub.get("keys") or {}).get("p256dh") or not sub["keys"].get("auth"):
        raise ValueError("invalid push subscription")
    db.execute("DELETE FROM push_subs WHERE endpoint=?", (sub["endpoint"],))
    return db.insert("push_subs", {"endpoint": sub["endpoint"], "sub": sub, "device": device[:200],
                                   "created_at": time.time()})


def unsubscribe(endpoint: str) -> None:
    db.execute("DELETE FROM push_subs WHERE endpoint=?", (endpoint,))


def subscriptions() -> list[dict]:
    out = []
    for r in db.q("SELECT id, endpoint, sub, device, created_at FROM push_subs"):
        s = r["sub"]
        r["sub"] = json.loads(s) if isinstance(s, str) else s
        out.append(r)
    return out


def send_all(title: str, body: str, url: str = "/#markets", tag: str | None = None, sender=None) -> dict:
    """Send one notification to every registered device. Returns counts."""
    subs = subscriptions()
    if not subs:
        return {"sent": 0, "removed": 0, "failed": 0}
    if sender is None:
        from pywebpush import WebPushException, webpush

        def sender(info, data):
            try:
                webpush(subscription_info=info, data=data, vapid_private_key=keys()["private_der"],
                        vapid_claims={"sub": config.PUSH_CONTACT}, ttl=6 * 3600)
                return 201
            except WebPushException as ex:
                code = getattr(ex, "status_code", None)
                if code is None and getattr(ex, "response", None) is not None:
                    code = ex.response.status_code
                return code or 500
    payload = json.dumps({"title": title, "body": body[:240], "url": url, "tag": tag or title[:60],
                          "icon": "/icon-192.png"})
    sent = removed = failed = 0
    for s in subs:
        try:
            code = sender(s["sub"], payload)
        except Exception as e:
            log.warning("push failed: %s", e)
            code = 500
        if code in (200, 201, 202):
            sent += 1
        elif code in (404, 410):            # the device unsubscribed or the browser was reset
            unsubscribe(s["endpoint"])
            removed += 1
        else:
            failed += 1
    return {"sent": sent, "removed": removed, "failed": failed}


def notify(title: str, body: str, url: str = "/#markets", tag: str | None = None) -> None:
    """Fire-and-forget, so a slow push service never delays a scan."""
    def run():
        try:
            r = send_all(title, body, url, tag)
            log.info("push %s: %s", title, r)
        except Exception as e:
            log.warning("push error: %s", e)
    threading.Thread(target=run, daemon=True).start()


def plain(markdown: str) -> str:
    """Notification text without markdown symbols."""
    import re
    t = re.sub(r"[*_`#>]", "", markdown)
    return re.sub(r"\s+", " ", t).strip()
