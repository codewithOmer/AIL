"""JSON-RPC 2.0 transport for the Open Interpreter App Server.

The OI App Server speaks a dialect of JSON-RPC 2.0 over stdio where:
- Messages are newline-delimited JSON
- Requests:  {"id": N, "method": "X", "params": {...}}
- Responses: {"id": N, "result": {...}}
- Errors:    {"id": N, "error": {"code": N, "message": "..."}}
- Notifications (server→client): {"method": "X", "params": {...}}
  (may include extra top-level fields like "emittedAtMs")
- Notifications (client→server): {"method": "X", "params": {...}}
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OIRequest:
    id: int
    method: str
    params: dict[str, Any] | None = None


@dataclass
class OIResponse:
    id: int
    result: Any = None


@dataclass
class OIError:
    id: int
    code: int = 0
    message: str = ""
    data: Any = None


@dataclass
class OINotification:
    method: str
    params: dict[str, Any] = field(default_factory=dict)


def encode_message(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj) + "\n").encode("utf-8")


def decode_message(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if not raw:
        return None
    return json.loads(raw)


def classify_message(msg: dict[str, Any]) -> OIRequest | OIResponse | OIError | OINotification | None:
    if "method" in msg and "id" in msg:
        return OIRequest(
            id=msg["id"],
            method=msg["method"],
            params=msg.get("params"),
        )
    if "method" in msg:
        return OINotification(
            method=msg["method"],
            params=msg.get("params") or {},
        )
    if "result" in msg and "id" in msg:
        return OIResponse(id=msg["id"], result=msg["result"])
    if "error" in msg and "id" in msg:
        err = msg["error"]
        return OIError(
            id=msg["id"],
            code=err.get("code", 0),
            message=err.get("message", ""),
            data=err.get("data"),
        )
    logger.warning("Unrecognized JSON-RPC message: %s", msg)
    return None
