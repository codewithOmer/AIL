"""High-level client for the Open Interpreter App Server.

This module provides ``OpenInterpreterClient``, which owns the full
lifecycle: start process → initialize → create thread → send turns →
collect streamed deltas → return final response → shut down.

AIL core interacts only through this class; raw JSON-RPC details stay
behind this boundary.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from integrations.open_interpreter.config import OIConfig
from integrations.open_interpreter.transport import (
    OINotification,
    OITransport,
    OITransportClosed,
    OITransportError,
)

logger = logging.getLogger(__name__)


class OIError(Exception):
    pass


class OIConnectionError(OIError):
    pass


class OITimeoutError(OIError):
    pass


class OIResponse:
    def __init__(self, text: str, thread_id: str, turn_id: str) -> None:
        self.text = text
        self.thread_id = thread_id
        self.turn_id = turn_id

    def __str__(self) -> str:
        return self.text


class OpenInterpreterClient:
    """High-level adapter for the OI App Server JSON-RPC protocol.

    Usage::

        client = OpenInterpreterClient(config)
        client.start()
        response = client.send_message("Hello!")
        print(response.text)
        client.shutdown()
    """

    def __init__(self, config: OIConfig | None = None) -> None:
        self._config = config or OIConfig()
        self._transport: OITransport | None = None
        self._thread_id: str | None = None

    def start(self) -> None:
        self._transport = OITransport(self._config)
        try:
            self._transport.start()
        except OITransportError as exc:
            raise OIConnectionError(str(exc)) from exc
        self._initialize()

    def _initialize(self) -> None:
        assert self._transport is not None
        logger.info("Sending initialize request")
        result = self._transport.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "AIL",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "experimentalApi": False,
                },
            },
            timeout=self._config.connect_timeout,
        )
        logger.info(
            "Initialize response: user_agent=%s codex_home=%s platform=%s/%s",
            result.get("userAgent", "?"),
            result.get("codexHome", "?"),
            result.get("platformFamily", "?"),
            result.get("platformOs", "?"),
        )
        self._transport.send_notification("initialized")
        logger.info("Initialization complete")

    def create_thread(self, cwd: str | None = None) -> str:
        assert self._transport is not None
        params: dict[str, Any] = {
            "approvalPolicy": self._config.approval_policy,
            "sandbox": self._config.sandbox,
        }
        if cwd:
            params["cwd"] = cwd
        elif self._config.cwd:
            params["cwd"] = self._config.cwd
        if self._config.model:
            params["model"] = self._config.model
        if self._config.model_provider:
            params["modelProvider"] = self._config.model_provider
        logger.info("Creating thread with params: %s", _safe_params(params))
        result = self._transport.request(
            "thread/start", params, timeout=self._config.connect_timeout
        )
        thread = result.get("thread", {})
        thread_id = thread.get("id", "")
        logger.info("Thread created: id=%s", thread_id)
        self._thread_id = thread_id
        return thread_id

    def send_message(
        self,
        message: str,
        thread_id: str | None = None,
        timeout: float | None = None,
    ) -> OIResponse:
        tid = thread_id or self._thread_id
        if tid is None:
            tid = self.create_thread()
        assert self._transport is not None
        req_timeout = timeout or self._config.request_timeout
        params: dict[str, Any] = {
            "threadId": tid,
            "input": [{"type": "text", "text": message}],
        }
        logger.info("Sending turn to thread %s", tid)
        result = self._transport.request(
            "turn/start", params, timeout=req_timeout
        )
        turn_id = result.get("turn", {}).get("id", "")
        logger.info("Turn started: turn_id=%s", turn_id)
        final = self._wait_for_turn_completion(
            tid, turn_id, timeout=req_timeout
        )
        return final

    def _wait_for_turn_completion(
        self,
        thread_id: str,
        turn_id: str,
        timeout: float = 120.0,
    ) -> OIResponse:
        assert self._transport is not None
        deadline = time.monotonic() + timeout
        text_parts: dict[str, str] = {}
        final_text = ""
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            notifs = self._transport.pop_notifications()
            for n in notifs:
                if n.method == "item/agentMessage/delta":
                    params = n.params
                    if params.get("turnId") == turn_id:
                        item_id = params.get("itemId", "")
                        delta = params.get("delta", "")
                        if item_id and delta:
                            text_parts[item_id] = text_parts.get(item_id, "") + delta
                            logger.debug(
                                "Delta [%s]: %s", item_id, delta[:80]
                            )
                elif n.method == "turn/completed":
                    params = n.params
                    if params.get("threadId") == thread_id:
                        turn = params.get("turn", {})
                        items = turn.get("items", [])
                        for item in items:
                            if item.get("type") == "AgentMessage":
                                final_text = item.get("text", "")
                        if not final_text and text_parts:
                            final_text = "".join(text_parts.values())
                        logger.info("Turn completed")
                        return OIResponse(
                            text=final_text or "".join(text_parts.values()),
                            thread_id=thread_id,
                            turn_id=turn_id,
                        )
                elif n.method == "error":
                    params = n.params
                    logger.error("Error notification: %s", params)
                    raise OIError(
                        f"Server error: {params.get('message', 'unknown')}"
                    )
            if not notifs:
                time.sleep(0.05)
        if text_parts:
            return OIResponse(
                text="".join(text_parts.values()),
                thread_id=thread_id,
                turn_id=turn_id,
            )
        raise OITimeoutError(
            f"Timeout after {timeout}s waiting for turn completion"
        )

    def shutdown(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        self._thread_id = None
        logger.info("Client shut down")

    def __enter__(self) -> OpenInterpreterClient:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()


def _safe_params(params: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for k, v in params.items():
        if "key" in k.lower() or "secret" in k.lower():
            safe[k] = "***"
        else:
            safe[k] = v
    return safe
