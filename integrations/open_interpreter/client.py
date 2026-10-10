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
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from integrations.open_interpreter.config import OIConfig
from integrations.open_interpreter.transport import (
    OINotification,
    OITransport,
    OITransportClosed,
    OITransportError,
)
from interfaces.image import LocalImage

logger = logging.getLogger(__name__)


class OIError(Exception):
    pass


class OIConnectionError(OIError):
    pass


class OITimeoutError(OIError):
    pass


class OITransportFailure(OIConnectionError):
    """The transport could not reliably complete an active turn."""

    pass


class OIResponse:
    def __init__(
        self,
        text: str,
        thread_id: str,
        turn_id: str,
        items: list[dict[str, Any]] | None = None,
    ) -> None:
        self.text = text
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.items = items if items is not None else []

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
        self._turn_lock = threading.Lock()

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
        mcp_config = {
            f"mcp_servers.{server.name}.{key}": value
            for server in self._config.mcp_servers
            if server.enabled
            for key, value in (
                ("command", server.command),
                ("args", server.args),
                ("env", server.env),
            )
        }
        if mcp_config:
            params["config"] = mcp_config
        logger.info("Creating thread with params: %s", _safe_params(params))
        result = self._transport.request(
            "thread/start", params, timeout=self._config.connect_timeout
        )
        thread = result.get("thread", {})
        thread_id = thread.get("id", "")
        logger.info("Thread created: id=%s", thread_id)
        self._thread_id = thread_id
        return thread_id

    def mcp_status(self) -> Any:
        assert self._transport is not None
        return self._transport.request(
            "mcpServerStatus/list", {}, timeout=self._config.request_timeout
        )

    def send_message(
        self,
        message: str,
        thread_id: str | None = None,
        timeout: float | None = None,
        images: Sequence[LocalImage | str | Path] | None = None,
    ) -> OIResponse:
        """Send one turn; the App Server adapter intentionally serializes turns."""
        with self._turn_lock:
            return self._send_message(message, thread_id, timeout, images)

    def _send_message(
        self,
        message: str,
        thread_id: str | None = None,
        timeout: float | None = None,
        images: Sequence[LocalImage | str | Path] | None = None,
    ) -> OIResponse:
        tid = thread_id or self._thread_id
        if tid is None:
            tid = self.create_thread()
        assert self._transport is not None
        req_timeout = timeout or self._config.request_timeout
        inputs: list[dict[str, Any]] = []
        for image in images or ():
            local_image = (
                image if isinstance(image, LocalImage) else LocalImage.from_path(image)
            )
            image_input: dict[str, Any] = {
                "type": "localImage",
                "path": str(local_image.validated_path()),
            }
            if local_image.detail is not None:
                image_input["detail"] = local_image.detail
            inputs.append(image_input)
        inputs.append({"type": "text", "text": message})
        params: dict[str, Any] = {"threadId": tid, "input": inputs}
        logger.info("Sending turn to thread %s", tid)
        try:
            result = self._transport.request(
                "turn/start", params, timeout=req_timeout
            )
        except OITransportError as exc:
            raise OITransportFailure(f"Failed to start turn: {exc}") from exc
        turn_id = result.get("turn", {}).get("id", "")
        if not turn_id:
            raise OITransportFailure("Turn start response did not include a turn id")
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
                    turn = params.get("turn")
                    if (
                        params.get("threadId") == thread_id
                        and isinstance(turn, dict)
                        and turn.get("id") == turn_id
                    ):
                        items = turn.get("items", [])
                        final_text = "".join(
                            item.get("text", "")
                            for item in items
                            if item.get("type") == "AgentMessage"
                        )
                        logger.info("Turn completed: turn_id=%s", turn_id)
                        return OIResponse(
                            text=final_text or "".join(text_parts.values()),
                            thread_id=thread_id,
                            turn_id=turn_id,
                            items=items,
                        )
                elif n.method == "error":
                    params = n.params
                    if (
                        params.get("threadId") == thread_id
                        and params.get("turnId") == turn_id
                    ):
                        message, details = _extract_error(params)
                        logger.error("Error notification for turn %s: %s", turn_id, message)
                        detail_text = f" ({details})" if details else ""
                        raise OIError(f"Server error: {message}{detail_text}")
            if not notifs:
                if not self._transport.is_running:
                    raise OITransportFailure(
                        f"Transport closed while waiting for turn {turn_id}"
                    )
                time.sleep(min(0.05, remaining))
        raise OITimeoutError(
            f"Timeout after {timeout}s waiting for turn {turn_id} completion"
        )

    def get_stderr(self) -> list[str]:
        if self._transport is None:
            return []
        return self._transport.get_stderr_lines()

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
        if any(
            marker in k.lower()
            for marker in ("key", "secret", "token", "password", "credential")
        ):
            safe[k] = "***"
        elif isinstance(v, dict):
            safe[k] = _safe_params(v)
        else:
            safe[k] = v
    return safe


def _extract_error(params: dict[str, Any]) -> tuple[str, str]:
    """Extract the App Server error object without exposing sensitive fields."""
    error = params.get("error")
    if not isinstance(error, dict):
        error = {}
    message = error.get("message") or params.get("message") or "unknown server error"
    if not isinstance(message, str):
        message = str(message)
    safe_error = _safe_params(
        {
            key: value
            for key, value in error.items()
            if key not in {"message", "stack", "traceback"}
        }
    )
    details = ", ".join(f"{key}={value}" for key, value in safe_error.items())
    return message, details
