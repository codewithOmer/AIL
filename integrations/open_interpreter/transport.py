"""Low-level transport that manages an OI App Server subprocess and
sends/receives JSON-RPC messages over its stdio pipes.
"""

from __future__ import annotations

import io
import logging
import subprocess
import threading
from typing import Any

from integrations.open_interpreter.config import OIConfig
from integrations.open_interpreter.protocol import (
    OIError,
    OINotification,
    OIRequest,
    OIResponse,
    classify_message,
    decode_message,
    encode_message,
)

logger = logging.getLogger(__name__)


class OITransportError(Exception):
    pass


class OITransportClosed(OITransportError):
    pass


class OITransport:
    """Manages an OI App Server subprocess and provides synchronous
    request/response and notification delivery over stdio.
    """

    def __init__(self, config: OIConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._pending: dict[int, threading.Event] = {}
        self._results: dict[int, OIResponse | OIError] = {}
        self._notifications: list[OINotification] = []
        self._notification_event = threading.Event()
        self._lock = threading.Lock()
        self._closed = False
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._next_id = 1
        self._server_requests: list[OIRequest] = []
        self._server_request_event = threading.Event()

    def start(self) -> None:
        cmd = [self._config.executable, "app-server", "--listen", self._config.listen]
        cmd.extend(self._config.extra_args)
        logger.info("Starting OI App Server: %s", " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise OITransportError(
                f"OI executable not found: {self._config.executable!r}. "
                "Set AIL_OI_EXECUTABLE to the full path."
            ) from exc

        logger.info("OI App Server started (pid=%s)", self._process.pid)
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="oi-reader"
        )
        self._reader_thread.start()

        stderr_thread = threading.Thread(
            target=self._stderr_loop, daemon=True, name="oi-stderr"
        )
        stderr_thread.start()

    def _read_loop(self) -> None:
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        stdout = proc.stdout
        while True:
            try:
                line = stdout.readline()
                if not line:
                    logger.info("OI stdout closed (EOF)")
                    break
                decoded = line.decode("utf-8", errors="replace")
                msg = decode_message(decoded)
                if msg is None:
                    continue
                self._handle_message(msg)
            except Exception:
                logger.exception("Error reading OI stdout")
                break
        self._mark_closed()

    def _stderr_loop(self) -> None:
        proc = self._process
        if proc is None or proc.stderr is None:
            return
        while True:
            try:
                line = proc.stderr.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                with self._stderr_lock:
                    self._stderr_lines.append(decoded)
                logger.debug("OI stderr: %s", decoded)
            except Exception:
                break

    def _handle_message(self, msg: dict[str, Any]) -> None:
        classified = classify_message(msg)
        if classified is None:
            return
        if isinstance(classified, (OIResponse, OIError)):
            with self._lock:
                ev = self._pending.pop(classified.id, None)
                if ev is not None:
                    self._results[classified.id] = classified
            if ev:
                ev.set()
        elif isinstance(classified, OINotification):
            with self._lock:
                self._notifications.append(classified)
            self._notification_event.set()
            logger.info("Notification: %s", classified.method)
        elif isinstance(classified, OIRequest):
            with self._lock:
                self._server_requests.append(classified)
            self._server_request_event.set()
            logger.warning("Server request (unhandled): %s", classified.method)
            self._respond_to_server_request(classified)

    def _respond_to_server_request(self, req: OIRequest) -> None:
        error_response = {
            "id": req.id,
            "error": {
                "code": -32601,
                "message": f"Method not implemented in AIL adapter: {req.method}",
            },
        }
        try:
            self._send_raw(error_response)
        except OITransportClosed:
            logger.debug("Could not respond to server request; transport closed")

    def _send_raw(self, msg: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                raise OITransportClosed("Transport is closed")
            proc = self._process
        if proc is None or proc.stdin is None or getattr(proc.stdin, "closed", False):
            raise OITransportClosed("Process not running or stdin unavailable")
        if proc.poll() is not None:
            self._mark_closed()
            raise OITransportClosed(f"Process terminated with exit code {proc.poll()}")

        payload = encode_message(msg)
        logger.debug("Sending: %s", payload.decode("utf-8", errors="replace").strip())
        try:
            proc.stdin.write(payload)
            proc.stdin.flush()
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError) as exc:
            self._mark_closed()
            raise OITransportClosed(
                f"Failed to write to subprocess stdin: {exc}"
            ) from exc

    def send_raw(self, msg: dict[str, Any]) -> None:
        self._send_raw(msg)

    def send_request(self, method: str, params: dict[str, Any] | None = None) -> int:
        with self._lock:
            if self._closed:
                raise OITransportClosed("Transport is closed")
            req_id = self._next_id
            self._next_id += 1
            ev = threading.Event()
            self._pending[req_id] = ev
        msg: dict[str, Any] = {"id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        try:
            self._send_raw(msg)
        except Exception:
            with self._lock:
                self._pending.pop(req_id, None)
            raise
        return req_id

    def wait_for_response(
        self, req_id: int, timeout: float | None = None
    ) -> OIResponse | OIError:
        with self._lock:
            result = self._results.pop(req_id, None)
            if result is not None:
                return result
            ev = self._pending.get(req_id)
            if ev is None:
                if self._closed:
                    raise OITransportClosed("Request expired or transport closed")
                raise OITransportError(f"Request {req_id} not found")

        if not ev.wait(timeout=timeout):
            with self._lock:
                result = self._results.pop(req_id, None)
                if result is not None:
                    return result
                self._pending.pop(req_id, None)
            raise TimeoutError(f"Timeout waiting for response to request {req_id}")

        with self._lock:
            result = self._results.pop(req_id, None)
            if result is not None:
                return result
            self._pending.pop(req_id, None)
            if self._closed:
                raise OITransportClosed("Request expired or transport closed")
            raise OITransportError("Request expired or transport closed")

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        req_id = self.send_request(method, params)
        result = self.wait_for_response(req_id, timeout=timeout or self._config.request_timeout)
        if isinstance(result, OIError):
            raise OITransportError(
                f"OI RPC error {result.code}: {result.message}"
            )
        return result.result

    def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        msg: dict[str, Any] = {"method": method}
        if params is not None:
            msg["params"] = params
        self._send_raw(msg)

    def pop_notifications(self) -> list[OINotification]:
        with self._lock:
            notifs = self._notifications[:]
            self._notifications.clear()
        self._notification_event.clear()
        return notifs

    def pop_server_requests(self) -> list[OIRequest]:
        with self._lock:
            reqs = self._server_requests[:]
            self._server_requests.clear()
        self._server_request_event.clear()
        return reqs

    def wait_for_notification(
        self, method: str | None = None, timeout: float = 5.0
    ) -> OINotification | None:
        deadline = __import__("time").monotonic() + timeout
        while True:
            remaining = deadline - __import__("time").monotonic()
            if remaining <= 0:
                return None
            self._notification_event.wait(timeout=min(remaining, 0.1))
            with self._lock:
                for i, n in enumerate(self._notifications):
                    if method is None or n.method == method:
                        return self._notifications.pop(i)
            self._notification_event.clear()

    def get_stderr_lines(self) -> list[str]:
        with self._stderr_lock:
            return self._stderr_lines[:]

    @property
    def is_running(self) -> bool:
        with self._lock:
            if self._closed or self._process is None:
                return False
            return self._process.poll() is None

    def _mark_closed(self) -> None:
        with self._lock:
            self._closed = True
            events = list(self._pending.values())
            self._pending.clear()
        for ev in events:
            ev.set()
        self._notification_event.set()
        self._server_request_event.set()

    def close(self, timeout: float = 10.0) -> None:
        with self._lock:
            if self._closed and self._process is None:
                return
            proc = self._process
            self._process = None

        self._mark_closed()

        if proc is not None:
            logger.info("Shutting down OI App Server (pid=%s)", proc.pid)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream:
                    try:
                        stream.close()
                    except Exception:
                        logger.debug("Error closing stream", exc_info=True)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("OI process did not exit, terminating")
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error("OI process could not be killed")
        if (
            self._reader_thread
            and self._reader_thread.is_alive()
            and self._reader_thread is not threading.current_thread()
        ):
            self._reader_thread.join(timeout=3)
        logger.info("OI App Server shut down")
