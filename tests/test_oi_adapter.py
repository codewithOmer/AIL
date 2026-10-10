"""Unit tests for the OI App Server adapter protocol handling.

These tests do NOT require the actual OI server.
"""

import json
import threading
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from integrations.open_interpreter.client import (
    OIError as ClientError,
    OITransportFailure,
    OITimeoutError,
    OIResponse as ClientResponse,
    OpenInterpreterClient,
    _safe_params,
)
from integrations.open_interpreter.config import MCPServerConfig, OIConfig
from interfaces.image import LocalImage
from integrations.open_interpreter.protocol import (
    OIError,
    OINotification,
    OIRequest,
    OIResponse,
    classify_message,
    decode_message,
    encode_message,
)
from integrations.open_interpreter.transport import (
    OITransport,
    OITransportClosed,
    OITransportError,
)


class TestProtocol(unittest.TestCase):
    def test_encode_request(self):
        msg = {"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "test"}}}
        encoded = encode_message(msg)
        self.assertTrue(encoded.endswith(b"\n"))
        parsed = json.loads(encoded)
        self.assertEqual(parsed["id"], 1)
        self.assertEqual(parsed["method"], "initialize")

    def test_encode_notification(self):
        msg = {"method": "initialized"}
        encoded = encode_message(msg)
        parsed = json.loads(encoded)
        self.assertEqual(parsed["method"], "initialized")
        self.assertNotIn("id", parsed)

    def test_decode_request(self):
        raw = '{"id": 1, "method": "initialize", "params": {}}\n'
        msg = decode_message(raw)
        self.assertIsNotNone(msg)
        self.assertEqual(msg["method"], "initialize")
        self.assertEqual(msg["id"], 1)

    def test_decode_response(self):
        raw = '{"id": 1, "result": {"userAgent": "test"}}\n'
        msg = decode_message(raw)
        self.assertIsNotNone(msg)
        self.assertIn("result", msg)

    def test_decode_error(self):
        raw = '{"id": 1, "error": {"code": -32601, "message": "not found"}}\n'
        msg = decode_message(raw)
        self.assertIsNotNone(msg)
        self.assertIn("error", msg)

    def test_decode_empty(self):
        msg = decode_message("")
        self.assertIsNone(msg)
        msg = decode_message("\n")
        self.assertIsNone(msg)

    def test_classify_request(self):
        msg = {"id": 1, "method": "initialize", "params": {}}
        result = classify_message(msg)
        self.assertIsInstance(result, OIRequest)
        self.assertEqual(result.id, 1)
        self.assertEqual(result.method, "initialize")

    def test_classify_response(self):
        msg = {"id": 1, "result": {"userAgent": "test"}}
        result = classify_message(msg)
        self.assertIsInstance(result, OIResponse)
        self.assertEqual(result.id, 1)
        self.assertEqual(result.result["userAgent"], "test")

    def test_classify_error(self):
        msg = {"id": 1, "error": {"code": -32601, "message": "not found"}}
        result = classify_message(msg)
        self.assertIsInstance(result, OIError)
        self.assertEqual(result.code, -32601)

    def test_classify_notification(self):
        msg = {"method": "turn/completed", "params": {"threadId": "abc"}}
        result = classify_message(msg)
        self.assertIsInstance(result, OINotification)
        self.assertEqual(result.method, "turn/completed")

    def test_classify_notification_with_extra_fields(self):
        msg = {"method": "configWarning", "params": {"summary": "test"}, "emittedAtMs": 1234}
        result = classify_message(msg)
        self.assertIsInstance(result, OINotification)
        self.assertEqual(result.method, "configWarning")

    def test_request_id_increments(self):
        from integrations.open_interpreter.protocol import OIRequest
        msg1 = {"id": 1, "method": "a"}
        msg2 = {"id": 2, "method": "b"}
        r1 = classify_message(msg1)
        r2 = classify_message(msg2)
        self.assertIsInstance(r1, OIRequest)
        self.assertIsInstance(r2, OIRequest)
        self.assertEqual(r1.id + 1, r2.id)


class TestTransportUnit(unittest.TestCase):
    def test_start_raises_on_missing_executable(self):
        config = OIConfig(executable="nonexistent_binary_12345")
        transport = OITransport(config)
        with self.assertRaises(OITransportError):
            transport.start()

    def _make_mock_transport(self) -> tuple[OITransport, MagicMock, BytesIO]:
        transport = OITransport(OIConfig())
        proc = MagicMock()
        stdin = BytesIO()
        proc.stdin = stdin
        proc.stdout = None
        proc.stderr = None
        proc.poll.return_value = None
        transport._process = proc
        return transport, proc, stdin

    def test_send_request_after_transport_closed_raises_transport_closed(self):
        transport, _, _ = self._make_mock_transport()
        transport.close()
        self.assertTrue(transport._closed)
        with self.assertRaises(OITransportClosed) as cm:
            transport.send_request("initialize")
        self.assertIn("closed", str(cm.exception).lower())

        with self.assertRaises(OITransportClosed):
            transport.send_raw({"id": 1, "method": "initialize"})

    def test_send_when_stdin_unavailable_or_closed_raises_transport_closed(self):
        # 1. Process is None
        transport = OITransport(OIConfig())
        transport._process = None
        with self.assertRaises(OITransportClosed) as cm1:
            transport.send_request("initialize")
        self.assertIn("Process not running or stdin unavailable", str(cm1.exception))

        # 2. stdin is None
        transport2, proc2, _ = self._make_mock_transport()
        proc2.stdin = None
        with self.assertRaises(OITransportClosed) as cm2:
            transport2.send_request("initialize")
        self.assertIn("Process not running or stdin unavailable", str(cm2.exception))

        # 3. stdin is closed
        transport3, proc3, _ = self._make_mock_transport()
        mock_stdin = MagicMock()
        mock_stdin.closed = True
        proc3.stdin = mock_stdin
        with self.assertRaises(OITransportClosed) as cm3:
            transport3.send_request("initialize")
        self.assertIn("Process not running or stdin unavailable", str(cm3.exception))

    def test_write_failing_due_to_pipe_closure_translates_to_transport_closed(self):
        transport, proc, _ = self._make_mock_transport()
        mock_stdin = MagicMock()
        mock_stdin.closed = False
        broken_err = BrokenPipeError("The pipe was broken")
        mock_stdin.write.side_effect = broken_err
        proc.stdin = mock_stdin

        with self.assertRaises(OITransportClosed) as cm:
            transport.send_request("initialize")
        self.assertIs(cm.exception.__cause__, broken_err)
        self.assertIn("Failed to write to subprocess stdin", str(cm.exception))
        self.assertTrue(transport._closed)

        # Flush failure with OSError
        transport2, proc2, _ = self._make_mock_transport()
        mock_stdin2 = MagicMock()
        mock_stdin2.closed = False
        os_err = OSError(22, "Invalid argument")
        mock_stdin2.flush.side_effect = os_err
        proc2.stdin = mock_stdin2

        with self.assertRaises(OITransportClosed) as cm2:
            transport2.send_raw({"method": "notify"})
        self.assertIs(cm2.exception.__cause__, os_err)
        self.assertIn("Failed to write to subprocess stdin", str(cm2.exception))

    def test_transport_closure_while_request_pending_wakes_waiter_without_keyerror(self):
        transport, _, _ = self._make_mock_transport()
        req_id = transport.send_request("initialize")

        def close_soon():
            import time
            time.sleep(0.05)
            transport.close()

        closer_thread = threading.Thread(target=close_soon)
        closer_thread.start()

        with self.assertRaises(OITransportClosed) as cm:
            transport.wait_for_response(req_id, timeout=2.0)

        closer_thread.join()
        self.assertNotIsInstance(cm.exception, KeyError)
        self.assertIn("closed", str(cm.exception).lower())

    def test_wait_for_unknown_or_expired_request_does_not_raise_keyerror(self):
        # Open transport:
        transport, _, _ = self._make_mock_transport()
        with self.assertRaises(OITransportError) as cm:
            transport.wait_for_response(9999, timeout=0.1)
        self.assertNotIsInstance(cm.exception, KeyError)
        self.assertIn("Request 9999 not found", str(cm.exception))

        # Closed transport:
        transport.close()
        with self.assertRaises(OITransportClosed) as cm2:
            transport.wait_for_response(9999, timeout=0.1)
        self.assertNotIsInstance(cm2.exception, KeyError)
        self.assertIn("closed", str(cm2.exception).lower())

    def test_successful_response_reaches_correct_waiter(self):
        transport, _, _ = self._make_mock_transport()
        req_id = transport.send_request("initialize")

        def deliver_response():
            import time
            time.sleep(0.05)
            transport._handle_message({"id": req_id, "result": {"userAgent": "oi-test"}})

        deliverer = threading.Thread(target=deliver_response)
        deliverer.start()

        resp = transport.wait_for_response(req_id, timeout=2.0)
        deliverer.join()

        self.assertIsInstance(resp, OIResponse)
        self.assertEqual(resp.id, req_id)
        self.assertEqual(resp.result, {"userAgent": "oi-test"})

    def test_response_received_before_wait_or_closure_is_delivered(self):
        transport, _, _ = self._make_mock_transport()
        req_id = transport.send_request("initialize")
        transport._handle_message({"id": req_id, "result": {"ready": True}})

        # Response already present in _results
        resp = transport.wait_for_response(req_id, timeout=0.1)
        self.assertEqual(resp.result, {"ready": True})

        # Response arrived, then transport closed before wait_for_response
        req_id2 = transport.send_request("thread/start")
        transport._handle_message({"id": req_id2, "result": {"thread": {"id": "t-1"}}})
        transport.close()
        resp2 = transport.wait_for_response(req_id2, timeout=0.1)
        self.assertEqual(resp2.result, {"thread": {"id": "t-1"}})

    def test_existing_timeout_behavior_remains_correct(self):
        transport, _, _ = self._make_mock_transport()
        req_id = transport.send_request("initialize")

        with self.assertRaises(TimeoutError) as cm:
            transport.wait_for_response(req_id, timeout=0.05)

        self.assertIn(f"Timeout waiting for response to request {req_id}", str(cm.exception))
        # Ensure timeout did not close the transport
        self.assertFalse(transport._closed)

    def test_repeated_close_and_cleanup_is_idempotent(self):
        transport, proc, _ = self._make_mock_transport()
        proc.stdout = BytesIO()
        proc.stderr = BytesIO()

        transport.close()
        self.assertTrue(transport._closed)
        self.assertFalse(transport.is_running)

        # Repeated close calls must not raise
        transport.close()
        transport._mark_closed()
        self.assertTrue(transport._closed)
        self.assertFalse(transport.is_running)

    def test_late_response_after_timeout_is_discarded_without_storing(self):
        transport, _, _ = self._make_mock_transport()
        try:
            req_id = transport.send_request("initialize")
            with self.assertRaises(TimeoutError):
                transport.wait_for_response(req_id, timeout=0)

            # Simulate a valid late response through the normal response-dispatch path
            transport._handle_message({"id": req_id, "result": {"userAgent": "oi-late"}})

            self.assertNotIn(req_id, transport._pending)
            self.assertNotIn(req_id, transport._results)

            # Also simulate a valid late error response
            transport._handle_message({"id": req_id, "error": {"code": -32600, "message": "late error"}})

            self.assertNotIn(req_id, transport._pending)
            self.assertNotIn(req_id, transport._results)
        finally:
            transport.close()

    def test_close_kills_unresponsive_process_without_deadlock(self):
        import subprocess
        import sys
        import time

        # Start a local python child that ignores stdin EOF and sleeps to force the kill path
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", "import time; time.sleep(30)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        transport = OITransport(OIConfig())
        transport._process = proc

        # Start stdout and stderr reader threads to reproduce buffered-reader lock contention
        t_reader = threading.Thread(
            target=transport._read_loop, daemon=True, name="test-oi-reader"
        )
        t_reader.start()
        transport._reader_thread = t_reader

        t_stderr = threading.Thread(
            target=transport._stderr_loop, daemon=True, name="test-oi-stderr"
        )
        t_stderr.start()
        transport._stderr_thread = t_stderr

        # Allow threads to enter readline() on the empty pipes
        time.sleep(0.1)

        closed_event = threading.Event()
        errors: list[Exception] = []

        def do_close():
            try:
                transport.close(timeout=0.5)
            except Exception as exc:
                errors.append(exc)
            finally:
                closed_event.set()

        close_thread = threading.Thread(target=do_close, daemon=True)
        close_thread.start()

        try:
            finished = closed_event.wait(timeout=5.0)
            self.assertTrue(finished, "transport.close() deadlocked on unresponsive process")
            self.assertEqual(errors, [], f"Unexpected error during close(): {errors}")
            self.assertIsNotNone(proc.poll(), "Child process was not reaped")
        finally:
            if proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
            close_thread.join(timeout=2)



class TestClientUnit(unittest.TestCase):
    def _completion_transport(self, notifications, running=True):
        transport = MagicMock()
        transport.pop_notifications.side_effect = [notifications, []]
        transport.is_running = running
        return transport

    def _client_waiter(self, transport):
        client = OpenInterpreterClient(OIConfig())
        client._transport = transport
        return client

    def test_partial_deltas_without_completion_fail(self):
        notification = OINotification(
            "item/agentMessage/delta",
            {"threadId": "thread-1", "turnId": "turn-1", "itemId": "item-1", "delta": "partial"},
        )
        transport = self._completion_transport([notification], running=False)
        with self.assertRaises(OITransportFailure):
            self._client_waiter(transport)._wait_for_turn_completion(
                "thread-1", "turn-1", timeout=1
            )

    def test_active_turn_completion_returns_accumulated_text(self):
        notification = OINotification(
            "turn/completed",
            {
                "threadId": "thread-1",
                "turn": {"id": "turn-1", "items": []},
            },
        )
        transport = self._completion_transport([notification])
        response = self._client_waiter(transport)._wait_for_turn_completion(
            "thread-1", "turn-1", timeout=1
        )
        self.assertEqual(response.text, "")
        self.assertEqual(response.turn_id, "turn-1")

    def test_stale_completion_does_not_complete_active_turn(self):
        stale = OINotification(
            "turn/completed",
            {"threadId": "thread-1", "turn": {"id": "old-turn", "items": []}},
        )
        transport = self._completion_transport([stale], running=False)
        with self.assertRaises(OITransportFailure):
            self._client_waiter(transport)._wait_for_turn_completion(
                "thread-1", "turn-1", timeout=1
            )

    def test_deadline_expiry_does_not_return_partial_text(self):
        transport = self._completion_transport([], running=True)
        with self.assertRaises(OITimeoutError):
            self._client_waiter(transport)._wait_for_turn_completion(
                "thread-1", "turn-1", timeout=0
            )

    def test_nested_server_error_preserves_message_and_details(self):
        error = OINotification(
            "error",
            {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "error": {
                    "message": "quota exhausted",
                    "code": "quota_exceeded",
                    "token": "do-not-leak",
                },
            },
        )
        transport = self._completion_transport([error])
        with self.assertRaises(ClientError) as cm:
            self._client_waiter(transport)._wait_for_turn_completion(
                "thread-1", "turn-1", timeout=1
            )
        self.assertIn("quota exhausted", str(cm.exception))
        self.assertIn("code=quota_exceeded", str(cm.exception))
        self.assertNotIn("do-not-leak", str(cm.exception))

    def test_mismatched_error_is_ignored(self):
        error = OINotification(
            "error",
            {
                "threadId": "thread-1",
                "turnId": "old-turn",
                "error": {"message": "old failure"},
            },
        )
        transport = self._completion_transport([error], running=False)
        with self.assertRaises(OITransportFailure):
            self._client_waiter(transport)._wait_for_turn_completion(
                "thread-1", "turn-1", timeout=1
            )

    def _send_message_params(self, images=None) -> dict:
        transport = MagicMock()
        transport.request.return_value = {"turn": {"id": "turn-1"}}
        client = OpenInterpreterClient(OIConfig())
        client._transport = transport
        client._thread_id = "thread-1"
        completed = ClientResponse("done", "thread-1", "turn-1")
        with patch.object(client, "_wait_for_turn_completion", return_value=completed):
            result = client.send_message("describe", images=images)
        self.assertIs(result, completed)
        return transport.request.call_args.args[1]

    def test_text_only_turn_payload_remains_unchanged(self) -> None:
        self.assertEqual(
            self._send_message_params(),
            {
                "threadId": "thread-1",
                "input": [{"type": "text", "text": "describe"}],
            },
        )

    def test_local_image_serializes_to_current_oi_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"not decoded by the client")

            params = self._send_message_params(
                [LocalImage.from_path(path, detail="high")]
            )

        self.assertEqual(
            params["input"],
            [
                {
                    "type": "localImage",
                    "path": str(path.resolve()),
                    "detail": "high",
                },
                {"type": "text", "text": "describe"},
            ],
        )

    def test_multiple_local_images_are_serialized_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "one.png", Path(directory) / "two.png"]
            for path in paths:
                path.write_bytes(b"image")

            params = self._send_message_params([str(path) for path in paths])

        self.assertEqual(
            params["input"],
            [
                {"type": "localImage", "path": str(path.resolve())}
                for path in paths
            ]
            + [{"type": "text", "text": "describe"}],
        )

    def test_missing_or_non_file_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.png"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                self._send_message_params([missing])
            with self.assertRaisesRegex(ValueError, "not a file"):
                self._send_message_params([Path(directory)])

    def test_config_defaults(self):
        config = OIConfig()
        self.assertEqual(config.approval_policy, "never")
        self.assertEqual(config.sandbox, "read-only")
        self.assertIsInstance(config.extra_args, list)
        self.assertEqual(config.mcp_servers, [])

    def _thread_request_params(self, config: OIConfig) -> dict:
        transport = MagicMock()
        transport.request.return_value = {"thread": {"id": "thread-1"}}
        client = OpenInterpreterClient(config)
        client._transport = transport
        client.create_thread()
        transport.request.assert_called_once()
        method, params = transport.request.call_args.args[:2]
        self.assertEqual(method, "thread/start")
        return params

    def test_create_thread_omits_empty_mcp_config(self):
        params = self._thread_request_params(OIConfig())

        self.assertNotIn("config", params)

    def test_create_thread_renders_enabled_mcp_server(self):
        server = MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "server-filesystem", "C:/workspace"],
            env={"MCP_MODE": "test"},
        )
        params = self._thread_request_params(OIConfig(mcp_servers=[server]))

        self.assertEqual(
            params["config"],
            {
                "mcp_servers.filesystem.command": "npx",
                "mcp_servers.filesystem.args": ["-y", "server-filesystem", "C:/workspace"],
                "mcp_servers.filesystem.env": {"MCP_MODE": "test"},
            },
        )

    def test_create_thread_renders_playwright_mcp_server(self):
        server = MCPServerConfig(
            name="browser",
            command="npx",
            args=[
                "-y",
                "@playwright/mcp@latest",
                "--headless",
                "--isolated",
                "--no-webmcp",
            ],
            env={},
        )
        params = self._thread_request_params(OIConfig(mcp_servers=[server]))

        self.assertEqual(
            params["config"],
            {
                "mcp_servers.browser.command": "npx",
                "mcp_servers.browser.args": [
                    "-y",
                    "@playwright/mcp@latest",
                    "--headless",
                    "--isolated",
                    "--no-webmcp",
                ],
                "mcp_servers.browser.env": {},
            },
        )

    def test_create_thread_renders_multiple_enabled_mcp_servers(self):
        servers = [
            MCPServerConfig(name="one", command="one-server"),
            MCPServerConfig(name="two", command="two-server", args=["--port", "1"]),
        ]
        params = self._thread_request_params(OIConfig(mcp_servers=servers))

        self.assertEqual(
            params["config"],
            {
                "mcp_servers.one.command": "one-server",
                "mcp_servers.one.args": [],
                "mcp_servers.one.env": {},
                "mcp_servers.two.command": "two-server",
                "mcp_servers.two.args": ["--port", "1"],
                "mcp_servers.two.env": {},
            },
        )

    def test_create_thread_skips_disabled_mcp_server(self):
        server = MCPServerConfig(
            name="disabled", command="unused", enabled=False
        )
        params = self._thread_request_params(OIConfig(mcp_servers=[server]))

        self.assertNotIn("config", params)

    def test_mcp_status_uses_generic_transport(self):
        transport = MagicMock()
        transport.request.return_value = {"servers": []}
        client = OpenInterpreterClient(OIConfig())
        client._transport = transport

        result = client.mcp_status()

        self.assertEqual(result, {"servers": []})
        transport.request.assert_called_once_with(
            "mcpServerStatus/list", {}, timeout=client._config.request_timeout
        )

    def test_safe_params_redacts_nested_secrets(self):
        safe = _safe_params(
            {
                "config": {
                    "mcp_servers.linear.env": {
                        "LINEAR_API_KEY": "secret-value",
                        "LINEAR_TEAM": "engineering",
                    }
                }
            }
        )

        self.assertEqual(
            safe,
            {
                "config": {
                    "mcp_servers.linear.env": {
                        "LINEAR_API_KEY": "***",
                        "LINEAR_TEAM": "engineering",
                    }
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
