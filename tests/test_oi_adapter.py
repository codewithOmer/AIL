"""Unit tests for the OI App Server adapter protocol handling.

These tests do NOT require the actual OI server.
"""

import json
import threading
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from integrations.open_interpreter.client import OpenInterpreterClient, _safe_params
from integrations.open_interpreter.config import MCPServerConfig, OIConfig
from integrations.open_interpreter.protocol import (
    OIError,
    OINotification,
    OIRequest,
    OIResponse,
    classify_message,
    decode_message,
    encode_message,
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
        from integrations.open_interpreter.config import OIConfig
        from integrations.open_interpreter.transport import (
            OITransport,
            OITransportError,
        )

        config = OIConfig(executable="nonexistent_binary_12345")
        transport = OITransport(config)
        with self.assertRaises(OITransportError):
            transport.start()


class TestClientUnit(unittest.TestCase):
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
