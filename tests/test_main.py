"""Focused tests for AIL's interactive entry points."""

from __future__ import annotations

import unittest
from io import StringIO
from unittest.mock import AsyncMock, patch

from core.application import AILApplication
from interfaces.planning import ExecutionReport, Goal

import main


class FakeApplication:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.closed = False

    def handle(self, message: str) -> ExecutionReport:
        self.messages.append(message)
        return ExecutionReport(goal=Goal(message), passed=True, steps=())

    def close(self) -> None:
        self.closed = True


class TestMain(unittest.TestCase):
    def test_mock_mode_remains_compatible(self) -> None:
        with patch("builtins.input", return_value="hello"), patch("builtins.print") as printer:
            main.run_mock_mode()

        printer.assert_called_once_with("AIL: AIL received: hello")

    def test_oi_mode_routes_supported_tasks_through_application(self) -> None:
        application = FakeApplication()
        with (
            patch.object(AILApplication, "create", return_value=application),
            patch(
                "builtins.input",
                side_effect=["Create a file named hello.txt containing ok", "quit"],
            ),
            patch("builtins.print") as printer,
        ):
            main.run_oi_mode()

        self.assertEqual(
            application.messages,
            ["Create a file named hello.txt containing ok"],
        )
        self.assertTrue(application.closed)
        printer.assert_called_once_with("AIL: verified task completed")

    def test_voice_mode_reports_pipeline_errors_and_closes_application(self) -> None:
        application = FakeApplication()
        with (
            patch.object(AILApplication, "create", return_value=application),
            patch(
                "voice.service.VoiceService.run_once",
                new=AsyncMock(side_effect=RuntimeError("speaker unavailable")),
            ),
            patch("sys.stderr", new_callable=StringIO) as stderr,
        ):
            main.run_voice_mode()

        self.assertTrue(application.closed)
        self.assertIn("Voice error: speaker unavailable", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()