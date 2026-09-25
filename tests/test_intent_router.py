"""Focused tests for the application intent router."""

from __future__ import annotations

import unittest

from core.intent_router import DefaultIntentRouter
from interfaces.intent import IntentKind


class TestDefaultIntentRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.router = DefaultIntentRouter()

    def assert_memory(self, message: str) -> None:
        result = self.router.route(message)
        self.assertIs(result.kind, IntentKind.MEMORY)
        self.assertEqual(result.reason, "explicit-memory")

    def assert_not_memory(self, message: str) -> None:
        result = self.router.route(message)
        self.assertIs(result.kind, IntentKind.TASK)
        self.assertEqual(result.reason, "not-explicit-memory")

    def test_explicit_memory_forms(self) -> None:
        for message in (
            "Remember that I like coffee",
            "Remember I like coffee",
            "My name is Omer",
            "Remember that I like coffee?",
            "My name is Omer!",
        ):
            with self.subTest(message=message):
                self.assert_memory(message)

    def test_tasks_and_other_messages_are_not_memory(self) -> None:
        for message in (
            "Create a file named notes.txt containing hello",
            "Create a file named notes.txt containing remember that I like coffee",
            "ordinary text",
            "",
            "   ",
        ):
            with self.subTest(message=message):
                self.assert_not_memory(message)

    def test_long_non_ascii_input_is_total(self) -> None:
        self.assert_not_memory("你好 " * 1000)


if __name__ == "__main__":
    unittest.main()
