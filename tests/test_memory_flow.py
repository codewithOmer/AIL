"""Tests for AIL memory-flow helpers (recall context + explicit fact store)."""

from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

# Isolated temp dir so these tests never read/write the default
# persistent memory file (data/memory.json).
_TEST_DIR = tempfile.TemporaryDirectory()


def _make_store() -> PersonalAIMemoryStore:
    path = Path(_TEST_DIR.name) / f"memory-{uuid.uuid4().hex}.json"
    return PersonalAIMemoryStore(memory_file=path)

from memory.flow import (  # noqa: E402
    apply_context,
    build_context_block,
    extract_explicit_facts,
    store_if_new,
)
from interfaces.memory import Memory  # noqa: E402
from integrations.personalai.memory import PersonalAIMemoryStore  # noqa: E402


class _Mem(Memory):
    def __init__(self, text: str, score: float = 1.0) -> None:
        super().__init__(id=text, text=text, score=score)


class TestContextBlock(unittest.TestCase):
    def test_empty_recall_returns_empty_block(self) -> None:
        self.assertEqual(build_context_block([]), "")

    def test_single_memory_renders_labeled_block(self) -> None:
        block = build_context_block([_Mem("The user's name is Omer.")])
        self.assertIn("[Memory context from previous conversations:]", block)
        self.assertIn("- The user's name is Omer.", block)

    def test_multiple_memories_render_list(self) -> None:
        block = build_context_block(
            [_Mem("A"), _Mem("B")]
        )
        self.assertIn("- A", block)
        self.assertIn("- B", block)


class TestApplyContext(unittest.TestCase):
    def test_no_recall_passthrough_unchanged(self) -> None:
        self.assertEqual(apply_context("Hello", []), "Hello")

    def test_recall_prepends_context_block(self) -> None:
        msg = apply_context(
            "What is my name?",
            [_Mem("The user's name is Omer.")],
        )
        self.assertTrue(msg.startswith("[Memory context from previous conversations:]"))
        self.assertIn("The user's name is Omer.", msg)
        self.assertTrue(msg.endswith("What is my name?"))


class TestExtractExplicitFacts(unittest.TestCase):
    def test_name_fact(self) -> None:
        self.assertEqual(
            extract_explicit_facts("My name is Omer."),
            ["The user's name is Omer."],
        )

    def test_remember_that_fact(self) -> None:
        self.assertEqual(
            extract_explicit_facts("remember that I like coffee"),
            ["I like coffee."],
        )

    def test_remember_without_that(self) -> None:
        self.assertEqual(
            extract_explicit_facts("Remember I play basketball"),
            ["I play basketball."],
        )

    def test_name_case_insensitive(self) -> None:
        self.assertEqual(
            extract_explicit_facts("MY NAME IS Omer"),
            ["The user's name is Omer."],
        )

    def test_non_fact_message_returns_empty(self) -> None:
        self.assertEqual(extract_explicit_facts("Hello, how are you?"), [])

    def test_strips_trailing_punctuation(self) -> None:
        self.assertEqual(
            extract_explicit_facts("my name is Omer!"),
            ["The user's name is Omer."],
        )

    def test_question_mark_is_normalized_once(self) -> None:
        self.assertEqual(
            extract_explicit_facts("My name is Omer?"),
            ["The user's name is Omer."],
        )
        self.assertEqual(
            extract_explicit_facts("Remember that I like coffee?"),
            ["I like coffee."],
        )

    def test_legitimate_fact_content_is_preserved(self) -> None:
        self.assertEqual(
            extract_explicit_facts("Remember that I like coffee and tea."),
            ["I like coffee and tea."],
        )
        self.assertEqual(
            extract_explicit_facts("Remember that my email is a@b.com."),
            ["my email is a@b.com."],
        )
        self.assertEqual(
            extract_explicit_facts("Remember that I work at 9:30."),
            ["I work at 9:30."],
        )
        self.assertEqual(
            extract_explicit_facts("Remember that I write code."),
            ["I write code."],
        )
        self.assertEqual(
            extract_explicit_facts("Remember that I run marathons."),
            ["I run marathons."],
        )
        self.assertEqual(
            extract_explicit_facts("Remember that I create art."),
            ["I create art."],
        )

    def test_rejects_composite_explicit_messages(self) -> None:
        messages = (
            "My name is Omer. Remember that I like coffee.",
            "Remember that I like coffee. Create a file named notes.txt containing secret",
            "My name is Omer, and I like coffee",
            "Create a file named notes.txt containing remember that I like coffee",
            "Remember that I like coffee. 2026 is next",
        )

        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(extract_explicit_facts(message), [])


class TestStoreIfNew(unittest.TestCase):
    def test_stores_new_fact(self) -> None:
        store = _make_store()
        mem = store_if_new(store, "The user's name is Omer.")
        self.assertIsNotNone(mem)
        self.assertEqual(store.list_all()[0].text, "The user's name is Omer.")

    def test_does_not_duplicate_identical_fact(self) -> None:
        store = _make_store()
        store_if_new(store, "The user's name is Omer.")
        second = store_if_new(store, "The user's name is Omer.")
        self.assertIsNone(second)
        self.assertEqual(len(store.list_all()), 1)


if __name__ == "__main__":
    unittest.main()