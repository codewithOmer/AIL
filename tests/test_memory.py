"""Tests for AIL long-term memory store and recall.

These tests exercise the PersonalAI-backed in-memory adapter.
No database, no external model, no OI integration.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import uuid
from pathlib import Path

# Ensure AIL root is importable.
AIL_ROOT = str(Path(__file__).resolve().parents[1])
if AIL_ROOT not in sys.path:
    sys.path.insert(0, AIL_ROOT)

from memory.interface import Memory, MemoryStore  # noqa: E402
from integrations.personalai.memory import PersonalAIMemoryStore  # noqa: E402

# Isolated temp dir so these tests never read/write the default
# persistent memory file (data/memory.json).
_TEST_DIR = tempfile.TemporaryDirectory()


def _make_store() -> MemoryStore:
    path = Path(_TEST_DIR.name) / f"memory-{uuid.uuid4().hex}.json"
    return PersonalAIMemoryStore(memory_file=path)


class TestStoreAndRecall(unittest.TestCase):
    def test_store_returns_memory_with_id(self) -> None:
        store = _make_store()
        mem = store.store("My name is Omer.")
        self.assertIsInstance(mem, Memory)
        self.assertTrue(mem.id)  # non-empty
        self.assertEqual(mem.text, "My name is Omer.")
        self.assertEqual(mem.kind, "semantic")

    def test_recall_returns_relevant_memory(self) -> None:
        store = _make_store()
        store.store("My name is Omer.")
        store.store("I am a 3rd semester student.")
        store.store("I play basketball.")

        results = store.recall("What is my name?")
        self.assertGreater(len(results), 0)
        # The top hit should be the memory mentioning "Omer" / "name".
        self.assertIn("Omer", results[0].text)

    def test_recall_ranks_by_similarity(self) -> None:
        store = _make_store()
        store.store("My name is Omer.")
        store.store("I am a 3rd semester student.")
        store.store("I play basketball.")

        results = store.recall("What is my name?", top_k=2)
        self.assertEqual(len(results), 2)
        # First result should be more relevant than second.
        self.assertIsNotNone(results[0].score)
        self.assertIsNotNone(results[1].score)
        self.assertGreaterEqual(results[0].score or 0.0, results[1].score or 0.0)


class TestNoResultBehavior(unittest.TestCase):
    def test_recall_empty_store(self) -> None:
        store = _make_store()
        results = store.recall("anything at all")
        self.assertEqual(results, [])

    def test_recall_unrelated_query(self) -> None:
        store = _make_store()
        store.store("My name is Omer.")
        store.store("I am a 3rd semester student.")
        store.store("I play basketball.")

        results = store.recall("quantum entanglement theory")
        # The fake embedding always yields some non-zero overlap, so results
        # may come back weakly ranked; whatever is returned must be ordered.
        for prev, cur in zip(results, results[1:]):
            self.assertGreaterEqual(prev.score or 0.0, cur.score or 0.0)


class TestMultipleMemories(unittest.TestCase):
    def test_recall_can_return_multiple(self) -> None:
        store = _make_store()
        store.store("My name is Omer.")
        store.store("My friend is Ali.")
        store.store("I play basketball.")

        results = store.recall("My name and my friend")
        # Both "My name is Omer" and "My friend is Ali" share the "My" token
        # pattern; basketball does not.
        self.assertGreaterEqual(len(results), 2)
        texts = {r.text for r in results}
        self.assertIn("My name is Omer.", texts)
        self.assertIn("My friend is Ali.", texts)


class TestLifecycle(unittest.TestCase):
    def test_list_all_returns_all_stored(self) -> None:
        store = _make_store()
        store.store("A")
        store.store("B")
        store.store("C")
        self.assertEqual(len(store.list_all()), 3)

    def test_get_returns_exact_memory(self) -> None:
        store = _make_store()
        mem = store.store("Hello world")
        fetched = store.get(mem.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.text, "Hello world")

    def test_get_returns_none_for_unknown_id(self) -> None:
        store = _make_store()
        self.assertIsNone(store.get("nonexistent"))

    def test_delete_removes_memory(self) -> None:
        store = _make_store()
        mem = store.store("Temporary fact")
        store.delete(mem.id)
        self.assertIsNone(store.get(mem.id))
        self.assertEqual(len(store.list_all()), 0)


if __name__ == "__main__":
    unittest.main()