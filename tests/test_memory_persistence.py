"""Tests for AIL long-term memory persistence."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from integrations.personalai.memory import PersonalAIMemoryStore

class TestMemoryPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.memory_file = Path(self.test_dir) / "test_memory.json"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_persistence_between_instances(self):
        # Instance A stores a memory
        store_a = PersonalAIMemoryStore(memory_file=self.memory_file)
        mem = store_a.store("My secret code is 1234.")
        
        # Instance B reloads it
        store_b = PersonalAIMemoryStore(memory_file=self.memory_file)
        all_mems = store_b.list_all()
        self.assertEqual(len(all_mems), 1)
        self.assertEqual(all_mems[0].text, "My secret code is 1234.")
        self.assertEqual(all_mems[0].id, mem.id)

    def test_recall_after_reload(self):
        store_a = PersonalAIMemoryStore(memory_file=self.memory_file)
        store_a.store("The user likes coffee.")
        store_a.store("The user dislikes tea.")
        
        store_b = PersonalAIMemoryStore(memory_file=self.memory_file)
        results = store_b.recall("What does the user want to drink?")
        self.assertGreater(len(results), 0)
        self.assertIn("coffee", results[0].text)

    def test_delete_persists(self):
        store_a = PersonalAIMemoryStore(memory_file=self.memory_file)
        mem = store_a.store("To be deleted")
        store_a.delete(mem.id)
        
        store_b = PersonalAIMemoryStore(memory_file=self.memory_file)
        self.assertEqual(len(store_b.list_all()), 0)

    def test_missing_file_starts_empty(self):
        store = PersonalAIMemoryStore(memory_file=self.memory_file)
        self.assertEqual(len(store.list_all()), 0)

    def test_malformed_json_starts_empty_and_does_not_crash(self):
        with open(self.memory_file, "w") as f:
            f.write("not json at all")
        
        # Should start empty without crashing
        store = PersonalAIMemoryStore(memory_file=self.memory_file)
        self.assertEqual(len(store.list_all()), 0)

if __name__ == "__main__":
    unittest.main()
