"""Offline tests for AIL image propagation through the existing task path."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from core.application import AILApplication
from interfaces.image import LocalImage


class _ImageRecordingClient:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.images = None

    def send_message(self, message, thread_id=None, timeout=None, images=None):
        self.images = images
        (self.workspace / "result.txt").write_text("image task completed")
        return "executor response"


class TestImagePropagation(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.workspace = Path(self.directory)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory)

    def test_application_propagates_local_image_without_parallel_agent(self) -> None:
        image_path = self.workspace / "input.png"
        image_path.write_bytes(b"image")
        client = _ImageRecordingClient(self.workspace)
        memory = MagicMock()
        memory.recall.return_value = []
        application = AILApplication.create(
            base_dir=self.workspace,
            memory_store=memory,
            client=client,
            start_client=False,
        )

        report = application.run(
            "Create a file named result.txt containing image task completed",
            images=[LocalImage.from_path(image_path)],
        )

        self.assertTrue(report.passed)
        self.assertEqual(client.images, [LocalImage.from_path(image_path)])

    def test_memory_messages_reject_images(self) -> None:
        application = AILApplication.create(
            base_dir=self.workspace,
            memory_store=MagicMock(),
            client=MagicMock(),
            start_client=False,
        )

        with self.assertRaisesRegex(ValueError, "only supported for task messages"):
            application.handle(
                "Remember that this is an image",
                images=[LocalImage.from_path(self.workspace / "input.png")],
            )


if __name__ == "__main__":
    unittest.main()