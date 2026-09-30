"""Real end-to-end test for the production AIL application composition."""

from __future__ import annotations

import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

AIL_ROOT = str(Path(__file__).resolve().parents[1])
if AIL_ROOT not in sys.path:
    sys.path.insert(0, AIL_ROOT)

from core.application import AILApplication  # noqa: E402
from integrations.open_interpreter.config import OIConfig  # noqa: E402
from interfaces.planning import ExecutionReport  # noqa: E402
from interfaces.memory import Memory, MemoryStore  # noqa: E402


def _find_oi_executable() -> str | None:
    candidates = [os.getenv("AIL_OI_EXECUTABLE", ""), "interpreter", "codex", "oi"]
    for candidate in candidates:
        if candidate:
            path = shutil.which(candidate)
            if path:
                return path

    default_path = Path(
        os.environ.get("LOCALAPPDATA", ""),
        "Programs",
        "Open Interpreter",
        "bin",
        "interpreter.exe",
    )
    return str(default_path) if default_path.is_file() else None


class IsolatedMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self.memories: list[Memory] = []

    def store(
        self,
        text: str,
        *,
        kind: str = "semantic",
        confidence: float = 0.8,
    ) -> Memory:
        memory = Memory(
            id=str(len(self.memories)),
            text=text,
            kind=kind,
            confidence=confidence,
        )
        self.memories.append(memory)
        return memory

    def recall(self, query: str, *, top_k: int = 5) -> list[Memory]:
        return self.memories[:top_k]

    def list_all(self) -> list[Memory]:
        return list(self.memories)

    def get(self, memory_id: str) -> Memory | None:
        return next(
            (memory for memory in self.memories if memory.id == memory_id),
            None,
        )

    def delete(self, memory_id: str) -> None:
        self.memories = [
            memory for memory in self.memories if memory.id != memory_id
        ]


class TestApplicationOpenInterpreterE2E(unittest.TestCase):
    def test_production_application_creates_and_verifies_file(self) -> None:
        executable = _find_oi_executable()
        if executable is None:
            self.skipTest("Open Interpreter executable unavailable")

        filename = f"ail_2p_{uuid.uuid4().hex}.txt"
        sentinel = f"AIL_2P_REAL_{uuid.uuid4().hex}"

        workspace = Path(AIL_ROOT) / "data" / f"2p_application_e2e_{uuid.uuid4().hex}"
        workspace.mkdir(parents=True, exist_ok=False)
        application: AILApplication | None = None
        try:
            config = OIConfig(
                executable=executable,
                cwd=str(workspace),
                sandbox="workspace-write",
                connect_timeout=30,
                request_timeout=120,
            )
            application = AILApplication.create(
                base_dir=workspace,
                memory_store=IsolatedMemoryStore(),
                oi_config=config,
                timeout=90,
            )
            responses = []
            original_send_message = application.client.send_message

            def record_response(
                message: str,
                thread_id: str | None = None,
                timeout: float | None = None,
            ) -> object:
                response = original_send_message(
                    message,
                    thread_id=thread_id,
                    timeout=timeout,
                )
                responses.append(response)
                return response

            application.client.send_message = record_response
            report = application.run(
                f"Create a file named {filename} containing {sentinel}"
            )

            self.assertIsInstance(report, ExecutionReport)
            if not report.passed:
                diagnostics = [f"ExecutionReport: {report!r}"]
                for index, response in enumerate(responses, start=1):
                    diagnostics.append(
                        f"OI response {index} text: {response.text!r}"
                    )
                    diagnostics.append(
                        f"OI response {index} items: {response.items!r}"
                    )
                diagnostics.append(
                    f"OI stderr: {application.client.get_stderr()!r}"
                )
                for index, step in enumerate(report.steps, start=1):
                    diagnostics.append(f"Step {index}: {step!r}")
                    if step.recovery is not None:
                        diagnostics.append(
                            f"Step {index} recovery: {step.recovery!r}"
                        )
                self.fail("\n".join(diagnostics))
            self.assertEqual(len(report.steps), 1)

            expected_file = workspace / filename
            self.assertTrue(expected_file.exists())
            self.assertIn(sentinel, expected_file.read_text(encoding="utf-8"))

            step = report.steps[0]
            self.assertIsNotNone(step.recovery)
            assert step.recovery is not None
            self.assertIsNotNone(step.recovery.verification)
            self.assertTrue(step.recovery.verification.passed)
        finally:
            if application is not None:
                application.close()
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
