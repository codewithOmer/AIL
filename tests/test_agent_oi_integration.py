"""Real end-to-end Agent and Open Interpreter integration test."""

from __future__ import annotations

import os
import shutil
import unittest
import uuid
from pathlib import Path

AIL_ROOT = str(Path(__file__).resolve().parents[1])

from core.agent import Agent  # noqa: E402
from core.plan_runner import PlanRunner  # noqa: E402
from integrations.open_interpreter.client import OpenInterpreterClient  # noqa: E402
from integrations.open_interpreter.config import OIConfig  # noqa: E402
from interfaces.planning import (  # noqa: E402
    Goal,
    Plan,
    PlanStep,
    Planner,
    StepStatus,
)
from interfaces.memory import Memory, MemoryStore  # noqa: E402
from core.verification import FileExpectation, FilesystemVerifier  # noqa: E402


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


class EmptyMemoryStore(MemoryStore):
    def store(
        self,
        text: str,
        *,
        kind: str = "semantic",
        confidence: float = 0.8,
    ) -> Memory:
        raise AssertionError("the integration test must not persist memories")

    def recall(self, query: str, *, top_k: int = 5) -> list[Memory]:
        return []

    def list_all(self) -> list[Memory]:
        raise AssertionError("the integration test must not list memories")

    def get(self, memory_id: str) -> Memory | None:
        raise AssertionError("the integration test must not get memories")

    def delete(self, memory_id: str) -> None:
        raise AssertionError("the integration test must not delete memories")


class HelloFilePlanner(Planner):
    def plan(self, goal: Goal) -> Plan:
        return Plan(
            goal=goal,
            steps=(
                PlanStep(
                    id="create-hello",
                    action=(
                        "Create hello.txt in the current workspace containing "
                        "exactly the text AIL_E2E_OK_2026."
                    ),
                    expectations=(
                        FileExpectation(
                            path="hello.txt",
                            exists=True,
                            contains="AIL_E2E_OK_2026",
                        ),
                    ),
                ),
            ),
        )


class RecordingClient:
    def __init__(self, client: OpenInterpreterClient) -> None:
        self.client = client
        self.responses: list[str] = []

    def send_message(
        self,
        message: str,
        thread_id: str | None = None,
        timeout: float | None = None,
    ) -> object:
        response = self.client.send_message(
            message,
            thread_id=thread_id,
            timeout=timeout,
        )
        self.responses.append(response.text)
        return response


class TestAgentOpenInterpreterIntegration(unittest.TestCase):
    def test_agent_runs_real_verified_file_creation(self) -> None:
        executable = _find_oi_executable()
        if executable is None:
            self.skipTest(
                "Open Interpreter executable unavailable; set AIL_OI_EXECUTABLE "
                "or install interpreter, codex, or oi"
            )

        workspace = Path(AIL_ROOT) / "data" / f"2i_smoke_{uuid.uuid4().hex}"
        workspace.mkdir(parents=True, exist_ok=False)
        client: OpenInterpreterClient | None = None
        try:
            client = OpenInterpreterClient(
                OIConfig(
                    executable=executable,
                    cwd=str(workspace),
                    sandbox="workspace-write",
                    connect_timeout=30,
                    request_timeout=120,
                )
            )
            client.start()
            thread_id = client.create_thread(cwd=str(workspace))
            recording_client = RecordingClient(client)

            runner = PlanRunner(
                planner=HelloFilePlanner(),
                client=recording_client,
                verifier=FilesystemVerifier(),
                base_dir=str(workspace),
                default_max_attempts=2,
                thread_id=thread_id,
            )
            report = Agent(EmptyMemoryStore(), runner).run(
                "Create hello.txt containing AIL_E2E_OK_2026"
            )

            verification = report.steps[0].recovery.verification
            diagnostic = (
                f"report={report!r}\n"
                f"verification={verification!r}\n"
                f"oi_responses={recording_client.responses!r}"
            )
            self.assertTrue(report.passed, diagnostic)
            self.assertEqual(len(report.steps), 1)
            step = report.steps[0]
            self.assertIs(step.status, StepStatus.PASSED)
            self.assertIsNotNone(step.recovery)
            assert step.recovery is not None
            self.assertTrue(step.recovery.passed)
            self.assertTrue(step.recovery.verification.passed)

            hello = workspace / "hello.txt"
            self.assertTrue(hello.exists())
            self.assertIn("AIL_E2E_OK_2026", hello.read_text(encoding="utf-8"))
        finally:
            if client is not None:
                client.shutdown()
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()