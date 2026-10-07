"""Deterministic, execution-free planning foundation (Milestone 2F).

``DeterministicPlanner`` produces a reflexive one-step plan per goal and never
executes, verifies or communicates with any executor.  ``validate_plan`` is an
AIL-owned integrity check over the static plan form.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath

from interfaces.planning import (
    ExecutionReport,
    Goal,
    Plan,
    PlanStep,
    Planner,
    Replanner,
    StepStatus,
)
from interfaces.verification import FileExpectation


class UnsupportedTaskError(ValueError):
    """Raised when the production deterministic planner cannot handle a goal."""


class UnrecognizedTaskError(UnsupportedTaskError):
    """Raised when a planner does not recognize the goal's syntax at all.

    The distinction from its base class is what makes planner composition safe:
    a composite may fall through on a recognition miss, but a planner that
    *recognized* the syntax and rejected the goal (for example because the
    requested path escapes the workspace) raises the base
    ``UnsupportedTaskError``, which is never caught during fall-through.
    """


_CREATE_FILE_RE = re.compile(
    r"create\s+(?:(?:a\s+)?file(?:\s+named)?\s+)?"
    r"(?P<filename>[^\s]+)\s+containing\s+(?P<content>.+?)\s*\.?$",
    re.IGNORECASE,
)


def _is_safe_filename(filename: str) -> bool:
    """Accept only one relative filename within the configured workspace."""
    if not filename or filename in {".", ".."}:
        return False
    if "/" in filename or "\\" in filename:
        return False

    posix_path = PurePosixPath(filename)
    windows_path = PureWindowsPath(filename)
    return not (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
    )


def _is_safe_relative_path(path: str) -> bool:
    """Accept only a relative path that stays inside the configured workspace.

    Unlike ``_is_safe_filename`` this permits safe relative subdirectories
    (``src/config/settings.json``) while still rejecting traversal, absolute
    paths, drive letters and UNC shares.

    This is a defence-in-depth convenience boundary, not the containment
    boundary.  Since 2F.2 ``FilesystemVerifier`` resolves every expectation
    path itself and refuses to stat or read anything outside its resolved base
    directory, independently of what a planner supplied, so an unsafe path
    cannot reach the filesystem even if it were accepted here.  Keeping the
    planner-side check too means an unsafe goal is rejected with a clear
    ``UnsupportedTaskError`` before any execution, rather than only at
    verification time.
    """
    if not path or not path.strip():
        return False
    if "\x00" in path:
        return False
    # Backslashes are rejected outright instead of being treated as separators,
    # so a Windows-style traversal can never be smuggled through as safe.
    if "\\" in path:
        return False

    posix_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
    ):
        return False

    parts = posix_path.parts
    if not parts:
        return False
    return not any(part in {".", ".."} for part in parts)


class SupportedFileTaskPlanner(Planner):
    """Plan the narrow, explicitly supported file-creation task shape."""

    def plan(self, goal: Goal) -> Plan:
        matches = list(_CREATE_FILE_RE.finditer(goal.description))
        match = matches[-1] if matches else None
        if match is None:
            raise UnrecognizedTaskError(
                "Unsupported task. Supported syntax: "
                "Create a file named <filename> containing <content>"
            )

        filename = match.group("filename")
        if not _is_safe_filename(filename):
            raise UnsupportedTaskError(
                "Unsupported task. The filename must be a single relative "
                "filename within the configured workspace."
            )
        content = match.group("content").strip()
        return Plan(
            goal=goal,
            steps=(
                PlanStep(
                    id="create-file",
                    action=(
                        f"Create {filename} in the current workspace containing "
                        f"exactly the text {content}."
                    ),
                    expectations=(
                        FileExpectation(
                            path=filename,
                            exists=True,
                            contains=content,
                        ),
                    ),
                ),
            ),
        )


class DeterministicFileReplanner(Replanner):
    """Create one corrective replacement plan for a failed file task."""

    def replan(self, plan: Plan, report: ExecutionReport) -> Plan | None:
        if report.passed or len(report.attempts) != 1:
            return None
        if len(plan.steps) != 1 or len(report.steps) != 1:
            return None

        step = plan.steps[0]
        result = report.steps[0]
        if result.status is not StepStatus.FAILED or result.recovery is None:
            return None
        if len(step.expectations) != 1:
            return None

        expectation = step.expectations[0]
        if not isinstance(expectation, FileExpectation):
            return None

        try:
            canonical = SupportedFileTaskPlanner().plan(plan.goal)
        except UnsupportedTaskError:
            return None
        canonical_expectation = canonical.steps[0].expectations[0]
        if canonical_expectation != expectation:
            return None

        replacement_step = PlanStep(
            id=step.id,
            action=(
                f"{step.action}\n\n"
                "The previous attempt did not satisfy the file requirement. "
                f"Ensure {expectation.path} exists and contains "
                f"{expectation.contains!r}."
            ),
            expectations=(
                FileExpectation(
                    path=expectation.path,
                    exists=expectation.exists,
                    contains=expectation.contains,
                ),
            ),
            depends_on=step.depends_on,
            max_attempts=step.max_attempts,
        )
        return Plan(goal=plan.goal, steps=(replacement_step,))


_SETUP_WORKSPACE_RE = re.compile(
    r"set\s+up\s+(?:a\s+)?project\s+workspace\s+(?P<dir>[^\s]+)\s+"
    r"with\s+a\s+readme(?:\s+file)?\s+containing\s+(?P<readme>.+?)\s+"
    r"and\s+a\s+config(?:\s+file)?\s+containing\s+(?P<config>.+?)\s*\.?$",
    re.IGNORECASE,
)


class WorkspaceSetupPlanner(Planner):
    """Plan a deterministic three-step project workspace setup.

    Supported goal shape (narrow and explicit)::

        set up a project workspace <dir> with a readme containing <readme>
        and a config file containing <config>

    Emits one directory step followed by two file steps that each depend on the
    directory step, so the plan exercises dependency ordering in the runner.
    Every step carries at least one :class:`FileExpectation`, which keeps the
    per-step verification evidence non-vacuous.  Output is fully deterministic:
    the same goal always produces an identical plan.
    """

    def plan(self, goal: Goal) -> Plan:
        matches = list(_SETUP_WORKSPACE_RE.finditer(goal.description))
        match = matches[-1] if matches else None
        if match is None:
            raise UnrecognizedTaskError(
                "Unsupported task. Supported syntax: set up a project workspace "
                "<dir> with a readme containing <readme> and a config file "
                "containing <config>"
            )

        directory = match.group("dir").rstrip("/")
        if not _is_safe_relative_path(directory):
            raise UnsupportedTaskError(
                "Unsupported task. The workspace path must be a relative path "
                "inside the configured workspace."
            )

        readme_content = match.group("readme").strip()
        config_content = match.group("config").strip()
        if not readme_content or not config_content:
            raise UnsupportedTaskError(
                "Unsupported task. The readme and config contents must be "
                "non-empty."
            )

        readme_path = f"{directory}/README.md"
        config_path = f"{directory}/config.json"
        for derived_path in (readme_path, config_path):
            if not _is_safe_relative_path(derived_path):
                raise UnsupportedTaskError(
                    "Unsupported task. Derived paths must stay inside the "
                    "configured workspace."
                )

        return Plan(
            goal=goal,
            steps=(
                PlanStep(
                    id="create-workspace",
                    action=(
                        f"Create the directory {directory} in the current "
                        f"workspace."
                    ),
                    expectations=(
                        FileExpectation(path=directory, exists=True),
                    ),
                ),
                PlanStep(
                    id="create-readme",
                    action=(
                        f"Create {readme_path} inside {directory} containing "
                        f"exactly the text {readme_content}."
                    ),
                    expectations=(
                        FileExpectation(
                            path=readme_path,
                            exists=True,
                            contains=readme_content,
                        ),
                    ),
                    depends_on=("create-workspace",),
                ),
                PlanStep(
                    id="create-config",
                    action=(
                        f"Create {config_path} inside {directory} containing "
                        f"exactly the text {config_content}."
                    ),
                    expectations=(
                        FileExpectation(
                            path=config_path,
                            exists=True,
                            contains=config_content,
                        ),
                    ),
                    depends_on=("create-workspace",),
                ),
            ),
        )


class DeterministicPlanner(Planner):
    """Reflexive planner: one executor step per goal.

    For Milestone 2F the plan is a single step whose action is the goal's
    description.  Verification/recovery/report remain runner responsibilities
    (see 2D/2E) and are deliberately NOT plan steps.
    """

    def plan(self, goal: Goal) -> Plan:
        return Plan(
            goal=goal,
            steps=(PlanStep(id="step-1", action=goal.description),),
        )


class DeterministicMultiStepPlanner(Planner):
    """Fixed two-step planner for proving structured plan execution."""

    def plan(self, goal: Goal) -> Plan:
        return Plan(
            goal=goal,
            steps=(
                PlanStep(
                    id="prepare-workspace",
                    action="Prepare the workspace for the requested task.",
                    expectations=("workspace-prepared",),
                ),
                PlanStep(
                    id="complete-task",
                    action=goal.description,
                    expectations=("task-completed",),
                    depends_on=("prepare-workspace",),
                ),
            ),
        )


class GoalAwarePlanner(Planner):
    """Select one of two predefined plans from an exact goal description."""

    def plan(self, goal: Goal) -> Plan:
        if goal.description == "Create a project workspace":
            return Plan(
                goal=goal,
                steps=(
                    PlanStep(
                        id="create-workspace",
                        action="Create the project workspace.",
                        expectations=("project-workspace-created",),
                    ),
                    PlanStep(
                        id="verify-workspace",
                        action="Verify the project workspace.",
                        expectations=("project-workspace-verified",),
                        depends_on=("create-workspace",),
                    ),
                ),
            )
        if goal.description == "Create a project summary":
            return Plan(
                goal=goal,
                steps=(
                    PlanStep(
                        id="collect-summary",
                        action="Collect project summary details.",
                        expectations=("summary-details-collected",),
                    ),
                    PlanStep(
                        id="write-summary",
                        action="Write the project summary.",
                        expectations=("project-summary-written",),
                        depends_on=("collect-summary",),
                    ),
                ),
            )
        return DeterministicPlanner().plan(goal)


def validate_plan(plan: Plan) -> None:
    """Reject structurally invalid plans.

    Execution order remains tuple order, so dependencies must point backward
    to an earlier step. Raises ``ValueError`` on the first problem found.
    """
    if not plan.steps:
        raise ValueError("plan must contain at least one step")

    step_ids: set[str] = set()
    for step in plan.steps:
        sid = step.id
        if not sid or not sid.strip():
            raise ValueError("every step must have a non-empty id")
        action = step.action
        if not isinstance(action, str) or not action.strip():
            raise ValueError(f"step {sid!r} action must be non-empty")
        if sid in step_ids:
            raise ValueError(f"duplicate step id: {sid!r}")
        step_ids.add(sid)
        if step.max_attempts is not None and step.max_attempts < 1:
            raise ValueError(
                f"step {sid!r} max_attempts must be None or >= 1, "
                f"got {step.max_attempts}"
            )
        if len(set(step.depends_on)) != len(step.depends_on):
            raise ValueError(f"step {sid!r} lists a dependency more than once")

    for step in plan.steps:
        for dep in step.depends_on:
            if dep == step.id:
                raise ValueError(f"step {step.id!r} cannot depend on itself")
            if dep not in step_ids:
                raise ValueError(
                    f"step {step.id!r} depends on unknown step {dep!r}"
                )

    graph = {step.id: step.depends_on for step in plan.steps}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visiting:
            raise ValueError(f"dependency cycle includes step {step_id!r}")
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency in graph[step_id]:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step in plan.steps:
        visit(step.id)

    positions = {step.id: index for index, step in enumerate(plan.steps)}
    for step in plan.steps:
        for dependency in step.depends_on:
            if positions[dependency] >= positions[step.id]:
                raise ValueError(
                    f"step {step.id!r} depends on later step {dependency!r}"
                )


class FirstMatchPlanner(Planner):
    """Delegate to the first planner that recognizes a goal's syntax.

    Fall-through happens on :class:`UnrecognizedTaskError` only.  A planner that
    recognizes the syntax but rejects the goal — an unsafe path, an empty
    required value — raises the base :class:`UnsupportedTaskError`, which is
    deliberately not caught here so that such a goal can never be silently
    retried against a different planner.
    """

    def __init__(self, *planners: Planner) -> None:
        self._planners = planners

    @property
    def planners(self) -> tuple[Planner, ...]:
        """Delegates in fall-through order."""
        return self._planners

    def plan(self, goal: Goal) -> Plan:
        for planner in self._planners:
            try:
                return planner.plan(goal)
            except UnrecognizedTaskError:
                continue
        raise UnsupportedTaskError(
            "Unsupported task. Supported syntax: "
            "Create a file named <filename> containing <content>; or "
            "set up a project workspace <dir> with a readme containing "
            "<readme> and a config file containing <config>"
        )