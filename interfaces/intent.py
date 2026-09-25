"""AIL intent-routing contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class IntentKind(Enum):
    MEMORY = "memory"
    TASK = "task"
    OTHER = "other"


@dataclass(frozen=True)
class IntentResult:
    kind: IntentKind
    reason: str


class IntentRouter(ABC):
    """Route a message without executing or rewriting it."""

    @abstractmethod
    def route(self, message: str) -> IntentResult:
        """Return a total intent result for *message*."""
        raise NotImplementedError
