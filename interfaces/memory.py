"""AIL-owned memory interface.

This is AIL's public contract for long-term memory.  Implementations can use
any backend (PersonalAI in-memory, PostgreSQL+pgvector, etc.) but AIL core
only depends on these types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Memory:
    """A single long-term memory."""

    id: str
    text: str
    kind: str = "semantic"
    confidence: float = 0.8
    score: float | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryStore(ABC):
    """Interface that every AIL memory backend must implement."""

    @abstractmethod
    def store(
        self,
        text: str,
        *,
        kind: str = "semantic",
        confidence: float = 0.8,
    ) -> Memory:
        """Persist a memory and return it with an assigned id."""

    @abstractmethod
    def recall(self, query: str, *, top_k: int = 5) -> list[Memory]:
        """Return memories most relevant to *query*, ordered by relevance."""

    @abstractmethod
    def list_all(self) -> list[Memory]:
        """Return every stored (non-superseded) memory."""

    @abstractmethod
    def get(self, memory_id: str) -> Memory | None:
        """Return a single memory by id, or None."""

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Remove a memory by id."""
