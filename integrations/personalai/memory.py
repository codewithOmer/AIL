"""AIL memory adapter backed by PersonalAI's in-memory fakes.

This adapter wires PersonalAI's ``InMemoryMemoryStore`` (dot-product similarity
search) behind AIL's own ``MemoryStore`` interface. It is the first integration
slice — no database, no external model, no OI dependency.

Embedding seam: ``FakeModelProvider._deterministic_vector`` hashes raw
character positions, which does not reflect word overlap (e.g. a name query
does not rank the matching memory first). AIL therefore adapts
``FakeModelProvider`` with a token-level deterministic embed so that dot-product
similarity tracks shared words. Storage/search still delegate to PersonalAI's
in-memory store; only the embedding function is AIL-owned.

The async PersonalAI fakes are bridged to a synchronous AIL API via
``asyncio.run()``.
"""

from __future__ import annotations

import asyncio
import math
import re
import sys
import uuid
import zlib
from pathlib import Path

# ---------------------------------------------------------------------------
# PersonalAI path setup (read-only dependency — never modified)
# ---------------------------------------------------------------------------
_PERSONALAI_ROOT = Path(__file__).resolve().parents[3] / "personal-ai"
_CONTRACTS_SRC = str(_PERSONALAI_ROOT / "contracts" / "src")
if _CONTRACTS_SRC not in sys.path:
    sys.path.insert(0, _CONTRACTS_SRC)

from personalai_contracts.ports.storage import MemoryKind  # noqa: E402
from personalai_contracts.testing import (  # noqa: E402
    FakeModelProvider,
    InMemoryMemoryStore,
)

from memory.interface import Memory, MemoryStore  # noqa: E402

_EMBED_DIM = 128
_TOKEN_RE = re.compile(r"\w+")


class _TokenFakeProvider(FakeModelProvider):
    """FakeModelProvider whose embed() produces token-overlap vectors.

    Only ``embed`` is overridden; everything else inherits from PersonalAI's
    fake so this stays a PersonalAI test double.
    """

    async def embed(self, texts, model=""):
        from personalai_contracts.ports.model_provider import EmbeddingResult

        return EmbeddingResult(
            vectors=[_token_overlap_vector(text) for text in texts],
            model=model,
            dimensions=_EMBED_DIM,
        )


def _token_overlap_vector(text: str) -> list[float]:
    """Deterministic unit-norm bag-of-token vector (feature hashing).

    Shared words between query and memory produce overlapping dimensions, so
    the dot-product used by ``InMemoryMemoryStore`` ranks by word overlap.
    """
    vec = [0.0] * _EMBED_DIM
    for token in _TOKEN_RE.findall(text.lower()):
        slot = zlib.crc32(token.encode("utf-8")) % _EMBED_DIM
        vec[slot] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

_PROVIDER = _TokenFakeProvider(name="ail-embedding")
_MODEL = "ail-fake"


class PersonalAIMemoryStore(MemoryStore):
    """In-memory memory store that delegates to PersonalAI's in-memory store.

    This is *not* a production adapter — it exists to prove the integration
    boundary works and to back unit tests without any external service.
    """

    def __init__(self) -> None:
        self._store = InMemoryMemoryStore()

    # -- helpers ----------------------------------------------------------

    def _run(self, coro):  # noqa: ANN001
        return asyncio.run(coro)

    def _embed(self, text: str):
        result = self._run(_PROVIDER.embed([text], _MODEL))
        return result.vectors[0]

    # -- MemoryStore interface -------------------------------------------

    def store(
        self,
        text: str,
        *,
        kind: str = "semantic",
        confidence: float = 0.8,
    ) -> Memory:
        embedding = self._embed(text)
        item = self._run(
            self._store.add(
                id=str(uuid.uuid4()),
                kind=MemoryKind(kind),
                text=text,
                embedding=embedding,
                confidence=confidence,
                source={"origin": "ail"},
            )
        )
        return Memory(
            id=item.id,
            text=item.text,
            kind=item.kind.value,
            confidence=item.confidence,
            created_at=item.created_at,
        )

    def recall(self, query: str, *, top_k: int = 5) -> list[Memory]:
        embedding = self._embed(query)
        items = self._run(self._store.search(embedding, top_k=top_k))
        return [
            Memory(
                id=m.id,
                text=m.text,
                kind=m.kind.value,
                confidence=m.confidence,
                score=m.score,
                created_at=m.created_at,
            )
            for m in items
        ]

    def list_all(self) -> list[Memory]:
        items = self._run(self._store.list())
        return [
            Memory(
                id=m.id,
                text=m.text,
                kind=m.kind.value,
                confidence=m.confidence,
                created_at=m.created_at,
            )
            for m in items
        ]

    def get(self, memory_id: str) -> Memory | None:
        item = self._run(self._store.get(memory_id))
        if item is None:
            return None
        return Memory(
            id=item.id,
            text=item.text,
            kind=item.kind.value,
            confidence=item.confidence,
            created_at=item.created_at,
        )

    def delete(self, memory_id: str) -> None:
        self._run(self._store.delete(memory_id))