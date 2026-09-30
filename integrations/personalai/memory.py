"""AIL memory adapter backed by PersonalAI's in-memory store, JSON-persisted.

This adapter wires PersonalAI's ``InMemoryMemoryStore`` (dot-product similarity
search) behind AIL's own ``MemoryStore`` interface. It is the first integration
slice — no database, no external model, no OI dependency.

Embedding seam: ``FakeModelProvider._deterministic_vector`` hashes raw
character positions, which does not reflect word overlap (e.g. a name query
does not rank the matching memory first). AIL therefore adapts
``FakeModelProvider`` with a token-level deterministic embed so that dot-product
similarity tracks shared words. Storage/search still delegate to PersonalAI's
in-memory store; only the embedding function is AIL-owned.

Persistence: every stored memory is mirrored to a small JSON file (AIL-owned,
default ``data/memory.json``) so long-term memory survives process restarts.
Embeddings are *not* serialised — they are recomputed deterministically on
load from the stored text. IDs, text, kind, confidence and ``created_at`` are
preserved across reloads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sys
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

from interfaces.memory import Memory, MemoryStore  # noqa: E402

_LOGGER = logging.getLogger(__name__)

_EMBED_DIM = 128
_TOKEN_RE = re.compile(r"\w+")

# ---------------------------------------------------------------------------
# Embedding provider (PersonalAI test double, embed() AIL-owned)
# ---------------------------------------------------------------------------


class _TokenFakeProvider(FakeModelProvider):
    """FakeModelProvider whose embed() produces token-overlap vectors."""

    async def embed(self, texts, model=""):
        from personalai_contracts.ports.model_provider import EmbeddingResult

        return EmbeddingResult(
            vectors=[_token_overlap_vector(text) for text in texts],
            model=model,
            dimensions=_EMBED_DIM,
        )


def _token_overlap_vector(text: str) -> list[float]:
    """Deterministic unit-norm bag-of-token vector (feature hashing)."""
    vec = [0.0] * _EMBED_DIM
    for token in _TOKEN_RE.findall(text.lower()):
        slot = zlib.crc32(token.encode("utf-8")) % _EMBED_DIM
        vec[slot] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


_PROVIDER = _TokenFakeProvider(name="ail-embedding")
_MODEL = "ail-fake"

# ---------------------------------------------------------------------------
# Default persistence location (AIL-owned)
# ---------------------------------------------------------------------------
_AIL_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MEMORY_FILE = _AIL_ROOT / "data" / "memory.json"


class PersonalAIMemoryStore(MemoryStore):
    """In-memory memory store that persists to a JSON snapshot file."""

    def __init__(self, memory_file: str | Path | None = None) -> None:
        self._store = InMemoryMemoryStore()
        self._memory_file = Path(memory_file or _DEFAULT_MEMORY_FILE)
        self._load_from_file()

    # -- helpers ----------------------------------------------------------

    def _run(self, coro):  # noqa: ANN001
        return asyncio.run(coro)

    def _embed(self, text: str):
        result = self._run(_PROVIDER.embed([text], _MODEL))
        return result.vectors[0]

    def _load_from_file(self) -> None:
        if not self._memory_file.exists():
            return
        try:
            with open(self._memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                embedding = _token_overlap_vector(item["text"])
                self._run(
                    self._store.add(
                        id=item["id"],
                        kind=MemoryKind(item["kind"]),
                        text=item["text"],
                        embedding=embedding,
                        confidence=item["confidence"],
                        source=item.get("source", {"origin": "ail"}),
                    )
                )
        except (json.JSONDecodeError, KeyError) as e:
            _LOGGER.warning("Failed to load memory from %s: %s", self._memory_file, e)

    def _save_to_file(self) -> None:
        self._memory_file.parent.mkdir(parents=True, exist_ok=True)
        items = self._run(self._store.list())
        data = [
            {
                "id": m.id,
                "kind": m.kind.value,
                "text": m.text,
                "confidence": m.confidence,
                "source": m.source,
                "created_at": m.created_at.isoformat(),
            }
            for m in items
        ]
        with open(self._memory_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

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
        self._save_to_file()
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
        self._save_to_file()
