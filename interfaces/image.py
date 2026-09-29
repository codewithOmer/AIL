"""AIL-owned image input representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalImage:
    """A local image path to be resolved by Open Interpreter."""

    path: Path
    detail: str | None = None

    @classmethod
    def from_path(cls, path: str | Path, *, detail: str | None = None) -> "LocalImage":
        return cls(path=Path(path), detail=detail)

    def validated_path(self) -> Path:
        resolved = self.path.expanduser().resolve()
        if not resolved.exists():
            raise ValueError(f"image path does not exist: {self.path}")
        if not resolved.is_file():
            raise ValueError(f"image path is not a file: {self.path}")
        return resolved