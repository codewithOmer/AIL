"""Application result for an ordinary model conversation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationResponse:
    """Text returned by the configured model without filesystem verification."""

    text: str
