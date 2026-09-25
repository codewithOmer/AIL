"""Helper functions for the AIL memory flow around Open Interpreter.

These pure helpers implement:
1. building a clearly-labeled context block from recalled memories
2. injecting that context into the message that is sent to OI
3. detecting a very small set of explicit user-provided facts
4. storing facts only when they are new

This module has no OI or PersonalAI dependency, so it is easy to remove or
replace once proper persistent memory is added.
"""

from __future__ import annotations

import re

from memory.interface import Memory, MemoryStore

_MEMORY_HEADER = "[Memory context from previous conversations:]"

_NAME_RE = re.compile(
    r"(?i:my name is)\s+(?P<fact>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[.!?]?\Z"
)
_REMEMBER_RE = re.compile(
    r"(?i:remember(?:\s+that)?\s+)(?P<fact>.+?)[.!?]?\Z"
)


def _normalize_standalone_fact(fact: str) -> str | None:
    """Normalize one trailing mark and reject clear clause boundaries."""
    normalized = fact.strip()
    if normalized.lower() == "that":
        return None
    if normalized and normalized[-1] in ".!?":
        normalized = normalized[:-1].rstrip()
    if not normalized or normalized[-1:] in ".!?":
        return None
    if re.search(r"[.!?]\s+\S|;|:\s|\s-\s", normalized):
        return None
    return normalized


def build_context_block(recalled: list[Memory]) -> str:
    """Render recalled memories as a labeled context block, or '' if none."""
    if not recalled:
        return ""
    lines = "\n".join(f"- {m.text}" for m in recalled)
    return f"{_MEMORY_HEADER}\n{lines}"


def apply_context(message: str, recalled: list[Memory]) -> str:
    """Prepend a labeled memory-context block to *message* when memories exist."""
    block = build_context_block(recalled)
    if not block:
        return message
    return f"{block}\n\n{message}"


def extract_explicit_facts(message: str) -> list[str]:
    """Detect only a small set of explicit user-provided facts.

    Supported today:
      * "my name is <name>"        -> "The user's name is <name>."
      * "remember (that) <fact>"   -> "<fact>."

    Anything else returns an empty list. No general LLM extractor yet.
    """
    facts: list[str] = []

    normalized_message = message.strip()
    name_match = _NAME_RE.fullmatch(normalized_message)
    if name_match:
        name = _normalize_standalone_fact(name_match.group("fact"))
        if name is not None:
            facts.append(f"The user's name is {name}.")

    remember_match = _REMEMBER_RE.fullmatch(normalized_message)
    if remember_match:
        fact = _normalize_standalone_fact(remember_match.group("fact"))
        if fact is not None:
            facts.append(f"{fact}.")

    return facts


def store_if_new(store: MemoryStore, text: str) -> Memory | None:
    """Store *text* only if an identical memory does not exist yet."""
    for existing in store.list_all():
        if existing.text == text:
            return None
    return store.store(text)