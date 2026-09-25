"""Default application intent routing."""

from __future__ import annotations

from interfaces.intent import IntentKind, IntentResult, IntentRouter
from memory.flow import extract_explicit_facts


class DefaultIntentRouter(IntentRouter):
    """Recognize only the existing explicit memory forms."""

    def route(self, message: str) -> IntentResult:
        if extract_explicit_facts(message):
            return IntentResult(IntentKind.MEMORY, "explicit-memory")
        return IntentResult(IntentKind.TASK, "not-explicit-memory")
