"""AIL action composition: execute through Open Interpreter, then verify.

Deliberately thin.  Open Interpreter is only the executor; verification is
AIL-owned and completely independent of the executor's output.  Recovery is
not implemented yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from interfaces.verification import VerificationResult

if TYPE_CHECKING:
    from integrations.open_interpreter.client import OIResponse


def execute_and_verify(
    client: Any,
    verifier: Any,
    message: str,
    expectations: Any,
    thread_id: str | None = None,
    base_dir: str = ".",
    timeout: float | None = None,
) -> tuple[OIResponse, VerificationResult]:
    """Execute *message* via *client*, then independently verify the outcome.

    1. Runs the operation through the client's ``send_message``.
    2. Ignores the OI response text for verification purposes.
    3. Verifies the real filesystem against *expectations* with *verifier*.
    4. Returns both the OI response and the ``VerificationResult``.
    """
    if timeout is None:
        response = client.send_message(message, thread_id=thread_id)
    else:
        response = client.send_message(
            message,
            thread_id=thread_id,
            timeout=timeout,
        )
    result = verifier.verify(expectations, base_dir)
    return response, result