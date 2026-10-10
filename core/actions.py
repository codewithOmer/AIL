"""AIL action composition: execute through Open Interpreter, then verify.

Deliberately thin.  Open Interpreter is only the executor; verification is
AIL-owned and completely independent of the executor's output.

Recovery is deliberately not handled here.  A single ``execute_and_verify``
call is one attempt; the bounded execute → verify → recover loop lives in
``core.recovery.execute_with_recovery``, which this module sits underneath.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from interfaces.image import LocalImage
from interfaces.verification import VerificationResult

if TYPE_CHECKING:
    from integrations.open_interpreter.client import OIResponse


class VerificationError(RuntimeError):
    """The verification stage failed to produce a verification result."""


def execute_and_verify(
    client: Any,
    verifier: Any,
    message: str,
    expectations: Any,
    thread_id: str | None = None,
    base_dir: str = ".",
    timeout: float | None = None,
    images: Sequence[LocalImage] | None = None,
) -> tuple[OIResponse, VerificationResult]:
    """Execute *message* via *client*, then independently verify the outcome.

    1. Runs the operation through the client's ``send_message``.
    2. Ignores the OI response text for verification purposes.
    3. Verifies the real filesystem against *expectations* with *verifier*.
    4. Returns both the OI response and the ``VerificationResult``.
    """
    if timeout is None:
        if images:
            response = client.send_message(
                message,
                thread_id=thread_id,
                images=images,
            )
        else:
            response = client.send_message(message, thread_id=thread_id)
    else:
        kwargs: dict[str, Any] = {"thread_id": thread_id, "timeout": timeout}
        if images:
            kwargs["images"] = images
        response = client.send_message(message, **kwargs)
    try:
        result = verifier.verify(expectations, base_dir)
    except Exception as exc:
        raise VerificationError(f"Verification failed: {exc}") from exc
    return response, result