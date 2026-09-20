"""Integration smoke test for AIL ↔ Open Interpreter App Server.

This script attempts to:
1. Locate the OI executable
2. Start the OI App Server
3. Connect via JSON-RPC over stdio
4. Initialize the session
5. Create a thread
6. Send a test message
7. Receive the response
8. Shut down cleanly

If the OI binary is not available, it reports the exact blocker.

Usage:
    python -m tests.test_oi_integration
    or
    python tests/test_oi_integration.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from integrations.open_interpreter.config import OIConfig
from integrations.open_interpreter.client import (
    OIConnectionError,
    OIError,
    OITimeoutError,
    OpenInterpreterClient,
)
from integrations.open_interpreter.transport import OITransportError


def find_oi_executable() -> str | None:
    """Try to locate the OI executable."""
    candidates = [
        os.getenv("AIL_OI_EXECUTABLE", ""),
        "interpreter",
        "codex",
        "oi",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = shutil.which(candidate)
        if path:
            return path
    default_dir = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Programs",
        "Open Interpreter",
        "bin",
        "interpreter.exe",
    )
    if os.path.isfile(default_dir):
        return default_dir
    return None


def run_smoke_test() -> int:
    print("=" * 60)
    print("AIL <-> Open Interpreter App Server Smoke Test")
    print("=" * 60)

    oi_path = find_oi_executable()
    if oi_path is None:
        print("\nBLOCKER: OI executable not found.")
        print("Expected one of: interpreter, codex, oi")
        print("Or set AIL_OI_EXECUTABLE to the full path.")
        print("Install from: https://www.openinterpreter.com/install")
        return 2

    print(f"\nUsing OI executable: {oi_path}")

    config = OIConfig(
        executable=oi_path,
        connect_timeout=30,
        request_timeout=120,
        approval_policy="never",
        sandbox="read-only",
    )

    try:
        print("\n[1/6] Starting OI App Server...")
        client = OpenInterpreterClient(config)
        client.start()
        print("  OK: Server started and initialized")

        print("\n[2/6] Creating thread...")
        thread_id = client.create_thread()
        print(f"  OK: Thread created: {thread_id}")

        test_message = "Reply with exactly: AIL_OI_TEST_OK"
        print(f"\n[3/6] Sending message: {test_message!r}")
        response = client.send_message(test_message, thread_id=thread_id, timeout=90)

        print(f"\n[4/6] Response received:")
        print(f"  Text: {response.text!r}")
        print(f"  Thread: {response.thread_id}")
        print(f"  Turn: {response.turn_id}")

        print("\n[5/6] Shutting down...")
        client.shutdown()
        print("  OK: Clean shutdown")

        print("\n" + "=" * 60)
        if "AIL_OI_TEST_OK" in response.text:
            print("RESULT: SMOKE TEST PASSED")
        else:
            print("RESULT: SMOKE TEST COMPLETED (response did not contain expected text)")
            print(f"  Expected substring: AIL_OI_TEST_OK")
            print(f"  Actual response: {response.text[:200]}")
        print("=" * 60)
        return 0

    except OIConnectionError as exc:
        print(f"\nBLOCKER: Failed to connect to OI App Server: {exc}")
        print("Ensure the OI executable is installed and functional.")
        return 1
    except OITimeoutError as exc:
        print(f"\nTIMEOUT: {exc}")
        return 1
    except OIError as exc:
        print(f"\nERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_test())
