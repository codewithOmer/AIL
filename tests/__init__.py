"""AIL test package.

Establishes the repository root on ``sys.path`` so that every test module can
import ``core``, ``interfaces``, ``integrations``, ``memory`` and ``tools``
regardless of how the test is invoked.  This is the single place that owns the
test-root bootstrap; individual test modules must not duplicate it.
"""

from pathlib import Path
import sys

AIL_ROOT = Path(__file__).resolve().parents[1]

if str(AIL_ROOT) not in sys.path:
    sys.path.insert(0, str(AIL_ROOT))
