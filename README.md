# AIL

**AIL** is an experimental assistant that plans a task, executes it, and then
**independently verifies the result against the real filesystem** before it
reports success.

Its defining property is that the executor is never trusted. AIL does not ask
the model "did that work?" — it inspects the filesystem itself and decides
pass or fail from that evidence alone.

```
                 ┌──────────┐
   user message →│  intent  │ memory write, or task execution
                 └────┬─────┘
                      ▼
     recall memories → PLAN → EXECUTE → VERIFY → RECOVER → (REPLAN) → report
                        │        │         │         │
                     planner  Open      AIL-owned  bounded
                              Interpreter filesystem  retry
                              App Server verifier
```

## Current capabilities (actually implemented)

| Capability | Status | Where |
|---|---|---|
| **Planning** | Implemented. Deterministic planner for one supported task shape (create a file in the workspace), plus plan validation (dependency/cycle checks) and one corrective replan. | `core/planner.py` |
| **Execution** | Implemented. Steps run in order, gated on dependency success, fail-fast. | `core/plan_runner.py`, `core/actions.py` |
| **Verification** | Implemented. Filesystem existence + content checks, independent of the executor. | `tools/fs_verifier.py` |
| **Recovery** | Implemented. Bounded retry loop; the strategy only rewrites the next message, never invents commands. | `core/recovery.py` |
| **Memory** | Implemented. Recall relevant memories before each turn; store a small set of explicit user facts after. JSON-persisted so it survives restarts. | `memory/`, `integrations/personalai/memory.py` |
| **Intent routing** | Implemented. Classifies a message as a memory write or a task. | `core/intent_router.py` |
| **Voice** | Implemented. One fixed-window turn: microphone → Whisper STT → AIL → Edge TTS → speaker. | `core/voice.py`, `integrations/audio/`, `integrations/tts/` |
| **Image input** | Partial. A validated local image path can be passed to the executor. There is **no** image understanding, analysis, or model-based captioning. | `interfaces/image.py` |
| **Open Interpreter adapter** | Implemented. Subprocess lifecycle, JSON-RPC over stdio, streamed turn completion. | `integrations/open_interpreter/` |
| **MCP** | Config only. `MCPServerConfig` entries are forwarded to the executor's `thread/start`. There is **no** MCP client in AIL. | `integrations/open_interpreter/config.py` |

### Not implemented

Do not expect any of the following. None of it exists in this repository:

- Full computer use / GUI automation
- Scheduling, reminders, or background jobs
- Device control (phone, smart home, IoT)
- Browser automation
- Real vision — no image understanding, only image *pass-through*
- A real LLM provider — `Config.llm_*` is read but unused, and `MockLLM` is the
  only `LLM` implementation
- Multi-agent orchestration, or long-horizon / open-ended goals

AIL currently supports exactly **one** production task shape: creating a file
in the workspace. Any other goal raises `UnsupportedTaskError`.

## Development status

Early, pre-1.0, single-developer. The plan → execute → verify → recover loop,
memory, voice, and the Open Interpreter adapter are working and unit-tested.
The planner is deliberately narrow. The repository layout has not yet been
organised by feature; that work is planned but not started.

## Requirements

- **Python 3.12+** (the `personal-ai` contracts package requires >= 3.12;
  the current `.venv` is 3.13.4)
- The **Open Interpreter** executable on `PATH` for `oi` mode
- Network access for Edge TTS and for the Open Interpreter/Codex backend

## Setup

```bash
git clone <repo-url>
cd AIL
python -m venv .venv
```

Activate it:

```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# bash
source .venv/bin/activate
```

### Install dependencies

```bash
# text / OI task mode
pip install -r requirements.txt

# + voice mode
pip install -r requirements.txt -r requirements-voice.txt
```

`requirements.txt` holds the runtime set; `requirements-voice.txt` holds the
voice-only extras (faster-whisper, sounddevice, edge-tts, playsound3) and is
meant to be installed *on top of* the main file.

### Configure

```bash
cp .env.example .env     # PowerShell: Copy-Item .env.example .env
```

`.env` is gitignored. Every variable is optional and falls back to the default
shown in `.env.example`; you only need to set what you want to change. Never
commit `.env` — use `.env.example` as the template.

## Running

AIL dispatches on the first command-line argument.

```bash
# Mock mode - no external services, echoes input. Good for a sanity check.
python main.py

# OI mode - interactive loop against a real Open Interpreter App Server.
python main.py oi

# Voice mode - one fixed-window microphone -> AIL -> speaker turn.
python main.py voice
```

`oi` mode needs `AIL_OI_EXECUTABLE` to point at the `interpreter` binary if it
is not on `PATH`. `voice` mode needs the voice dependencies installed.

## Tests

The suite is standard-library `unittest` — no test runner dependency to install.

```bash
# everything
python -m unittest discover
```

The two real end-to-end tests reach a live Open Interpreter App Server and
therefore need the upstream backend to be reachable and to have quota left.
They skip themselves if the `interpreter` executable is not found. To run only
the offline suite:

```bash
python -m unittest \
  tests.test_agent tests.test_application \
  tests.test_application_intent_routing tests.test_application_memory \
  tests.test_image tests.test_intent_router tests.test_main \
  tests.test_memory tests.test_memory_flow tests.test_memory_persistence \
  tests.test_oi_adapter tests.test_planning tests.test_plan_runner \
  tests.test_recovery tests.test_transcription tests.test_verification \
  tests.test_voice
```

## External dependencies

**Sibling `personal-ai` repository.** `integrations/personalai/memory.py`
imports `personalai_contracts` from a sibling checkout at
`../personal-ai/contracts/src`, injecting it onto `sys.path` at import time.
That path is currently hardcoded and there is no fallback. Consequences:

- The repository will not import without that sibling checkout present.
- Its own dependency (`pydantic>=2.7`) is declared in `requirements.txt`,
  because AIL imports it transitively at runtime.
- It also pulls `InMemoryMemoryStore` and `FakeModelProvider` from
  `personalai_contracts.testing` — test doubles from an external repo are used
  as production components. This is a known design debt, not an oversight in
  the setup.

**Open Interpreter / Codex.** The executor is a separate application. The real
E2E tests fail with `usageLimitExceeded` when the upstream account is out of
quota. That is an external account limit, not an AIL defect.
