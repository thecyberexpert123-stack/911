# Agent 911

**Agent 911** is a complete, from-scratch, instruction-driven, self-expanding AI agent designed around evidence-based task execution, persistent SQLite memory and experience, targeted failure recovery, organic capability growth, and exactly two formal task-evaluation mechanisms:

1. **Verification** (`agent911.evaluation.verification.Verifier`)
2. **Live Run Testing** (`agent911.evaluation.live_testing.LiveRunTester`)

---

## 1. Architectural Overview

Agent 911 executes tasks through a unified, direct lifecycle without any separate Guardian or policy-approval subsystem:

```text
USER INSTRUCTION
  │
  ▼
INTENT UNDERSTANDING & CONTEXT / MEMORY RETRIEVAL
  │
  ▼
TASK REPRESENTATION & EXPERIENCE-INFORMED PLANNING
  │
  ▼
CAPABILITY DISCOVERY & SELECTION (including Capability Gap Detection)
  │
  ▼
ACTUAL EXECUTION (with Error Diagnosis & Targeted Recovery)
  │
  ▼
FORMAL EVALUATION MECHANISM 1: VERIFICATION
  │
  ▼
FORMAL EVALUATION MECHANISM 2: LIVE RUN TESTING
  │
  ▼
RESULT CLASSIFICATION (SUCCESS | PARTIAL_SUCCESS | FAILED | BLOCKED | UNVERIFIED)
  │
  ▼
EXPERIENCE & PERSISTENT MEMORY UPDATE
```

### Principal Modules

- `agent911/models.py`: Structured data models (`TaskState`, `InstructionIntent`, `TaskRequirement`, `PlannedAction`, `CapabilityResult`, `VerificationCheckResult`, `LiveRunTestRecord`, `ExecutionErrorRecord`, `RecoveryAttemptRecord`, `MemoryRecord`, `ExperienceRecord`) and formal status enums.
- `agent911/config.py`: Configuration loader (`AgentConfig`, `ModelConfig`), validator (`ConfigurationError`), and recursive secret redaction (`redact_secrets`).
- `agent911/memory.py`: SQLite-backed `PersistentMemoryStore` managing five logical memory categories (`FACTUAL`, `INSTRUCTION`, `EPISODIC`, `EXPERIENCE`, `WORKING_STATE`), experience records, persisted task states, redacted execution logs, and custom capability metadata.
- `agent911/context.py`: `ContextManager` for selective memory/experience retrieval, context-window enforcement, context compression/summarization (`compress_context`), and interrupted-task resumption validation (`prepare_task_for_resumption`).
- `agent911/model_provider.py`: `ModelInterface` supporting OpenAI-compatible (`/v1/chat/completions`), Anthropic (`/v1/messages`), Ollama (`/api/chat`), and deterministic test doubles (`deterministic`), with honest connectivity reporting (`check_connectivity`).
- `agent911/capabilities/`:
  - `base.py` & `registry.py`: `Capability` contract, `CapabilityContext`, and `CapabilityRegistry` (rejecting empty/stub classes, validating inputs, checking availability, and dynamically loading custom capabilities).
  - `filesystem.py`: `fs.list`, `fs.read`, `fs.write`, `fs.update`, `fs.mkdir`, `fs.stat` with strict workspace boundary enforcement (`resolve_in_workspace`).
  - `project_inspection.py`: `project.inspect` discovering project files, source directories, configuration files, documentation, tests, package manifests, entry points, dependencies, and Python AST symbols.
  - `command_execution.py`: `cmd.run` executing development commands with timeout, workspace `cwd`, exit code, `stdout`, and `stderr` capture.
  - `test_execution.py`: `test.run` executing `unittest` or `pytest` test suites and parsing actual test counts and outcomes.
  - `memory_ops.py` & `experience_ops.py`: `memory.store`, `memory.retrieve`, `memory.update`, `experience.record`, `experience.retrieve`.
  - `model_ops.py`: `capability.discover`, `model.connectivity`, `model.query`.
  - `growth.py`: `capability.develop` for organic capability synthesis, AST/interface verification, live runtime testing, dynamic registration, and SQLite persistence.
- `agent911/planner.py`: `TaskPlanner` combining AI-model planning (when connected) with a local natural-language instruction parser, capability gap detection, and automatic application of remedies from past experiences.
- `agent911/recovery.py`: `RecoveryEngine` diagnosing step failures, selecting targeted corrections (e.g., creating missing parent directories, developing missing capabilities, initializing missing files), re-executing only the affected step, and preventing retry loops via retry caps and failure-signature tracking.
- `agent911/evaluation/`:
  - `verification.py`: `Verifier` (`PASSED`, `FAILED`, `INCONCLUSIVE`, `UNAVAILABLE`).
  - `live_testing.py`: `LiveRunTester` (`PASSED`, `FAILED`, `UNAVAILABLE`, `TIMED_OUT`).
  - `classifier.py`: `OutcomeClassifier` (`SUCCESS`, `PARTIAL_SUCCESS`, `FAILED`, `BLOCKED`, `UNVERIFIED`).
- `agent911/orchestrator.py`: `Agent911Orchestrator` coordinating the end-to-end lifecycle and task resumption.
- `agent911/cli.py` & `agent911/__main__.py`: Command-line interface (`agent911` / `python3 -m agent911`).
- `agent911/bridge.py` & `ui/`: Node.js Chat & Operations Console (`ui/server.js`, `ui/public/index.html`, `ui/public/styles.css`, `ui/public/app.js`, `ui/test_ui.js`) bridged directly to the Python `Agent911Orchestrator`.

---

## 2. Requirements & Installation

### Required Runtime & Dependencies

- **Python:** Python `>= 3.10` (tested on Python `3.11.2`).
- **Runtime Dependencies:** Zero mandatory third-party packages; built on Python's standard library (`sqlite3`, `urllib.request`, `subprocess`, `ast`, `importlib`, `unittest`, `argparse`, `dataclasses`, `json`).
- **Optional Development Dependencies:** `pytest >= 7.0.0` (standard library `unittest` is also fully supported).

### Installation Commands

From the repository root (`/home/user/911`):

```bash
# Editable installation (creates the `agent911` CLI script)
pip install -e .

# Or on Debian/PEP-668 managed environments without a virtualenv:
pip install --break-system-packages --no-build-isolation -e .
```

You can also invoke Agent 911 directly without installing via `python3 -m agent911`.

---

## 3. Multi-Provider & Multi-Model Configuration System

### Configuration Files, Schema v2 & Credential Store

Copy the example configuration file if you want a local configuration file:

```bash
cp agent911.config.example.json agent911.config.local.json
```

Agent 911 uses a versioned configuration schema (`schema_version = 2`, backwards-compatible with v1) supporting:
- **Multiple Providers (`providers` registry):** Configure, edit, enable/disable, and remove multiple providers (`openai_compatible`, `anthropic`, `ollama`, `local_only`, `deterministic`) with dedicated `ProviderAdapter` implementations.
- **Multiple Model Profiles (`models` registry):** Configure multiple models (including multiple models per provider) with human-readable name, provider ID, model ID, optional endpoint override, context window (`null` when unknown), supported inputs, tool-calling/structured-output/streaming support (`true`/`false`/`"unknown"`), temperature, max tokens, timeout, and retry policy.
- **Active, Default, and Fallback Model Selection:** Switch the `active_model` at runtime without restarting, persist `default_model` across restarts, and configure an optional `fallback_model` for automatic recovery when a primary provider fails.
- **Secure Credential Management (`CredentialStore`):** API keys are never written into `agent911.config.local.json` or allowed in endpoint URLs. Secrets are resolved from environment variables (`AGENT911_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or provider-specific `api_key_env`) or stored in `.agent911/credentials.json` with `0600` file permissions and displayed only as masked previews (`sk-...3210`).

### Supported Environment Variables

- `AGENT911_API_KEY` (or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`): API credential for the configured model provider. Never commit API keys to source control.
- `AGENT911_WORKSPACE`: Override `workspace_root`.
- `AGENT911_DB_PATH`: Override SQLite database path.
- `AGENT911_PROVIDER`: Override active provider adapter (`openai_compatible`, `anthropic`, `ollama`, `deterministic`, `local_only`).
- `AGENT911_MODEL_ID`: Override active model identifier.
- `AGENT911_ENDPOINT`: Override active model endpoint.
- `AGENT911_TIMEOUT`: Override active model timeout seconds.

### Persistent Memory & Workspace Locations

- **Workspace Root (`workspace_root`):** All filesystem (`fs.*`), project inspection (`project.inspect`), command (`cmd.run`), and test (`test.run`) operations resolve paths inside `workspace_root` and reject path traversal outside the workspace with `WorkspaceBoundaryError`.
- **SQLite Database (`database_path`):** Defaults to `<workspace_root>/.agent911/agent911.db`. Stores memories (with `status` `ACTIVE`/`SUPERSEDED`/`OBSOLETE` and `provenance_history`), evidence-backed experiences, task states, redacted execution logs, and custom capability metadata.
- **Credential Vault (`credentials.json`):** Defaults to `<workspace_root>/.agent911/credentials.json` (mode `0600`).
- **Custom Grown Capabilities (`custom_capabilities_dir`):** Defaults to `<workspace_root>/.agent911/capabilities/`.

---

## 4. Launch & Usage Commands

```bash
# Check application configuration, model connectivity, and capability availability
python3 -m agent911 status

# List, add, update, enable/disable, or remove providers
python3 -m agent911 providers
python3 -m agent911 providers --upsert-id anthropic_main --name "Anthropic API" \
  --adapter anthropic --endpoint https://api.anthropic.com/v1/messages \
  --api-key-env ANTHROPIC_API_KEY

# List, add, update, enable/disable, or remove model profiles
python3 -m agent911 models
python3 -m agent911 models --upsert-id claude-sonnet --name "Claude 3.5 Sonnet" \
  --provider-id anthropic_main --model-name claude-3-5-sonnet-latest --context-window 200000

# Switch active model at runtime, persist as default, or configure fallback model
python3 -m agent911 model-select claude-sonnet --set-default --fallback local-planner

# Run a genuine connection test against the active or specified model profile
python3 -m agent911 model-test
python3 -m agent911 model-test --profile local-planner

# List all registered capabilities (built-in and custom grown)
python3 -m agent911 capabilities

# Execute a task instruction
python3 -m agent911 run "Write a Python file 'calculator.py' with a function 'add_values'."

# Develop, verify, live-test, and register a new capability
python3 -m agent911 develop-capability text.metrics \
  --name "Text Metrics Analyzer" \
  --purpose "Count words, lines, and characters in text or files"

# Store and query persistent memory
python3 -m agent911 memory --store-key "project.owner" --store-content "Agent 911 Team"
python3 -m agent911 memory --query "owner"

# List recent tasks and inspect detailed execution/evaluation records
python3 -m agent911 tasks
python3 -m agent911 inspect <task_id>

# Resume an interrupted task
python3 -m agent911 resume <task_id>

# Inspect recorded task experiences and remedies
python3 -m agent911 experiences

# Inspect structured self-diagnosis, strategy history, capability gap analysis, and resource metrics
python3 -m agent911 diagnose <task_id>

# Scan execution history for operational weaknesses and list prioritized ImprovementProposals
python3 -m agent911 proposals --detect

# Execute controlled self-improvement for a proposal (with Verification, Live Run Testing, and automatic rollback)
python3 -m agent911 self-improve <proposal_id>

# Launch the Node.js Chat Interface (binds to 0.0.0.0:3000 by default)
npm start
```

Add `--json` before the subcommand (e.g., `python3 -m agent911 --json status`) for machine-readable JSON output, or `--workspace /path/to/workspace` to target an isolated workspace directory.

---

## 5. Running Automated Tests

Both Python (`unittest` and `pytest`) and Node.js (`node --test`) suites are supported:

```bash
# Run Python test suite with unittest
python3 -m unittest discover -s tests -v

# Run Python test suite with pytest
pytest -v

# Run Node.js Chat UI test suite
npm test
```

---

## 6. Troubleshooting

- **`Model Connectivity: MISSING_CREDENTIALS` / `Task BLOCKED` on `model.query`:**
  No API key was found in `AGENT911_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`. Export a valid API key in your environment to enable live external model queries. Local capabilities (`fs.*`, `project.inspect`, `cmd.run`, `test.run`, `memory.*`, `experience.*`, `capability.develop`) continue to operate when `allow_local_fallback` is `true`.
- **`WorkspaceBoundaryError`:**
  A file or directory path resolved outside the configured `workspace_root`. Pass a relative path within the workspace or adjust `--workspace`.
- **`ConfigurationError`:**
  Check that your JSON config file is valid JSON, contains no unknown keys, and uses positive numbers for timeouts.
