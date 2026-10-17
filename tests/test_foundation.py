"""Section 19.1: Automated tests for application foundation, config, model interface, and capabilities."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

from agent911.capabilities import (
    Capability,
    CapabilityContext,
    CapabilityRegistrationError,
    CapabilityRegistry,
)
from agent911.config import AgentConfig, ConfigurationError, ModelConfig, redact_secrets
from agent911.evaluation import OutcomeClassifier
from agent911.model_provider import (
    MalformedModelOutputError,
    ModelInterface,
    ModelProviderError,
    ProviderConnectivityStatus,
    extract_json_object,
)
from agent911.models import (
    ActionStatus,
    CapabilityAvailability,
    CapabilityResult,
    ExecutionErrorRecord,
    LiveRunTestRecord,
    LiveTestStatus,
    PlannedAction,
    TaskLifecycleStage,
    TaskOutcome,
    TaskRequirement,
    TaskState,
    VerificationCheckResult,
    VerificationStatus,
)
from agent911.orchestrator import Agent911Orchestrator


class _StubPassCapability(Capability):
    identifier = "stub.pass"
    name = "Stub Pass"
    purpose = "Invalid stub capability"

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        pass


class TestApplicationFoundation(unittest.TestCase):
    """Tests for Section 19.1: Application initialization, configuration, tasks, and capabilities."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp_dir.name)
        self.config = AgentConfig(workspace_root=str(self.workspace))
        self.orchestrator = Agent911Orchestrator(config=self.config)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_01_application_initialization(self) -> None:
        """Test application initialization creates workspace, data_dir, db, and custom capabilities dir."""
        self.assertTrue(self.config.workspace_path.exists())
        self.assertTrue(self.config.data_path.exists())
        self.assertTrue(self.config.db_path.exists())
        self.assertTrue(self.config.capabilities_path.exists())
        self.assertGreaterEqual(len(self.orchestrator.registry.list_capabilities()), 17)

    def test_02_configuration_loading_and_secret_redaction(self) -> None:
        """Test loading valid configuration from JSON file and redacting secrets."""
        cfg_file = self.workspace / "custom.config.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "workspace_root": str(self.workspace),
                    "max_recovery_attempts_per_step": 3,
                    "command_timeout_seconds": 15.0,
                    "model": {
                        "provider": "openai_compatible",
                        "model_id": "gpt-4o-mini",
                        "timeout_seconds": 12.0,
                        "max_retries": 1,
                        "temperature": 0.2,
                    },
                }
            ),
            encoding="utf-8",
        )
        loaded = AgentConfig.load(config_path=cfg_file)
        self.assertEqual(loaded.max_recovery_attempts_per_step, 3)
        self.assertEqual(loaded.command_timeout_seconds, 15.0)
        self.assertEqual(loaded.model.timeout_seconds, 12.0)

        # Verify secret redaction (including OpenAI sk-, Anthropic ant-, and Groq gsk_ keys)
        sensitive = {
            "api_key": "sk-secret1234567890abcdef",
            "message": "Auth header Bearer token_abcdef123456 and key gsk_testgroqsecret1234567890 used.",
        }
        redacted = redact_secrets(sensitive, ["token_abcdef123456"])
        self.assertEqual(redacted["api_key"], "[REDACTED]")
        self.assertNotIn("token_abcdef123456", redacted["message"])
        self.assertNotIn("gsk_testgroqsecret1234567890", redacted["message"])
        self.assertIn("[REDACTED_KEY]", redacted["message"])

    def test_03_missing_or_invalid_configuration(self) -> None:
        """Test handling of missing config file, invalid JSON, unknown keys, and invalid values."""
        with self.assertRaises(ConfigurationError):
            AgentConfig.load(config_path=self.workspace / "nonexistent.json")

        bad_json = self.workspace / "bad.json"
        bad_json.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            AgentConfig.load(config_path=bad_json)

        unknown_key_json = self.workspace / "unknown.json"
        unknown_key_json.write_text(json.dumps({"bogus_setting": 123}), encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            AgentConfig.load(config_path=unknown_key_json)

        with self.assertRaises(ConfigurationError):
            AgentConfig(workspace_root=str(self.workspace), command_timeout_seconds=-5).validate()

        with self.assertRaises(ConfigurationError):
            ModelConfig(provider="unknown_provider").validate()

        with self.assertRaises(ConfigurationError):
            ModelConfig(temperature=3.5).validate()

    def test_04_task_creation_and_instruction_preservation(self) -> None:
        """Test task creation, state updates, serialization, and exact instruction preservation."""
        original = "  Create a file 'notes.txt' with content 'Hello Agent 911'  "
        task = self.orchestrator.create_task(original)
        self.assertEqual(task.original_instruction, original)
        self.assertEqual(task.current_execution_state, TaskLifecycleStage.CREATED)

        # Verify persisted in SQLite and round-trip deserializable
        reloaded = self.orchestrator.memory_store.load_task(task.task_id)
        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        self.assertEqual(reloaded.original_instruction, original)
        self.assertEqual(reloaded.task_id, task.task_id)

        with self.assertRaises(ValueError):
            self.orchestrator.create_task("   ")

    def test_05_actionable_planning_output(self) -> None:
        """Test planning produces structured intent, requirements, and actionable steps."""
        task = self.orchestrator.create_task(
            "Write a Python file 'calc.py' with a function 'compute_total'."
        )
        ctx = self.orchestrator.context_manager.retrieve_task_context(task.original_instruction)
        self.orchestrator.planner.understand_and_plan(task, ctx)

        self.assertIsNotNone(task.intent)
        self.assertGreaterEqual(len(task.planned_actions), 1)
        self.assertEqual(task.planned_actions[0].capability_id, "fs.write")
        self.assertEqual(task.planned_actions[0].parameters["path"], "calc.py")
        req_categories = {r.category for r in task.requirements}
        self.assertIn("file_exists", req_categories)
        self.assertIn("code_compiles", req_categories)
        self.assertIn("function_exists", req_categories)

    def test_06_capability_registration_and_discovery(self) -> None:
        """Test capability registration rejects empty stubs and supports discovery."""
        reg = CapabilityRegistry(self.config)
        with self.assertRaises(CapabilityRegistrationError):
            reg.register(_StubPassCapability())

        # Discover built-in capabilities
        fs_caps = self.orchestrator.registry.find_capabilities("file")
        self.assertGreater(len(fs_caps), 0)
        self.assertEqual(
            self.orchestrator.registry.get_availability("fs.write"),
            CapabilityAvailability.AVAILABLE,
        )
        self.assertEqual(
            self.orchestrator.registry.get_availability("does.not.exist"),
            CapabilityAvailability.NOT_IMPLEMENTED,
        )

    def test_07_capability_invocation_and_error_propagation(self) -> None:
        """Test real invocation of filesystem, project inspection, and command capabilities, plus error handling."""
        ctx = CapabilityContext(
            config=self.config,
            memory_store=self.orchestrator.memory_store,
            model_interface=self.orchestrator.model_interface,
            registry=self.orchestrator.registry,
            task_id="test-task",
        )

        # Write a file
        w_res = self.orchestrator.registry.invoke(
            "fs.write", {"path": "hello.txt", "content": "world\n"}, ctx
        )
        self.assertTrue(w_res.success)
        self.assertEqual(w_res.data["bytes_written"], 6)

        # Read it back
        r_res = self.orchestrator.registry.invoke("fs.read", {"path": "hello.txt"}, ctx)
        self.assertTrue(r_res.success)
        self.assertEqual(r_res.data["content"], "world\n")

        # Update file
        u_res = self.orchestrator.registry.invoke(
            "fs.update",
            {"path": "hello.txt", "old_text": "world", "new_text": "agent911"},
            ctx,
        )
        self.assertTrue(u_res.success)

        # Stat file
        s_res = self.orchestrator.registry.invoke("fs.stat", {"path": "hello.txt"}, ctx)
        self.assertTrue(s_res.success)
        self.assertTrue(s_res.data["is_file"])

        # Workspace boundary violation must fail with WorkspaceBoundaryError
        escape_res = self.orchestrator.registry.invoke(
            "fs.read", {"path": "../../etc/passwd"}, ctx
        )
        self.assertFalse(escape_res.success)
        self.assertEqual(escape_res.error_type, "WorkspaceBoundaryError")

        # Invalid input parameter type must fail validation
        bad_param_res = self.orchestrator.registry.invoke(
            "fs.read", {"path": 12345}, ctx
        )
        self.assertFalse(bad_param_res.success)
        self.assertEqual(bad_param_res.error_type, "InputValidationError")

    def test_08_task_outcome_classification_states(self) -> None:
        """Verify all 5 TaskOutcome states: SUCCESS, PARTIAL_SUCCESS, FAILED, BLOCKED, UNVERIFIED."""
        # 1. SUCCESS
        t_succ = TaskState(task_id="t1", original_instruction="do x")
        t_succ.planned_actions = [
            PlannedAction(step_id="s1", description="d", capability_id="fs.write", status=ActionStatus.COMPLETED)
        ]
        t_succ.requirements = [
            TaskRequirement(requirement_id="r1", description="req", category="file_exists", satisfied=True)
        ]
        t_succ.verification_results = [
            VerificationCheckResult(
                check_id="v1", task_id="t1", requirement="req", method="fs", observed_result="ok", status=VerificationStatus.PASSED
            )
        ]
        t_succ.live_run_results = [
            LiveRunTestRecord(
                test_id="l1", task_id="t1", executed_operation="op", target_component="c", environment={}, exit_code=0, stdout="ok", stderr="", expected_behavior_occurred=True, status=LiveTestStatus.PASSED
            )
        ]
        outcome, _ = OutcomeClassifier.classify(t_succ)
        self.assertEqual(outcome, TaskOutcome.SUCCESS)

        # 2. PARTIAL_SUCCESS
        t_part = TaskState(task_id="t2", original_instruction="do x and y")
        t_part.requirements = [
            TaskRequirement(requirement_id="r1", description="req1", category="file_exists", satisfied=True),
            TaskRequirement(requirement_id="r2", description="req2", category="file_exists", satisfied=False),
        ]
        t_part.verification_results = [
            VerificationCheckResult(
                check_id="v1", task_id="t2", requirement="req1", method="fs", observed_result="ok", status=VerificationStatus.PASSED
            ),
            VerificationCheckResult(
                check_id="v2", task_id="t2", requirement="req2", method="fs", observed_result="missing", status=VerificationStatus.FAILED
            ),
        ]
        t_part.live_run_results = [
            LiveRunTestRecord(
                test_id="l1", task_id="t2", executed_operation="op", target_component="c", environment={}, exit_code=0, stdout="ok", stderr="", expected_behavior_occurred=True, status=LiveTestStatus.PASSED
            )
        ]
        outcome, _ = OutcomeClassifier.classify(t_part)
        self.assertEqual(outcome, TaskOutcome.PARTIAL_SUCCESS)

        # 3. FAILED
        t_fail = TaskState(task_id="t3", original_instruction="do fail")
        t_fail.requirements = [
            TaskRequirement(requirement_id="r1", description="req1", category="file_exists", satisfied=False)
        ]
        t_fail.verification_results = [
            VerificationCheckResult(
                check_id="v1", task_id="t3", requirement="req1", method="fs", observed_result="missing", status=VerificationStatus.FAILED
            )
        ]
        outcome, _ = OutcomeClassifier.classify(t_fail)
        self.assertEqual(outcome, TaskOutcome.FAILED)

        # 4. BLOCKED
        t_blk = TaskState(task_id="t4", original_instruction="query blocked api")
        t_blk.errors = [
            ExecutionErrorRecord(
                error_id="e1", task_id="t4", step_id="s1", capability_id="model.query", error_type="CapabilityUnavailableError", message="Missing API key", recovered=False
            )
        ]
        outcome, _ = OutcomeClassifier.classify(t_blk)
        self.assertEqual(outcome, TaskOutcome.BLOCKED)

        # 5. UNVERIFIED
        t_unv = TaskState(task_id="t5", original_instruction="unverified task")
        t_unv.verification_results = [
            VerificationCheckResult(
                check_id="v1", task_id="t5", requirement="req1", method="ai", observed_result="maybe", status=VerificationStatus.INCONCLUSIVE
            )
        ]
        outcome, _ = OutcomeClassifier.classify(t_unv)
        self.assertEqual(outcome, TaskOutcome.UNVERIFIED)

    def test_09_model_interface_http_and_error_handling(self) -> None:
        """Test ModelInterface against a local HTTP server for 200 OK, 401 Auth Error, and malformed output."""
        state = {"mode": "ok"}

        class MockHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                if state["mode"] == "auth_error":
                    self.send_response(401)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": "invalid_api_key"}')
                    return
                if state["mode"] == "malformed":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {"choices": [{"message": {"content": "not a json object"}}]}
                        ).encode("utf-8")
                    )
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "choices": [
                                {"message": {"content": '{"status": "live_http_ok"}'}}
                            ],
                            "usage": {"total_tokens": 12},
                        }
                    ).encode("utf-8")
                )

        server = HTTPServer(("127.0.0.1", 0), MockHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            m_cfg = ModelConfig(
                provider="openai_compatible",
                model_id="test-model",
                endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
                api_key="sk-testkey123456789",
                max_retries=0,
            )
            iface = ModelInterface(m_cfg)

            # 200 OK JSON response
            resp = iface.generate("sys", "user", expect_json=True, use_cache=False)
            self.assertTrue(resp.provider_reached)
            self.assertFalse(resp.is_simulated)
            self.assertEqual(resp.structured_data, {"status": "live_http_ok"})

            # Connectivity report
            conn = iface.check_connectivity()
            self.assertEqual(conn.status, ProviderConnectivityStatus.CONNECTED)
            self.assertTrue(conn.provider_reached)

            # 401 Auth Error
            state["mode"] = "auth_error"
            with self.assertRaises(ModelProviderError) as ctx_err:
                iface.generate("sys", "user_auth", use_cache=False)
            self.assertEqual(ctx_err.exception.error_type, "auth_error")

            # Malformed JSON output when expect_json=True
            state["mode"] = "malformed"
            with self.assertRaises(MalformedModelOutputError):
                iface.generate("sys", "user_bad_json", expect_json=True, use_cache=False)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
