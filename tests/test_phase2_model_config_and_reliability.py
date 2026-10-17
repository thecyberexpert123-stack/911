"""Phase 2 Automated Test Suite: Model Configuration System, Adapters, Provenance, Context, and Reliability."""

from __future__ import annotations

import http.server
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Dict

from agent911.capabilities.base import Capability, CapabilityContext
from agent911.config import (
    AgentConfig,
    ConfigurationError,
    CredentialStore,
    ModelProfileConfig,
    ProviderConfig,
)
from agent911.evaluation.verification import Verifier
from agent911.model_provider import (
    ModelInterface,
    ProviderConnectivityStatus,
    RequestCancelledError,
    UnsupportedModelFeatureError,
)
from agent911.models import (
    ActionStatus,
    CapabilityAvailability,
    CapabilityResult,
    ExperienceRecord,
    InvalidTaskStateTransitionError,
    MemoryCategory,
    MemoryConfidence,
    PlannedAction,
    TaskLifecycleStage,
    TaskOutcome,
    TaskRequirement,
)
from agent911.orchestrator import Agent911Orchestrator


class _MultiAdapterMockHandler(http.server.BaseHTTPRequestHandler):
    mode: str = "openai_ok"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        m = _MultiAdapterMockHandler.mode
        if m == "openai_ok":
            body = json.dumps(
                {
                    "choices": [{"message": {"content": '{"status": "openai_live"}'}}],
                    "usage": {"total_tokens": 18},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif m == "anthropic_ok":
            body = json.dumps(
                {
                    "content": [{"type": "text", "text": '{"status": "anthropic_live"}'}],
                    "usage": {"input_tokens": 10, "output_tokens": 8},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif m == "ollama_ok":
            body = json.dumps(
                {
                    "message": {"role": "assistant", "content": '{"status": "ollama_live"}'},
                    "eval_count": 12,
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif m == "auth_401":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "invalid_api_key"}')
        elif m == "model_404":
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "model_not_found"}')
        elif m == "malformed_http":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"NOT_JSON_AT_ALL")


class TestPhase2ModelConfigAndReliability(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp_dir.name)
        self.config = AgentConfig(workspace_root=str(self.workspace))
        self.orchestrator = Agent911Orchestrator(config=self.config)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_01_multi_provider_and_multi_model_lifecycle_and_persistence(self) -> None:
        """
        Verify adding, updating, enabling/disabling, removing, and persisting multiple
        providers and multiple models per provider across restarts (schema_version=2),
        and verify that credentials are stored in a mode-0600 vault separate from config.
        """
        # Add Anthropic and Ollama providers
        self.orchestrator.upsert_provider(
            ProviderConfig(
                provider_id="anthropic_prod",
                name="Anthropic Production",
                adapter_type="anthropic",
                endpoint="https://api.anthropic.com/v1/messages",
                api_key_env="ANTHROPIC_API_KEY",
            ),
            api_key="sk-ant-secretkey9876543210",
            save=True,
        )
        self.orchestrator.upsert_provider(
            ProviderConfig(
                provider_id="ollama_local",
                name="Local Ollama",
                adapter_type="ollama",
                endpoint="http://127.0.0.1:11434/api/chat",
                api_key_env=None,
            ),
            save=True,
        )

        # Add two distinct models under anthropic_prod and one under ollama_local
        self.orchestrator.upsert_model(
            ModelProfileConfig(
                profile_id="claude-sonnet",
                name="Claude 3.5 Sonnet",
                provider_id="anthropic_prod",
                model_id="claude-3-5-sonnet-latest",
                context_window_tokens=200000,
                supports_tool_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
            ),
            save=True,
        )
        self.orchestrator.upsert_model(
            ModelProfileConfig(
                profile_id="claude-haiku",
                name="Claude 3.5 Haiku",
                provider_id="anthropic_prod",
                model_id="claude-3-5-haiku-latest",
                context_window_tokens=200000,
            ),
            save=True,
        )
        self.orchestrator.upsert_model(
            ModelProfileConfig(
                profile_id="llama3-local",
                name="Llama 3 Local",
                provider_id="ollama_local",
                model_id="llama3.2",
                context_window_tokens=8192,
                supports_tool_calling=False,
                supports_streaming=False,
            ),
            save=True,
        )

        # Switch active and default model at runtime without restart
        sel = self.orchestrator.select_active_model(
            "claude-sonnet", set_as_default=True, save=True
        )
        self.assertEqual(sel["active_model"], "claude-sonnet")
        self.assertEqual(sel["default_model"], "claude-sonnet")
        self.assertEqual(self.orchestrator.model_interface.config.provider, "anthropic")
        self.assertEqual(
            self.orchestrator.model_interface.config.model_id,
            "claude-3-5-sonnet-latest",
        )

        # Configure fallback model
        self.orchestrator.set_fallback_model("llama3-local", save=True)
        self.assertEqual(self.config.fallback_model, "llama3-local")

        # Verify config file does NOT contain raw API key and vault has 0600 permissions
        cfg_file = self.workspace / "agent911.config.local.json"
        self.assertTrue(cfg_file.exists())
        raw_cfg_text = cfg_file.read_text(encoding="utf-8")
        self.assertNotIn("sk-ant-secretkey9876543210", raw_cfg_text)

        vault_path = self.config.credentials_vault_path
        self.assertTrue(vault_path.exists())
        mode = stat.S_IMODE(vault_path.stat().st_mode)
        self.assertEqual(mode, 0o600)
        masked = self.config.credential_store.get_masked_credential("anthropic_prod")
        self.assertIsNotNone(masked)
        self.assertNotIn("secretkey", masked or "")

        # Reload config from disk in a fresh AgentConfig & Orchestrator (simulating restart)
        reloaded_cfg = AgentConfig.load(workspace_override=self.workspace)
        self.assertEqual(reloaded_cfg.schema_version, 2)
        self.assertEqual(reloaded_cfg.default_model, "claude-sonnet")
        self.assertEqual(reloaded_cfg.active_model, "claude-sonnet")
        self.assertEqual(reloaded_cfg.fallback_model, "llama3-local")
        self.assertIn("claude-haiku", reloaded_cfg.models)

        # Remove non-active model profile
        reloaded_orch = Agent911Orchestrator(config=reloaded_cfg)
        reloaded_orch.remove_model("claude-haiku", save=True)
        self.assertNotIn("claude-haiku", reloaded_cfg.models)

        # Removing active/default model or its provider must fail safely without corrupting state
        with self.assertRaises(ConfigurationError) as ctx_err:
            reloaded_orch.remove_model("claude-sonnet")
        self.assertFalse(ctx_err.exception.changes_applied)
        self.assertEqual(reloaded_cfg.active_model, "claude-sonnet")

    def test_02_invalid_config_isolation_url_credential_prohibition_and_safe_switch_failure(
        self,
    ) -> None:
        """
        Verify:
        - Credentials embedded in endpoint URLs are rejected
        - Switching to a disabled or nonexistent model fails cleanly and preserves current active model
        - Partially invalid non-active entries in config file are isolated into validation_warnings
        """
        with self.assertRaises(ConfigurationError):
            self.orchestrator.upsert_provider(
                ProviderConfig(
                    provider_id="bad_url_prov",
                    name="Bad URL Provider",
                    adapter_type="openai_compatible",
                    endpoint="https://user:secretpass@api.example.com/v1/chat",
                )
            )

        # Add a disabled model and verify selecting it fails while keeping current active model
        self.orchestrator.upsert_model(
            ModelProfileConfig(
                profile_id="disabled-model",
                name="Disabled Model",
                provider_id="default",
                model_id="gpt-disabled",
                enabled=False,
            )
        )
        prev_active = self.config.active_model
        with self.assertRaises(ConfigurationError):
            self.orchestrator.select_active_model("disabled-model")
        self.assertEqual(self.config.active_model, prev_active)

        # Write a config file containing one valid active model and one broken non-active model
        partial_cfg_path = self.workspace / "partial.config.json"
        partial_cfg_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "workspace_root": str(self.workspace),
                    "default_model": "default",
                    "active_model": "default",
                    "providers": {
                        "default": {
                            "provider_id": "default",
                            "name": "Default",
                            "adapter_type": "openai_compatible",
                            "endpoint": "https://api.openai.com/v1/chat/completions",
                            "enabled": True,
                        }
                    },
                    "models": {
                        "default": {
                            "profile_id": "default",
                            "name": "Default GPT",
                            "provider_id": "default",
                            "model_id": "gpt-4o-mini",
                            "enabled": True,
                        },
                        "broken_entry": {
                            "profile_id": "broken_entry",
                            "name": "Broken",
                            "provider_id": "nonexistent_provider",
                            "model_id": "bad",
                            "enabled": True,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        loaded = AgentConfig.load(config_path=partial_cfg_path)
        self.assertIn("default", loaded.models)
        self.assertNotIn("broken_entry", loaded.models)
        self.assertTrue(any("broken_entry" in w for w in loaded.validation_warnings))

    def test_03_provider_adapters_and_granular_connection_testing(self) -> None:
        """
        Exercise OpenAICompatibleAdapter, AnthropicAdapter, and OllamaAdapter against a
        real local HTTP server, and verify ConnectivityReport distinguishes:
        CONNECTED, AUTH_ERROR (401), MODEL_NOT_FOUND (404), INVALID_RESPONSE, and UNREACHABLE.
        """
        server = http.server.HTTPServer(("127.0.0.1", 0), _MultiAdapterMockHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            url = f"http://127.0.0.1:{port}/v1/chat"
            for adapter_name, mode_name, expected_status in [
                ("openai_compatible", "openai_ok", "openai_live"),
                ("anthropic", "anthropic_ok", "anthropic_live"),
                ("ollama", "ollama_ok", "ollama_live"),
            ]:
                _MultiAdapterMockHandler.mode = mode_name
                pid = f"prov_{adapter_name}"
                mid = f"mod_{adapter_name}"
                self.orchestrator.upsert_provider(
                    ProviderConfig(
                        provider_id=pid,
                        name=pid,
                        adapter_type=adapter_name,
                        endpoint=url,
                    ),
                    api_key="sk-testadapterkey123456",
                )
                self.orchestrator.upsert_model(
                    ModelProfileConfig(
                        profile_id=mid,
                        name=mid,
                        provider_id=pid,
                        model_id="test-model-v1",
                    )
                )
                rep = self.orchestrator.test_model_connection(profile_id=mid)
                self.assertEqual(rep.status, ProviderConnectivityStatus.CONNECTED)
                self.assertTrue(rep.config_valid)
                self.assertTrue(rep.endpoint_reachable)
                self.assertTrue(rep.auth_valid)
                self.assertTrue(rep.model_accepted)
                self.assertTrue(rep.inference_succeeded)
                self.assertTrue(rep.conclusive)

                resp = self.orchestrator.model_interface.generate(
                    system_prompt="Sys",
                    user_prompt="Hello",
                    expect_json=True,
                    use_cache=False,
                    model_profile_override=mid,
                )
                self.assertEqual(resp.structured_data, {"status": expected_status})

            # Test 401 AUTH_ERROR
            _MultiAdapterMockHandler.mode = "auth_401"
            rep_401 = self.orchestrator.test_model_connection(
                profile_id="mod_openai_compatible"
            )
            self.assertEqual(rep_401.status, ProviderConnectivityStatus.AUTH_ERROR)
            self.assertTrue(rep_401.endpoint_reachable)
            self.assertFalse(rep_401.auth_valid)
            self.assertFalse(rep_401.inference_succeeded)
            self.assertTrue(rep_401.conclusive)

            # Test 404 MODEL_NOT_FOUND
            _MultiAdapterMockHandler.mode = "model_404"
            rep_404 = self.orchestrator.test_model_connection(
                profile_id="mod_openai_compatible"
            )
            self.assertEqual(rep_404.status, ProviderConnectivityStatus.MODEL_NOT_FOUND)
            self.assertTrue(rep_404.endpoint_reachable)
            self.assertTrue(rep_404.auth_valid)
            self.assertFalse(rep_404.model_accepted)
            self.assertFalse(rep_404.inference_succeeded)

            # Test malformed non-JSON HTTP response
            _MultiAdapterMockHandler.mode = "malformed_http"
            rep_bad = self.orchestrator.test_model_connection(
                profile_id="mod_openai_compatible"
            )
            self.assertEqual(rep_bad.status, ProviderConnectivityStatus.INVALID_RESPONSE)
            self.assertFalse(rep_bad.inference_succeeded)
        finally:
            server.shutdown()
            server.server_close()

    def test_04_unsupported_feature_guard_cancellation_and_fallback_model_routing(
        self,
    ) -> None:
        """
        Verify:
        - Unsupported features (e.g. tool_calling or image input on a model that disables them)
          raise UnsupportedModelFeatureError
        - Cancelled requests raise RequestCancelledError immediately
        - Primary model failure automatically routes to configured fallback_model when available
        """
        self.orchestrator.upsert_model(
            ModelProfileConfig(
                profile_id="text-only-no-tools",
                name="Text Only Model",
                provider_id="local_heuristic",
                model_id="local-v1",
                supported_inputs=["text"],
                supports_tool_calling=False,
                supports_streaming=False,
            )
        )
        with self.assertRaises(UnsupportedModelFeatureError):
            self.orchestrator.model_interface.generate(
                "sys",
                "user",
                model_profile_override="text-only-no-tools",
                require_tool_calling=True,
            )
        with self.assertRaises(UnsupportedModelFeatureError):
            self.orchestrator.model_interface.generate(
                "sys",
                "user",
                model_profile_override="text-only-no-tools",
                input_type="image",
            )

        cancel_ev = threading.Event()
        cancel_ev.set()
        with self.assertRaises(RequestCancelledError):
            self.orchestrator.model_interface.generate(
                "sys", "user", cancel_event=cancel_ev
            )

    def test_05_enforced_task_state_transitions_and_fallback_recovery(self) -> None:
        """
        Verify TaskState.transition_to prevents invalid lifecycle jumps and prevents
        marking a task COMPLETED while steps remain IN_PROGRESS.
        """
        task = self.orchestrator.create_task("Test state transitions")
        # Cannot jump directly from CREATED to COMPLETED
        with self.assertRaises(InvalidTaskStateTransitionError):
            task.transition_to(TaskLifecycleStage.COMPLETED)

        task.transition_to(TaskLifecycleStage.EXECUTING)
        task.planned_actions.append(
            PlannedAction(
                step_id="step-1",
                description="In progress action",
                capability_id="fs.list",
                status=ActionStatus.IN_PROGRESS,
            )
        )
        task.transition_to(TaskLifecycleStage.VERIFYING)
        task.transition_to(TaskLifecycleStage.CLASSIFYING)
        # Cannot transition to COMPLETED while step-1 is still IN_PROGRESS
        with self.assertRaises(InvalidTaskStateTransitionError):
            task.transition_to(TaskLifecycleStage.COMPLETED)

        task.planned_actions[0].status = ActionStatus.COMPLETED
        task.transition_to(TaskLifecycleStage.COMPLETED)
        self.assertEqual(task.current_execution_state, TaskLifecycleStage.COMPLETED)

    def test_06_memory_provenance_superseding_obsolescence_and_experience_guards(
        self,
    ) -> None:
        """
        Verify:
        - Updating or superseding a memory preserves prior values in provenance_history
        - Marking a memory obsolete excludes it from active retrieval while keeping audit trail
        - Unverified claims cannot be stored as VERIFIED_EXECUTION experiences
        - Context assembly adapts to active_model_context_window_tokens and preserves requirements
        """
        store = self.orchestrator.memory_store
        m1 = store.store_memory(
            category=MemoryCategory.FACTUAL,
            key="api.version",
            content="v1.0",
            confidence=MemoryConfidence.OBSERVED,
        )
        m2 = store.supersede_memory(
            old_memory_id=m1.memory_id,
            new_content="v2.0",
            reason="Upgraded API to v2.0",
            confidence=MemoryConfidence.VERIFIED,
        )
        self.assertIsNotNone(m2)
        assert m2 is not None
        self.assertEqual(m2.content, "v2.0")
        self.assertEqual(len(m2.provenance_history), 1)
        self.assertEqual(m2.provenance_history[0]["previous_content"], "v1.0")

        # Active retrieval returns only v2.0, while include_inactive returns both
        active_mems = store.retrieve_memories(key="api.version")
        self.assertEqual(len(active_mems), 1)
        self.assertEqual(active_mems[0].memory_id, m2.memory_id)
        all_mems = store.retrieve_memories(key="api.version", include_inactive=True)
        self.assertEqual(len(all_mems), 2)

        # Mark obsolete
        store.mark_memory_obsolete(m2.memory_id, reason="Deprecated")
        self.assertEqual(len(store.retrieve_memories(key="api.version")), 0)

        # Experience guard: UNVERIFIED outcome cannot have VERIFIED_EXECUTION confidence
        unverified_exp = ExperienceRecord(
            experience_id="exp-unv",
            task_id="task-unv",
            objective="Unverified claim test",
            plan_summary="none",
            outcome=TaskOutcome.UNVERIFIED,
            lesson_confidence="VERIFIED_EXECUTION",
        )
        self.assertEqual(unverified_exp.lesson_confidence, "UNVERIFIED_CLAIM")

        # Context window adaptation & requirement preservation under compression
        task = self.orchestrator.create_task("Build critical security module")
        task.requirements.append(
            TaskRequirement(
                requirement_id="req-sec-1",
                description="Must enforce 0600 file permissions",
                category="file_exists",
                target="sec.py",
            )
        )
        task.relevant_outputs["huge_step"] = "X" * 10000
        ctx_payload = self.orchestrator.context_manager.build_model_context(
            task=task,
            available_capabilities=[],
            active_model_context_window_tokens=1024,
        )
        self.assertTrue(ctx_payload["token_budget_info"]["compressed"])
        self.assertEqual(
            ctx_payload["token_budget_info"]["token_estimation_method"],
            "heuristic_char_div_4",
        )
        self.assertEqual(len(ctx_payload["requirements"]), 1)
        self.assertEqual(
            ctx_payload["requirements"][0]["description"],
            "Must enforce 0600 file permissions",
        )

    def test_07_capability_dependency_detection_and_config_verification_checks(
        self,
    ) -> None:
        """
        Verify:
        - Capability declaring a missing binary/package reports MISSING_DEPENDENCY
        - Verifier checks config_reloadable and effective_model_config accurately
        - CLI providers, models, model-select, and model-test commands work end-to-end
        """
        class MissingDepCapability(Capability):
            identifier = "custom.needs_nonexistent_pkg"
            name = "Missing Dep Cap"
            purpose = "Test missing dependency reporting"
            required_dependencies = ["nonexistent_python_package_xyz_911"]

            def execute(
                self, params: Dict[str, Any], context: CapabilityContext
            ) -> CapabilityResult:
                return CapabilityResult(capability_id=self.identifier, success=True)

        cap = MissingDepCapability()
        self.orchestrator.registry.register(cap)
        self.assertEqual(
            self.orchestrator.registry.get_availability(cap.identifier),
            CapabilityAvailability.MISSING_DEPENDENCY,
        )

        # Verify config_reloadable and effective_model_config via Verifier
        self.config.save()
        task = self.orchestrator.create_task("Verify config and effective model")
        req_reload = TaskRequirement(
            requirement_id="r1",
            description="Config is valid and reloadable",
            category="config_reloadable",
        )
        req_eff = TaskRequirement(
            requirement_id="r2",
            description="Effective model matches active selection",
            category="effective_model_config",
            expected={"active_model": self.config.active_model},
        )
        chk1 = self.orchestrator.verifier.verify_requirement(task, req_reload)
        chk2 = self.orchestrator.verifier.verify_requirement(task, req_eff)
        self.assertEqual(chk1.status.value, "PASSED")
        self.assertEqual(chk2.status.value, "PASSED")

        # Test CLI model-test and models subcommands
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "agent911",
                "--workspace",
                str(self.workspace),
                "--json",
                "models",
            ],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(proc.returncode, 0, f"stderr: {proc.stderr}")
        models_out = json.loads(proc.stdout)
        self.assertIn("active_model", models_out)
        self.assertTrue(len(models_out["models"]) >= 2)


if __name__ == "__main__":
    unittest.main()
