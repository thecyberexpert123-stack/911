"""Section 19.6 & Section 11: Automated tests for organic capability development and evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent911.capabilities import CapabilityContext
from agent911.config import AgentConfig
from agent911.models import CapabilityAvailability, TaskOutcome
from agent911.orchestrator import Agent911Orchestrator


class TestCapabilityGrowth(unittest.TestCase):
    """Tests for Section 19.6: Organic capability development, verification, live testing, and persistence."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp_dir.name)
        self.config = AgentConfig(workspace_root=str(self.workspace))
        self.orchestrator = Agent911Orchestrator(config=self.config)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_01_develop_register_discover_invoke_and_persist_capability(self) -> None:
        """
        Test developing a new capability ('text.metrics'), verifying it via AST/bytecode,
        exercising it via Live Run Testing, invoking it in a task, and reloading it across restart.
        """
        self.assertEqual(
            self.orchestrator.registry.get_availability("text.metrics"),
            CapabilityAvailability.NOT_IMPLEMENTED,
        )

        # Execute a task that requires 'text.metrics' (which does not exist yet!)
        sample_file = self.workspace / "article.txt"
        sample_file.write_text("Agent 911 builds working capabilities.\nSecond line here.\n", encoding="utf-8")

        task = self.orchestrator.execute_task(
            f"Compute word count and text metrics on '{sample_file.name}' using text.metrics"
        )
        self.assertEqual(task.final_outcome, TaskOutcome.SUCCESS)

        # Verify 'text.metrics' is now registered and AVAILABLE
        self.assertTrue(self.orchestrator.registry.has("text.metrics"))
        self.assertEqual(
            self.orchestrator.registry.get_availability("text.metrics"),
            CapabilityAvailability.AVAILABLE,
        )

        # Invoke the newly grown capability directly through the registry
        ctx = CapabilityContext(
            config=self.config,
            memory_store=self.orchestrator.memory_store,
            model_interface=self.orchestrator.model_interface,
            registry=self.orchestrator.registry,
            task_id="direct-invoke",
        )
        res = self.orchestrator.registry.invoke(
            "text.metrics", {"path": "article.txt"}, ctx
        )
        self.assertTrue(res.success)
        self.assertEqual(res.data["word_count"], 8)
        self.assertEqual(res.data["line_count"], 2)

        # Create a fresh Agent911Orchestrator instance to simulate process restart
        restarted_orch = Agent911Orchestrator(config=self.config)
        self.assertTrue(restarted_orch.registry.has("text.metrics"))
        self.assertEqual(
            restarted_orch.registry.get_availability("text.metrics"),
            CapabilityAvailability.AVAILABLE,
        )

    def test_02_broken_capability_rejected_by_verification_or_live_test(self) -> None:
        """
        Confirm that source code generation alone does NOT mark a capability operational
        if Verification or Live Run Testing fails.
        """
        ctx = CapabilityContext(
            config=self.config,
            memory_store=self.orchestrator.memory_store,
            model_interface=self.orchestrator.model_interface,
            registry=self.orchestrator.registry,
            task_id="bad-cap-test",
        )

        # 1. Syntax error in implementation_code -> fails Verification
        res_syntax = self.orchestrator.registry.invoke(
            "capability.develop",
            {
                "capability_id": "broken.syntax",
                "name": "Broken Syntax Cap",
                "purpose": "Should fail verification",
                "implementation_code": "def invalid python syntax(((",
            },
            ctx,
        )
        self.assertFalse(res_syntax.success)
        self.assertEqual(res_syntax.error_type, "CapabilityVerificationError")
        self.assertFalse(self.orchestrator.registry.has("broken.syntax"))

        # 2. Compiles cleanly (passes Verification), but raises RuntimeError during Live Run Testing
        res_runtime = self.orchestrator.registry.invoke(
            "capability.develop",
            {
                "capability_id": "broken.runtime",
                "name": "Broken Runtime Cap",
                "purpose": "Compiles but fails live runtime test",
                "implementation_code": "raise RuntimeError('runtime failure inside capability')",
                "test_params": {"x": 1},
            },
            ctx,
        )
        self.assertFalse(res_runtime.success)
        self.assertEqual(res_runtime.error_type, "CapabilityLiveTestError")
        self.assertTrue(res_runtime.data["verified"])
        self.assertFalse(res_runtime.data["live_tested"])
        self.assertFalse(res_runtime.data["operational"])
        # Must NOT be marked AVAILABLE in registry
        self.assertEqual(
            self.orchestrator.registry.get_availability("broken.runtime"),
            CapabilityAvailability.UNAVAILABLE,
        )


if __name__ == "__main__":
    unittest.main()
