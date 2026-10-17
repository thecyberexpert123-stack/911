"""Section 19.4: Automated tests for Formal Evaluation Mechanism 2 (Live Run Testing)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from agent911.capabilities import CapabilityContext
from agent911.config import AgentConfig
from agent911.evaluation import LiveRunTester, OutcomeClassifier
from agent911.models import LiveTestStatus, TaskOutcome, VerificationStatus
from agent911.orchestrator import Agent911Orchestrator


class TestLiveRunTestingMechanism(unittest.TestCase):
    """Tests for Section 19.4: Live Run Testing execution, timeouts, unavailability, and propagation."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp_dir.name)
        self.config = AgentConfig(workspace_root=str(self.workspace))
        self.orchestrator = Agent911Orchestrator(config=self.config)
        self.cap_context = CapabilityContext(
            config=self.config,
            memory_store=self.orchestrator.memory_store,
            model_interface=self.orchestrator.model_interface,
            registry=self.orchestrator.registry,
            task_id="task-live",
        )
        self.live_tester = LiveRunTester(self.cap_context)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_01_execute_real_supported_command_and_capture_output(self) -> None:
        """Test executing a real command captures exit_code=0, stdout, and PASSED status."""
        rec = self.live_tester.test_command_live(
            task_id="task-live",
            command=[sys.executable, "-c", "print('live_check_ok_911')"],
            expected_stdout="live_check_ok_911",
        )
        self.assertEqual(rec.status, LiveTestStatus.PASSED)
        self.assertTrue(rec.expected_behavior_occurred)
        self.assertEqual(rec.exit_code, 0)
        self.assertIn("live_check_ok_911", rec.stdout)

    def test_02_invoke_target_capability_live(self) -> None:
        """Test invoking a registered capability through LiveRunTester."""
        (self.workspace / "sample.txt").write_text("alpha beta\n", encoding="utf-8")
        rec = self.live_tester.test_capability_invocation(
            task_id="task-live",
            capability_id="fs.read",
            sample_params={"path": "sample.txt"},
            expected_output_subset={"content": "alpha beta\n"},
        )
        self.assertEqual(rec.status, LiveTestStatus.PASSED)
        self.assertTrue(rec.expected_behavior_occurred)
        self.assertEqual(rec.exit_code, 0)

    def test_03_report_reproducible_failure_accurately(self) -> None:
        """Test that a command or script exiting non-zero is reported as FAILED with stderr."""
        rec = self.live_tester.test_command_live(
            task_id="task-live",
            command=[sys.executable, "-c", "import sys; sys.stderr.write('boom_err\\n'); sys.exit(3)"],
        )
        self.assertEqual(rec.status, LiveTestStatus.FAILED)
        self.assertFalse(rec.expected_behavior_occurred)
        self.assertEqual(rec.exit_code, 3)
        self.assertIn("boom_err", rec.stderr)

    def test_04_handle_process_timeout(self) -> None:
        """Test that a command exceeding timeout is reported as TIMED_OUT."""
        rec = self.live_tester.test_command_live(
            task_id="task-live",
            command=[sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_seconds=0.2,
        )
        self.assertEqual(rec.status, LiveTestStatus.TIMED_OUT)
        self.assertFalse(rec.expected_behavior_occurred)
        self.assertIsNone(rec.exit_code)
        self.assertIsNotNone(rec.limitations)

    def test_05_report_unavailable_external_tests_honestly(self) -> None:
        """Test that external model live test reports UNAVAILABLE when credentials are absent."""
        rec = self.live_tester.test_external_model_live("task-live")
        self.assertEqual(rec.status, LiveTestStatus.UNAVAILABLE)
        self.assertFalse(rec.expected_behavior_occurred)
        self.assertIsNotNone(rec.limitations)

        # Also test invoking model.query capability without credentials reports UNAVAILABLE
        cap_rec = self.live_tester.test_capability_invocation(
            task_id="task-live",
            capability_id="model.query",
            sample_params={"prompt": "hello"},
        )
        self.assertEqual(cap_rec.status, LiveTestStatus.UNAVAILABLE)

    def test_06_propagation_of_live_test_failure_even_when_verification_passes(self) -> None:
        """
        Verify Section 4.3 & 13: A structurally valid Python file that passes Verification
        (file_exists, code_compiles, function_exists) but raises a runtime error during
        Live Run Testing must NOT be classified as SUCCESS.
        """
        buggy_code = (
            "def divide_by_zero(x=10):\n"
            "    return x / 0\n\n"
            "if __name__ == '__main__':\n"
            "    divide_by_zero()\n"
        )
        task = self.orchestrator.execute_task(
            "Write 'runtime_bug.py' with function 'divide_by_zero'.",
            explicit_plan=[
                {
                    "step_id": "step-1",
                    "description": "Write runtime_bug.py",
                    "capability_id": "fs.write",
                    "parameters": {"path": "runtime_bug.py", "content": buggy_code},
                }
            ],
            explicit_requirements=[
                {
                    "requirement_id": "req-1",
                    "description": "runtime_bug.py exists",
                    "category": "file_exists",
                    "target": "runtime_bug.py",
                },
                {
                    "requirement_id": "req-2",
                    "description": "runtime_bug.py compiles",
                    "category": "code_compiles",
                    "target": "runtime_bug.py",
                },
                {
                    "requirement_id": "req-3",
                    "description": "divide_by_zero function exists",
                    "category": "function_exists",
                    "target": "runtime_bug.py",
                    "expected": "divide_by_zero",
                },
            ],
        )

        # All static Verification checks passed!
        for v in task.verification_results:
            self.assertEqual(v.status, VerificationStatus.PASSED)

        # Live Run Testing actually invoked divide_by_zero() and caught ZeroDivisionError!
        self.assertEqual(len(task.live_run_results), 1)
        self.assertEqual(task.live_run_results[0].status, LiveTestStatus.FAILED)
        self.assertIn("ZeroDivisionError", task.live_run_results[0].stderr)

        # Therefore final task outcome is PARTIAL_SUCCESS (static requirements met, runtime live test failed), never SUCCESS!
        self.assertNotEqual(task.final_outcome, TaskOutcome.SUCCESS)
        self.assertEqual(task.final_outcome, TaskOutcome.PARTIAL_SUCCESS)


if __name__ == "__main__":
    unittest.main()
