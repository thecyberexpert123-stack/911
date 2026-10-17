"""Section 19.5 & Section 10: Automated tests for failure diagnosis, targeted recovery, and experience reuse."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent911.capabilities import CapabilityContext
from agent911.config import AgentConfig
from agent911.models import (
    ActionStatus,
    CapabilityResult,
    ExecutionErrorRecord,
    PlannedAction,
    TaskOutcome,
    TaskState,
)
from agent911.orchestrator import Agent911Orchestrator
from agent911.recovery import RecoveryEngine


class TestFailureRecoveryAndExperience(unittest.TestCase):
    """Tests for Section 19.5: Failure diagnosis, targeted recovery, loop prevention, and experience learning."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp_dir.name)
        self.config = AgentConfig(
            workspace_root=str(self.workspace),
            max_recovery_attempts_per_step=2,
        )
        self.orchestrator = Agent911Orchestrator(config=self.config)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_01_targeted_recovery_preserves_prior_work_and_learns_experience(self) -> None:
        """
        Test a 2-step task where step 1 succeeds and step 2 initially fails due to missing
        parent directory ('reports/2026/summary.txt' with create_parents=False).
        Verify:
        - Step 1 is NOT re-run.
        - Step 2 failure is diagnosed, parent directory is created, and step 2 is re-executed.
        - Verification and Live Run Testing pass after recovery -> SUCCESS.
        - Subsequent task writing to a nested path applies the learned experience remedy
          during planning and succeeds on the first try without error!
        """
        task1 = self.orchestrator.execute_task(
            "Write root_note.txt and nested reports/2026/summary.txt",
            explicit_plan=[
                {
                    "step_id": "step-1",
                    "description": "Write root_note.txt",
                    "capability_id": "fs.write",
                    "parameters": {"path": "root_note.txt", "content": "root ok\n"},
                },
                {
                    "step_id": "step-2",
                    "description": "Write nested summary file",
                    "capability_id": "fs.write",
                    "parameters": {
                        "path": "reports/2026/summary.txt",
                        "content": "nested summary ok\n",
                        "create_parents": False,
                    },
                },
            ],
            explicit_requirements=[
                {
                    "requirement_id": "req-1",
                    "description": "root_note.txt exists",
                    "category": "file_exists",
                    "target": "root_note.txt",
                },
                {
                    "requirement_id": "req-2",
                    "description": "reports/2026/summary.txt exists",
                    "category": "file_exists",
                    "target": "reports/2026/summary.txt",
                },
            ],
        )

        # Verify error was captured, diagnosed, and recovered
        self.assertEqual(len(task1.errors), 1)
        self.assertEqual(task1.errors[0].error_type, "ParentDirectoryNotFoundError")
        self.assertTrue(task1.errors[0].recovered)
        self.assertEqual(len(task1.recovery_attempts), 1)
        self.assertTrue(task1.recovery_attempts[0].succeeded)
        self.assertEqual(task1.final_outcome, TaskOutcome.SUCCESS)

        # Verify step-1 was executed only once (preserved completed work!)
        step1_calls = [c for c in task1.capability_calls if c.step_id == "step-1"]
        self.assertEqual(len(step1_calls), 1)
        # Step-2 had 1 failed call + 1 recovered call
        step2_calls = [c for c in task1.capability_calls if c.step_id == "step-2"]
        self.assertEqual(len(step2_calls), 2)

        # Now run a second task writing to another nested path ('nested/deep/output.txt')
        # without specifying create_parents. The planner should retrieve the experience from task1
        # and automatically enable create_parents=True!
        task2 = self.orchestrator.execute_task(
            "Write file 'nested/deep/output.txt' with content 'learned from experience'"
        )
        self.assertEqual(task2.final_outcome, TaskOutcome.SUCCESS)
        self.assertEqual(len(task2.errors), 0)
        self.assertTrue(
            any("Applied prior experience remedy" in d for d in task2.decisions),
            msg=f"Expected experience decision in {task2.decisions}",
        )

    def test_02_prevention_of_uncontrolled_retry_loops(self) -> None:
        """Test that repeated identical failures or exceeding max_attempts stops retries cleanly."""
        engine = RecoveryEngine(max_attempts_per_step=2)
        task = TaskState(task_id="t-loop", original_instruction="test loop guard")
        action = PlannedAction(
            step_id="step-1",
            description="Write with fallback that also fails",
            capability_id="fs.read",
            parameters={
                "path": "still_missing.txt",
                "_recovery_parameters": {"path": "still_missing.txt"},
            },
        )
        failed_res = CapabilityResult(
            capability_id="fs.read",
            success=False,
            error="File does not exist: 'still_missing.txt'",
            error_type="FileNotFoundError",
        )
        err_rec = ExecutionErrorRecord(
            error_id="e1",
            task_id="t-loop",
            step_id="step-1",
            capability_id="fs.read",
            error_type="FileNotFoundError",
            message="File does not exist: 'still_missing.txt'",
        )
        ctx = CapabilityContext(
            config=self.config,
            memory_store=self.orchestrator.memory_store,
            model_interface=self.orchestrator.model_interface,
            registry=self.orchestrator.registry,
            task_id="t-loop",
        )

        # First recovery attempt tries the fallback and fails
        ok1, _ = engine.attempt_step_recovery(task, action, failed_res, err_rec, ctx)
        self.assertFalse(ok1)

        # Second attempt with the exact same failure signature must be rejected immediately!
        can_try, reason = engine.can_attempt_recovery(action, err_rec)
        self.assertFalse(can_try)
        self.assertIn("Repeated identical failure", reason)

    def test_03_accurate_reporting_when_recovery_fails_or_is_blocked(self) -> None:
        """Test accurate reporting when an operation fails and cannot be recovered."""
        # 1. Unrecoverable failing command -> FAILED
        task_fail = self.orchestrator.execute_task(
            "Run a failing command",
            explicit_plan=[
                {
                    "step_id": "step-1",
                    "description": "Run nonexistent subcommand",
                    "capability_id": "cmd.run",
                    "parameters": {"command": ["python3", "-c", "import sys; sys.exit(2)"]},
                }
            ],
            explicit_requirements=[
                {
                    "requirement_id": "req-1",
                    "description": "Command succeeds",
                    "category": "command_succeeds",
                }
            ],
        )
        self.assertEqual(task_fail.final_outcome, TaskOutcome.FAILED)
        self.assertEqual(len(task_fail.errors), 1)
        self.assertFalse(task_fail.errors[0].recovered)

        # 2. Workspace boundary violation -> BLOCKED
        task_blocked = self.orchestrator.execute_task(
            "Attempt to read outside workspace",
            explicit_plan=[
                {
                    "step_id": "step-1",
                    "description": "Read outside workspace",
                    "capability_id": "fs.read",
                    "parameters": {"path": "../../../etc/hosts"},
                }
            ],
            explicit_requirements=[],
        )
        self.assertEqual(task_blocked.final_outcome, TaskOutcome.BLOCKED)


if __name__ == "__main__":
    unittest.main()
