"""Section 19.3: Automated tests for Formal Evaluation Mechanism 1 (Verification)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent911.config import AgentConfig
from agent911.evaluation import OutcomeClassifier, Verifier
from agent911.models import (
    ActionStatus,
    LiveRunTestRecord,
    LiveTestStatus,
    PlannedAction,
    TaskOutcome,
    TaskRequirement,
    TaskState,
    VerificationCheckResult,
    VerificationStatus,
)
from agent911.orchestrator import Agent911Orchestrator


class TestVerificationMechanism(unittest.TestCase):
    """Tests for Section 19.3: Verification statuses, evidence inspection, and propagation."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp_dir.name)
        self.config = AgentConfig(workspace_root=str(self.workspace))
        self.orchestrator = Agent911Orchestrator(config=self.config)
        self.verifier: Verifier = self.orchestrator.verifier

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_01_genuinely_passing_checks(self) -> None:
        """Test genuinely passing verification checks on real files and Python AST."""
        py_file = self.workspace / "module_ok.py"
        py_file.write_text(
            "def add_numbers(a: int, b: int) -> int:\n    return a + b\n",
            encoding="utf-8",
        )
        req_exists = TaskRequirement(
            requirement_id="r1",
            description="module_ok.py exists",
            category="file_exists",
            target="module_ok.py",
        )
        req_compiles = TaskRequirement(
            requirement_id="r2",
            description="module_ok.py compiles",
            category="code_compiles",
            target="module_ok.py",
        )
        req_func = TaskRequirement(
            requirement_id="r3",
            description="add_numbers is defined",
            category="function_exists",
            target="module_ok.py",
            expected="add_numbers",
        )

        task = TaskState(task_id="t-pass", original_instruction="create module_ok.py")
        for r in (req_exists, req_compiles, req_func):
            res = self.verifier.verify_requirement(task, r)
            self.assertEqual(res.status, VerificationStatus.PASSED)
            self.assertEqual(res.task_id, "t-pass")
            self.assertIsNone(res.failure_explanation)

    def test_02_genuinely_failing_checks(self) -> None:
        """Test genuinely failing verification checks on missing files, bad syntax, and missing symbols."""
        bad_py = self.workspace / "broken.py"
        bad_py.write_text("def broken_syntax(\n", encoding="utf-8")

        task = TaskState(task_id="t-fail", original_instruction="create broken.py")
        res_missing = self.verifier.verify_requirement(
            task,
            TaskRequirement(
                requirement_id="r-miss",
                description="missing.txt exists",
                category="file_exists",
                target="missing.txt",
            ),
        )
        self.assertEqual(res_missing.status, VerificationStatus.FAILED)
        self.assertIsNotNone(res_missing.failure_explanation)

        res_syntax = self.verifier.verify_requirement(
            task,
            TaskRequirement(
                requirement_id="r-syn",
                description="broken.py compiles",
                category="code_compiles",
                target="broken.py",
            ),
        )
        self.assertEqual(res_syntax.status, VerificationStatus.FAILED)
        self.assertIn("SyntaxError", res_syntax.observed_result)

    def test_03_unavailable_check(self) -> None:
        """Test an unavailable check is marked UNAVAILABLE, never PASSED."""
        task = TaskState(task_id="t-unavail", original_instruction="check unavailable")
        res = self.verifier.verify_requirement(
            task,
            TaskRequirement(
                requirement_id="r-unavail",
                description="External inspector check",
                category="unavailable_check",
                expected="External tool not installed",
            ),
        )
        self.assertEqual(res.status, VerificationStatus.UNAVAILABLE)
        self.assertNotEqual(res.status, VerificationStatus.PASSED)

    def test_04_inconclusive_check_and_ai_assertion_guard(self) -> None:
        """Test inconclusive checks and verify AI assertions alone cannot be marked PASSED."""
        task = TaskState(task_id="t-inc", original_instruction="check inconclusive")
        res_ai = self.verifier.verify_requirement(
            task,
            TaskRequirement(
                requirement_id="r-ai",
                description="AI claims success",
                category="ai_assertion",
                expected="Model says it worked",
            ),
        )
        self.assertEqual(res_ai.status, VerificationStatus.INCONCLUSIVE)

        # Even if someone constructs VerificationCheckResult with status=PASSED and is_ai_assertion_only=True,
        # __post_init__ must downgrade it to INCONCLUSIVE.
        guarded = VerificationCheckResult(
            check_id="c1",
            task_id="t-inc",
            requirement="AI assertion",
            method="model",
            observed_result="claimed ok",
            status=VerificationStatus.PASSED,
            is_ai_assertion_only=True,
        )
        self.assertEqual(guarded.status, VerificationStatus.INCONCLUSIVE)

    def test_05_task_with_unmet_required_criterion_and_propagation(self) -> None:
        """Test a task where one required criterion is met and another is unmet propagates to PARTIAL_SUCCESS or FAILED."""
        task = self.orchestrator.execute_task(
            "Create 'present.txt' and verify 'absent.txt' also exists.",
            explicit_plan=[
                {
                    "step_id": "step-1",
                    "description": "Write present.txt",
                    "capability_id": "fs.write",
                    "parameters": {"path": "present.txt", "content": "exists\n"},
                }
            ],
            explicit_requirements=[
                {
                    "requirement_id": "req-1",
                    "description": "present.txt exists",
                    "category": "file_exists",
                    "target": "present.txt",
                    "required": True,
                },
                {
                    "requirement_id": "req-2",
                    "description": "absent.txt exists",
                    "category": "file_exists",
                    "target": "absent.txt",
                    "required": True,
                },
            ],
        )
        # Every verification result must be associated with task.task_id
        for v in task.verification_results:
            self.assertEqual(v.task_id, task.task_id)

        statuses = {v.requirement: v.status for v in task.verification_results}
        self.assertEqual(statuses["present.txt exists"], VerificationStatus.PASSED)
        self.assertEqual(statuses["absent.txt exists"], VerificationStatus.FAILED)
        self.assertEqual(task.final_outcome, TaskOutcome.PARTIAL_SUCCESS)

    def test_06_task_with_incomplete_evidence_classified_unverified(self) -> None:
        """Test that a task with only inconclusive evidence is classified as UNVERIFIED, never SUCCESS."""
        task = TaskState(task_id="t-incomplete", original_instruction="check incomplete")
        task.planned_actions = [
            PlannedAction(
                step_id="step-1",
                description="noop",
                capability_id="fs.list",
                status=ActionStatus.COMPLETED,
            )
        ]
        task.requirements = [
            TaskRequirement(
                requirement_id="req-inc",
                description="Unverified assertion",
                category="ai_assertion",
                required=True,
            )
        ]
        self.verifier.verify_task(task)
        task.live_run_results = [
            LiveRunTestRecord(
                test_id="l1",
                task_id=task.task_id,
                executed_operation="fs.list",
                target_component="fs.list",
                environment={},
                exit_code=0,
                stdout="ok",
                stderr="",
                expected_behavior_occurred=True,
                status=LiveTestStatus.PASSED,
            )
        ]
        outcome, _ = OutcomeClassifier.classify(task)
        self.assertEqual(outcome, TaskOutcome.UNVERIFIED)


if __name__ == "__main__":
    unittest.main()
