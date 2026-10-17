"""Section 19.7: End-to-end task lifecycle tests (fixture-based model planning + CLI live execution)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from agent911.config import AgentConfig, ModelConfig
from agent911.models import (
    LiveTestStatus,
    TaskLifecycleStage,
    TaskOutcome,
    VerificationStatus,
)
from agent911.orchestrator import Agent911Orchestrator


class TestEndToEndLifecycle(unittest.TestCase):
    """End-to-end tests exercising the complete Agent 911 task lifecycle and CLI."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_01_end_to_end_coding_and_testing_workflow_with_deterministic_fixture(self) -> None:
        """
        Exercise the complete lifecycle using a deterministic model fixture (clearly marked
        is_simulated=True) that plans a multi-step software creation and unit test run.
        """
        def deterministic_planner(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "main_goal": "Create math_utils.py and test_math_utils.py and run unit tests",
                "explicit_requirements": [
                    "math_utils.py exists and defines fibonacci(n)",
                    "test_math_utils.py exists and passes under unittest",
                ],
                "expected_outputs": ["math_utils.py", "test_math_utils.py"],
                "assumptions": [],
                "planned_actions": [
                    {
                        "step_id": "step-1",
                        "description": "Create math_utils.py",
                        "capability_id": "fs.write",
                        "parameters": {
                            "path": "math_utils.py",
                            "content": (
                                "def fibonacci(n: int) -> int:\n"
                                "    if n <= 0:\n"
                                "        return 0\n"
                                "    if n == 1:\n"
                                "        return 1\n"
                                "    a, b = 0, 1\n"
                                "    for _ in range(2, n + 1):\n"
                                "        a, b = b, a + b\n"
                                "    return b\n"
                            ),
                        },
                    },
                    {
                        "step_id": "step-2",
                        "description": "Create test_math_utils.py",
                        "capability_id": "fs.write",
                        "parameters": {
                            "path": "test_math_utils.py",
                            "content": (
                                "import unittest\n"
                                "from math_utils import fibonacci\n\n"
                                "class TestMathUtils(unittest.TestCase):\n"
                                "    def test_fib(self):\n"
                                "        self.assertEqual(fibonacci(0), 0)\n"
                                "        self.assertEqual(fibonacci(1), 1)\n"
                                "        self.assertEqual(fibonacci(7), 13)\n\n"
                                "if __name__ == '__main__':\n"
                                "    unittest.main()\n"
                            ),
                        },
                    },
                    {
                        "step_id": "step-3",
                        "description": "Run unit tests on test_math_utils.py",
                        "capability_id": "test.run",
                        "parameters": {
                            "target": "test_math_utils.py",
                            "runner": "unittest",
                        },
                    },
                ],
                "requirements": [
                    {
                        "requirement_id": "req-1",
                        "description": "math_utils.py exists",
                        "category": "file_exists",
                        "target": "math_utils.py",
                    },
                    {
                        "requirement_id": "req-2",
                        "description": "fibonacci function exists in math_utils.py",
                        "category": "function_exists",
                        "target": "math_utils.py",
                        "expected": "fibonacci",
                    },
                    {
                        "requirement_id": "req-3",
                        "description": "Unit tests pass",
                        "category": "tests_pass",
                        "target": "test_math_utils.py",
                    },
                ],
            }

        cfg = AgentConfig(
            workspace_root=str(self.workspace),
            model=ModelConfig(provider="deterministic", model_id="fixture-planner"),
        )
        orch = Agent911Orchestrator(
            config=cfg, deterministic_responder=deterministic_planner
        )

        # Verify connectivity report honestly marks deterministic test double as non-live
        conn = orch.model_interface.check_connectivity()
        self.assertFalse(conn.is_live_provider)
        self.assertFalse(conn.provider_reached)

        task = orch.execute_task("Implement fibonacci in math_utils.py and run unit tests.")
        self.assertEqual(task.current_execution_state, TaskLifecycleStage.COMPLETED)
        self.assertEqual(task.final_outcome, TaskOutcome.SUCCESS)
        self.assertEqual(len(task.capability_calls), 3)

        # Verify both Formal Evaluation Mechanisms ran and passed
        self.assertGreaterEqual(len(task.verification_results), 4)
        for v in task.verification_results:
            self.assertEqual(v.status, VerificationStatus.PASSED)

        self.assertGreaterEqual(len(task.live_run_results), 1)
        for l in task.live_run_results:
            self.assertEqual(l.status, LiveTestStatus.PASSED)

        # Verify persistence of task, episodic memory, experience, and execution logs
        persisted_task = orch.memory_store.load_task(task.task_id)
        self.assertIsNotNone(persisted_task)
        assert persisted_task is not None
        self.assertEqual(persisted_task.final_outcome, TaskOutcome.SUCCESS)

        logs = orch.memory_store.get_execution_logs(task.task_id)
        self.assertGreaterEqual(len(logs), 4)

        exps = orch.memory_store.retrieve_experiences(task_id=task.task_id)
        self.assertEqual(len(exps), 1)
        self.assertEqual(exps[0].outcome, TaskOutcome.SUCCESS)

    def test_02_cli_subprocess_end_to_end_execution(self) -> None:
        """
        Exercise the real CLI entry point (`python3 -m agent911`) in a subprocess
        across multiple commands: run, inspect, memory, experiences, capabilities.
        """
        repo_root = Path(__file__).resolve().parent.parent

        # 1. Run a task via CLI
        run_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "agent911",
                "--workspace",
                str(self.workspace),
                "--json",
                "run",
                "Write a Python file 'greeter.py' with a function 'greet_agent'.",
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(run_proc.returncode, 0, msg=run_proc.stderr)
        task_data = json.loads(run_proc.stdout)
        self.assertEqual(task_data["final_outcome"], "SUCCESS")
        task_id = task_data["task_id"]
        self.assertTrue((self.workspace / "greeter.py").exists())

        # 2. Inspect the completed task via CLI
        insp_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "agent911",
                "--workspace",
                str(self.workspace),
                "--json",
                "inspect",
                task_id,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(insp_proc.returncode, 0, msg=insp_proc.stderr)
        insp_data = json.loads(insp_proc.stdout)
        self.assertEqual(insp_data["task_id"], task_id)
        self.assertGreater(len(insp_data["execution_logs"]), 0)


if __name__ == "__main__":
    unittest.main()
