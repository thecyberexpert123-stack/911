"""Section 19.2: Automated tests for persistent memory, experience, context compression, and resumption."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agent911.config import AgentConfig
from agent911.memory import PersistentMemoryStore
from agent911.models import (
    ExperienceRecord,
    MemoryCategory,
    MemoryConfidence,
    TaskLifecycleStage,
    TaskOutcome,
    new_id,
)
from agent911.orchestrator import Agent911Orchestrator


class TestMemoryAndExperience(unittest.TestCase):
    """Tests for Section 19.2 & Section 8-10: Persistent memory, experience, and context."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp_dir.name)
        self.db_path = self.workspace / ".agent911" / "agent911.db"
        self.store = PersistentMemoryStore(self.db_path)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_01_store_retrieve_and_update_memory(self) -> None:
        """Test storing, retrieving, and updating memory records across categories."""
        rec1 = self.store.store_memory(
            category=MemoryCategory.FACTUAL,
            key="project.language",
            content="Python 3.11 with SQLite persistence",
            task_id="task-101",
            source="inspection",
            evidence={"verified_by": "project.inspect"},
            confidence=MemoryConfidence.VERIFIED,
        )
        rec2 = self.store.store_memory(
            category=MemoryCategory.INSTRUCTION,
            key="user.coding_style",
            content="Use type annotations and standard library where practical",
            confidence=MemoryConfidence.OBSERVED,
        )
        rec_assumption = self.store.store_memory(
            category=MemoryCategory.WORKING_STATE,
            key="default.encoding",
            content="Assumed UTF-8 file encoding",
            confidence=MemoryConfidence.ASSUMPTION,
        )
        self.assertEqual(rec_assumption.confidence, MemoryConfidence.ASSUMPTION)

        # Retrieve by keyword query
        found = self.store.retrieve_memories(query="SQLite persistence")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].memory_id, rec1.memory_id)

        # Update existing memory
        updated = self.store.update_memory(
            memory_id=rec2.memory_id,
            content="Use strict type annotations and zero-dependency stdlib design",
            confidence=MemoryConfidence.VERIFIED,
            evidence={"confirmed": True},
        )
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.confidence, MemoryConfidence.VERIFIED)
        self.assertIn("zero-dependency", updated.content)

    def test_02_avoid_duplicate_memory_records(self) -> None:
        """Test that storing the same key or identical content updates rather than duplicates."""
        first = self.store.store_memory(
            category=MemoryCategory.FACTUAL,
            key="env.os",
            content="Linux x86_64",
            evidence={"step": 1},
        )
        second = self.store.store_memory(
            category=MemoryCategory.FACTUAL,
            key="env.os",
            content="Linux x86_64 kernel 6",
            evidence={"step": 2},
        )
        self.assertEqual(first.memory_id, second.memory_id)
        all_os = self.store.retrieve_memories(category=MemoryCategory.FACTUAL, key="env.os")
        self.assertEqual(len(all_os), 1)
        self.assertEqual(all_os[0].content, "Linux x86_64 kernel 6")
        self.assertEqual(all_os[0].evidence, {"step": 1, "step": 2})

    def test_03_record_and_retrieve_experience_associated_with_task(self) -> None:
        """Test recording an execution experience, associating with a task, and retrieving it."""
        exp = ExperienceRecord(
            experience_id=new_id("exp"),
            task_id="task-exp-1",
            objective="Write report file into nested reports/2026/summary.txt",
            plan_summary="step-1:fs.write",
            capabilities_used=["fs.mkdir", "fs.write"],
            operations_performed=["step-1:fs.write(COMPLETED)"],
            unexpected_failures=["ParentDirectoryNotFoundError: reports/2026 missing"],
            identified_causes=["Parent directory reports/2026 did not exist"],
            corrections_attempted=["Create parent directory via fs.mkdir and set create_parents=True"],
            successful_remedies=["Create parent directory via fs.mkdir and set create_parents=True"],
            reusable_procedures=["fs.mkdir -> fs.write"],
            outcome=TaskOutcome.SUCCESS,
            tags=["filesystem", "nested_path"],
        )
        saved = self.store.record_experience(exp)
        self.assertEqual(saved.task_id, "task-exp-1")

        # Recording again for same task_id updates in-place instead of duplicating
        self.store.record_experience(exp)
        by_task = self.store.retrieve_experiences(task_id="task-exp-1")
        self.assertEqual(len(by_task), 1)

        # Retrieve by failure/objective query
        relevant = self.store.retrieve_experiences(query="ParentDirectoryNotFoundError nested")
        self.assertEqual(len(relevant), 1)
        self.assertEqual(relevant[0].experience_id, saved.experience_id)

    def test_04_persistence_across_process_boundary(self) -> None:
        """Test that memories and experiences survive across an actual OS process boundary."""
        self.store.store_memory(
            category=MemoryCategory.FACTUAL,
            key="restart.token",
            content="persisted-across-subprocess-911",
            confidence=MemoryConfidence.VERIFIED,
        )

        # Spawn a separate Python process to open the SQLite DB and read the record
        script = (
            "import json\n"
            "from agent911.memory import PersistentMemoryStore\n"
            f"store = PersistentMemoryStore({str(self.db_path)!r})\n"
            "recs = store.retrieve_memories(key='restart.token')\n"
            "print(json.dumps([r.to_dict() for r in recs]))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        loaded = json.loads(proc.stdout.strip())
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["content"], "persisted-across-subprocess-911")

    def test_05_context_compression_and_task_resumption(self) -> None:
        """Test context compression and long-running task interruption + resumption."""
        cfg = AgentConfig(workspace_root=str(self.workspace), max_context_chars=800)
        orch = Agent911Orchestrator(config=cfg)

        # Execute a 2-step task, stopping after step 1 to simulate interruption
        task = orch.execute_task(
            "Create two files 'first.txt' and 'second.txt'.",
            explicit_plan=[
                {
                    "step_id": "step-1",
                    "description": "Write first.txt",
                    "capability_id": "fs.write",
                    "parameters": {"path": "first.txt", "content": "one\n"},
                },
                {
                    "step_id": "step-2",
                    "description": "Write second.txt",
                    "capability_id": "fs.write",
                    "parameters": {"path": "second.txt", "content": "two\n"},
                },
            ],
            stop_after_step=1,
        )
        self.assertEqual(task.current_execution_state, TaskLifecycleStage.INTERRUPTED)
        self.assertEqual(task.completed_actions, ["step-1"])
        self.assertEqual(task.pending_actions, ["step-2"])
        self.assertTrue((self.workspace / "first.txt").exists())
        self.assertFalse((self.workspace / "second.txt").exists())

        # Create a brand new orchestrator instance and resume the interrupted task
        orch_restarted = Agent911Orchestrator(config=cfg)
        resumed = orch_restarted.resume_task(task.task_id)
        self.assertEqual(resumed.final_outcome, TaskOutcome.SUCCESS)
        self.assertTrue((self.workspace / "first.txt").exists())
        self.assertTrue((self.workspace / "second.txt").exists())
        # Confirm step-1 was NOT blindly repeated (only 2 capability calls total: step-1 and step-2)
        self.assertEqual(len(resumed.capability_calls), 2)

        # Test context compression preserves essential fields
        summary_str = orch_restarted.context_manager.compress_context(resumed)
        summary_obj = json.loads(summary_str)
        self.assertEqual(summary_obj["original_instruction"], task.original_instruction)
        self.assertEqual(len(summary_obj["completed_work"]), 2)


if __name__ == "__main__":
    unittest.main()
