"""Context management, active-model context-window-aware compression, and task resumption validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent911.memory import PersistentMemoryStore
from agent911.models import (
    ActionStatus,
    ExperienceRecord,
    MemoryCategory,
    MemoryRecord,
    TaskState,
)


class ContextManager:
    """Manages task execution context, retrieval, model-limit-aware compression, and resumption integrity."""

    def __init__(
        self,
        memory_store: PersistentMemoryStore,
        max_context_chars: int = 16000,
    ) -> None:
        self.memory_store = memory_store
        self.max_context_chars = max_context_chars

    def retrieve_task_context(
        self,
        instruction: str,
        error_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve relevant active memories and non-superseded experiences for a task."""
        factual_memories = self.memory_store.retrieve_memories(
            query=instruction, category=MemoryCategory.FACTUAL, limit=5
        )
        instruction_memories = self.memory_store.retrieve_memories(
            query=instruction, category=MemoryCategory.INSTRUCTION, limit=5
        )
        query_for_exp = f"{instruction} {error_hint}" if error_hint else instruction
        experiences = self.memory_store.retrieve_experiences(
            query=query_for_exp, limit=5
        )
        return {
            "factual_memories": [m.to_dict() for m in factual_memories],
            "instruction_memories": [m.to_dict() for m in instruction_memories],
            "relevant_experiences": [e.to_dict() for e in experiences],
        }

    def compress_context(self, task: TaskState) -> str:
        """
        Summarize task execution state when context grows large.
        Strictly preserves original instruction, current objective, constraints,
        completed work, unresolved requirements, relevant decisions, errors/remedies,
        and key evidence.
        """
        unresolved_reqs = [
            f"{r.requirement_id}: {r.description} (target={r.target})"
            for r in task.requirements
            if not r.satisfied
        ]
        satisfied_reqs = [
            f"{r.requirement_id}: {r.description}"
            for r in task.requirements
            if r.satisfied
        ]
        completed_steps = [
            f"{a.step_id} [{a.capability_id}]: {a.description} -> {a.output_summary or 'done'}"
            for a in task.planned_actions
            if a.status == ActionStatus.COMPLETED
        ]
        pending_steps = [
            f"{a.step_id} [{a.capability_id}]: {a.description}"
            for a in task.planned_actions
            if a.status in (
                ActionStatus.PENDING,
                ActionStatus.IN_PROGRESS,
                ActionStatus.FAILED,
            )
        ]
        error_summaries = [
            f"{e.step_id or 'general'} ({e.error_type}): {e.message[:160]} [recovered={e.recovered}]"
            for e in task.errors[-5:]
        ]
        verif_summaries = [
            f"{v.requirement}: {v.status.value} ({v.observed_result[:120]})"
            for v in task.verification_results
        ]
        live_summaries = [
            f"{l.target_component}: {l.status.value} (exit={l.exit_code})"
            for l in task.live_run_results
        ]

        summary_obj = {
            "original_instruction": task.original_instruction,
            "current_objective": task.interpreted_objective,
            "constraints": list(task.intent.constraints) if task.intent else [],
            "assumptions": task.assumptions,
            "decisions": task.decisions,
            "completed_work": completed_steps,
            "pending_steps": pending_steps,
            "satisfied_requirements": satisfied_reqs,
            "unresolved_requirements": unresolved_reqs,
            "recent_errors": error_summaries,
            "verification_findings": verif_summaries,
            "live_testing_results": live_summaries,
            "essential_outputs": {
                k: (str(v)[:300] if len(str(v)) > 300 else v)
                for k, v in list(task.relevant_outputs.items())[-8:]
            },
        }
        summary_str = json.dumps(summary_obj, indent=2, default=str)
        task.context_summary = summary_str
        return summary_str

    def build_model_context(
        self,
        task: TaskState,
        available_capabilities: List[Dict[str, Any]],
        retrieved_context: Optional[Dict[str, Any]] = None,
        current_focus: str = "planning",
        active_model_context_window_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Construct bounded context payload for the AI model interface.
        Adapts to active_model_context_window_tokens when known and automatically
        compresses intermediate details while never dropping user requirements or constraints.
        """
        effective_max_chars = self.max_context_chars
        if (
            active_model_context_window_tokens is not None
            and active_model_context_window_tokens > 0
        ):
            model_budget_chars = max(2000, int(active_model_context_window_tokens * 2))
            effective_max_chars = min(self.max_context_chars, model_budget_chars)

        unresolved = [r.to_dict() for r in task.requirements if not r.satisfied]
        completed = [
            {
                "step_id": a.step_id,
                "capability_id": a.capability_id,
                "description": a.description,
                "output_summary": a.output_summary,
            }
            for a in task.planned_actions
            if a.status == ActionStatus.COMPLETED
        ]
        pending = [
            a.to_dict()
            for a in task.planned_actions
            if a.status != ActionStatus.COMPLETED
        ]

        payload: Dict[str, Any] = {
            "focus": current_focus,
            "task_id": task.task_id,
            "original_instruction": task.original_instruction,
            "current_objective": task.interpreted_objective,
            "constraints": list(task.intent.constraints) if task.intent else [],
            "assumptions": task.assumptions,
            "decisions": task.decisions,
            "requirements": [r.to_dict() for r in task.requirements],
            "unresolved_requirements": unresolved,
            "completed_steps": completed,
            "pending_steps": pending,
            "intermediate_outputs": task.relevant_outputs,
            "errors": [e.to_dict() for e in task.errors],
            "recovery_attempts": [r.to_dict() for r in task.recovery_attempts],
            "verification_findings": [v.to_dict() for v in task.verification_results],
            "live_testing_results": [l.to_dict() for l in task.live_run_results],
            "available_capabilities": [
                {
                    "identifier": c["identifier"],
                    "purpose": c["purpose"],
                    "availability": c["availability"],
                    "input_schema": c.get("input_schema", {}),
                }
                for c in available_capabilities
            ],
            "retrieved_memory": retrieved_context or {},
        }

        serialized = json.dumps(payload, default=str)
        if len(serialized) <= effective_max_chars:
            payload["token_budget_info"] = {
                "estimated_tokens": max(1, len(serialized) // 4),
                "token_estimation_method": "heuristic_char_div_4",
                "effective_max_chars": effective_max_chars,
                "active_model_context_window_tokens": active_model_context_window_tokens,
                "compressed": False,
            }
            return payload

        # Context exceeds window: compress intermediate outputs and verbose histories
        compressed_summary = self.compress_context(task)
        compact_payload: Dict[str, Any] = {
            "focus": current_focus,
            "task_id": task.task_id,
            "original_instruction": task.original_instruction,
            "current_objective": task.interpreted_objective,
            "constraints": list(task.intent.constraints) if task.intent else [],
            "assumptions": task.assumptions,
            "decisions": task.decisions,
            "requirements": [r.to_dict() for r in task.requirements],
            "unresolved_requirements": unresolved,
            "compressed_state_summary": json.loads(compressed_summary),
            "available_capabilities": [
                {
                    "identifier": c["identifier"],
                    "purpose": c["purpose"],
                    "availability": c["availability"],
                }
                for c in available_capabilities
            ],
            "retrieved_memory": {
                "factual_memories": (retrieved_context or {}).get(
                    "factual_memories", []
                )[:3],
                "instruction_memories": (retrieved_context or {}).get(
                    "instruction_memories", []
                )[:3],
                "relevant_experiences": [
                    {
                        "objective": e.get("objective"),
                        "outcome": e.get("outcome"),
                        "successful_remedies": e.get("successful_remedies", []),
                        "reusable_procedures": e.get("reusable_procedures", []),
                    }
                    for e in (retrieved_context or {}).get("relevant_experiences", [])[:3]
                ],
            },
        }
        compact_str = json.dumps(compact_payload, default=str)
        compact_payload["token_budget_info"] = {
            "estimated_tokens": max(1, len(compact_str) // 4),
            "token_estimation_method": "heuristic_char_div_4",
            "effective_max_chars": effective_max_chars,
            "active_model_context_window_tokens": active_model_context_window_tokens,
            "compressed": True,
        }
        return compact_payload

    def prepare_task_for_resumption(
        self, task: TaskState, workspace_root: Path
    ) -> List[str]:
        """
        Inspect a restored task before resuming execution.
        Completed steps whose filesystem artifacts still exist are preserved.
        Completed steps whose output files were deleted or invalidated are reset to PENDING.
        Returns a list of diagnostic notes describing any reset steps.
        """
        notes: List[str] = []
        for action in task.planned_actions:
            if action.status == ActionStatus.IN_PROGRESS:
                action.status = ActionStatus.PENDING
                notes.append(
                    f"Reset interrupted in-progress step '{action.step_id}' to PENDING."
                )
            elif action.status == ActionStatus.COMPLETED:
                if action.capability_id in ("fs.write", "fs.update", "fs.mkdir"):
                    rel_path = action.parameters.get("path")
                    if rel_path:
                        target = (workspace_root / rel_path).resolve()
                        if not target.exists():
                            action.status = ActionStatus.PENDING
                            notes.append(
                                f"Artifact '{rel_path}' from completed step '{action.step_id}' "
                                "is missing on disk; marked step as PENDING for re-execution."
                            )
            elif action.status == ActionStatus.FAILED:
                action.status = ActionStatus.PENDING
                notes.append(
                    f"Reset previously failed step '{action.step_id}' to PENDING for resumption."
                )

        task.refresh_action_lists()
        return notes
