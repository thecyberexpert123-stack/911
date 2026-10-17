"""Experience capabilities for recording and retrieving actionable task experiences."""

from __future__ import annotations

from typing import Any, Dict

from agent911.capabilities.base import Capability, CapabilityContext
from agent911.models import CapabilityResult, ExperienceRecord, TaskOutcome, new_id


class RecordExperienceCapability(Capability):
    identifier = "experience.record"
    name = "Record Task Experience"
    purpose = "Record an execution experience lesson into persistent storage."
    input_schema = {
        "required": ["objective", "plan_summary"],
        "properties": {
            "task_id": {"type": "string", "description": "Associated task ID"},
            "objective": {"type": "string", "description": "Task objective"},
            "plan_summary": {"type": "string", "description": "Summary of plan executed"},
            "capabilities_used": {"type": "array", "description": "List of capability IDs"},
            "operations_performed": {"type": "array", "description": "List of operations"},
            "unexpected_failures": {"type": "array", "description": "Observed failures"},
            "identified_causes": {"type": "array", "description": "Diagnosed root causes"},
            "successful_remedies": {"type": "array", "description": "Remedies that worked"},
            "reusable_procedures": {"type": "array", "description": "Reusable steps"},
            "outcome": {"type": "string", "description": "TaskOutcome value"},
            "tags": {"type": "array", "description": "Searchable tags"},
        },
    }
    output_format = {"experience_id": "str", "task_id": "str", "outcome": "str"}

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        raw_outcome = str(params.get("outcome", TaskOutcome.SUCCESS.value)).upper()
        try:
            outcome = TaskOutcome(raw_outcome)
        except ValueError:
            outcome = TaskOutcome.UNVERIFIED

        exp = ExperienceRecord(
            experience_id=new_id("exp"),
            task_id=str(params.get("task_id") or context.task_id or new_id("task")),
            objective=str(params["objective"]),
            plan_summary=str(params["plan_summary"]),
            capabilities_used=list(params.get("capabilities_used", [])),
            operations_performed=list(params.get("operations_performed", [])),
            actual_results=dict(params.get("actual_results", {})),
            verification_summary=dict(params.get("verification_summary", {})),
            live_test_summary=dict(params.get("live_test_summary", {})),
            unexpected_failures=list(params.get("unexpected_failures", [])),
            identified_causes=list(params.get("identified_causes", [])),
            corrections_attempted=list(params.get("corrections_attempted", [])),
            successful_remedies=list(params.get("successful_remedies", [])),
            remaining_limitations=list(params.get("remaining_limitations", [])),
            reusable_procedures=list(params.get("reusable_procedures", [])),
            outcome=outcome,
            tags=list(params.get("tags", [])),
        )
        saved = context.memory_store.record_experience(exp)
        return CapabilityResult(
            capability_id=self.identifier,
            success=True,
            data=saved.to_dict(),
        )


class RetrieveExperienceCapability(Capability):
    identifier = "experience.retrieve"
    name = "Retrieve Task Experiences"
    purpose = "Retrieve relevant past task experiences, remedies, and reusable procedures."
    input_schema = {
        "required": [],
        "properties": {
            "query": {"type": "string", "description": "Search query matching objectives or errors"},
            "task_id": {"type": "string", "description": "Optional task_id filter"},
            "outcome": {"type": "string", "description": "Optional TaskOutcome filter"},
            "limit": {"type": "integer", "description": "Max records to return (default 5)"},
        },
    }
    output_format = {"count": "int", "experiences": "list[dict]"}

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        records = context.memory_store.retrieve_experiences(
            query=params.get("query"),
            task_id=params.get("task_id"),
            outcome=params.get("outcome"),
            limit=int(params.get("limit", 5)),
        )
        return CapabilityResult(
            capability_id=self.identifier,
            success=True,
            data={"count": len(records), "experiences": [r.to_dict() for r in records]},
        )
