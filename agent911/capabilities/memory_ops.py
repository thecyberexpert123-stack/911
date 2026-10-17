"""Memory capabilities exposing persistent SQLite memory operations to the execution pipeline."""

from __future__ import annotations

from typing import Any, Dict

from agent911.capabilities.base import Capability, CapabilityContext
from agent911.models import CapabilityResult, MemoryCategory, MemoryConfidence


class StoreMemoryCapability(Capability):
    identifier = "memory.store"
    name = "Store Persistent Memory"
    purpose = "Store or update a factual, instruction, episodic, or working-state memory record."
    input_schema = {
        "required": ["key", "content"],
        "properties": {
            "category": {
                "type": "string",
                "description": "FACTUAL, INSTRUCTION, EPISODIC, EXPERIENCE, or WORKING_STATE",
            },
            "key": {"type": "string", "description": "Memory key or topic identifier"},
            "content": {"type": "string", "description": "Memory content to store"},
            "source": {"type": "string", "description": "Source of the information"},
            "confidence": {
                "type": "string",
                "description": "VERIFIED, OBSERVED, ASSUMPTION, or UNVERIFIED",
            },
            "evidence": {"type": "object", "description": "Supporting evidence dictionary"},
        },
    }
    output_format = {"memory_id": "str", "category": "str", "key": "str", "confidence": "str"}

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        raw_cat = str(params.get("category", MemoryCategory.FACTUAL.value)).upper()
        raw_conf = str(params.get("confidence", MemoryConfidence.OBSERVED.value)).upper()
        try:
            category = MemoryCategory(raw_cat)
            confidence = MemoryConfidence(raw_conf)
        except ValueError as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Invalid memory category or confidence: {exc}",
                error_type="InputValidationError",
            )

        rec = context.memory_store.store_memory(
            category=category,
            key=str(params["key"]),
            content=str(params["content"]),
            task_id=context.task_id,
            source=str(params.get("source", "capability")),
            evidence=params.get("evidence") or {},
            confidence=confidence,
        )
        return CapabilityResult(
            capability_id=self.identifier,
            success=True,
            data=rec.to_dict(),
        )


class RetrieveMemoryCapability(Capability):
    identifier = "memory.retrieve"
    name = "Retrieve Persistent Memory"
    purpose = "Retrieve relevant records from persistent memory by query, category, or key."
    input_schema = {
        "required": [],
        "properties": {
            "query": {"type": "string", "description": "Keyword search query"},
            "category": {"type": "string", "description": "Optional memory category filter"},
            "key": {"type": "string", "description": "Optional exact key filter"},
            "limit": {"type": "integer", "description": "Max records to return (default 10)"},
        },
    }
    output_format = {"count": "int", "records": "list[dict]"}

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        query = params.get("query")
        cat_raw = params.get("category")
        key = params.get("key")
        limit = int(params.get("limit", 10))

        cat = None
        if cat_raw:
            try:
                cat = MemoryCategory(str(cat_raw).upper())
            except ValueError as exc:
                return CapabilityResult(
                    capability_id=self.identifier,
                    success=False,
                    error=f"Invalid memory category '{cat_raw}': {exc}",
                    error_type="InputValidationError",
                )

        records = context.memory_store.retrieve_memories(
            query=query, category=cat, key=key, limit=limit
        )
        return CapabilityResult(
            capability_id=self.identifier,
            success=True,
            data={"count": len(records), "records": [r.to_dict() for r in records]},
        )


class UpdateMemoryCapability(Capability):
    identifier = "memory.update"
    name = "Update Persistent Memory"
    purpose = "Update an existing memory record by memory_id."
    input_schema = {
        "required": ["memory_id"],
        "properties": {
            "memory_id": {"type": "string", "description": "Target memory ID"},
            "content": {"type": "string", "description": "Updated content"},
            "confidence": {"type": "string", "description": "Updated confidence level"},
            "evidence": {"type": "object", "description": "Additional evidence"},
        },
    }
    output_format = {"updated": "bool", "record": "dict"}

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        mem_id = str(params["memory_id"])
        conf_raw = params.get("confidence")
        conf = None
        if conf_raw:
            try:
                conf = MemoryConfidence(str(conf_raw).upper())
            except ValueError as exc:
                return CapabilityResult(
                    capability_id=self.identifier,
                    success=False,
                    error=f"Invalid confidence '{conf_raw}': {exc}",
                    error_type="InputValidationError",
                )

        updated = context.memory_store.update_memory(
            memory_id=mem_id,
            content=params.get("content"),
            confidence=conf,
            evidence=params.get("evidence"),
        )
        if updated is None:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Memory record '{mem_id}' not found.",
                error_type="KeyError",
            )
        return CapabilityResult(
            capability_id=self.identifier,
            success=True,
            data={"updated": True, "record": updated.to_dict()},
        )
