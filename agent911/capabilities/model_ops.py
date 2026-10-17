"""Capability discovery and AI model integration capabilities."""

from __future__ import annotations

from typing import Any, Dict

from agent911.capabilities.base import Capability, CapabilityContext
from agent911.config import AgentConfig
from agent911.model_provider import ModelProviderError, ProviderConnectivityStatus
from agent911.models import CapabilityAvailability, CapabilityResult


class DiscoverCapabilitiesCapability(Capability):
    identifier = "capability.discover"
    name = "Discover Capabilities"
    purpose = (
        "Discover registered capabilities in Agent 911, inspect their declared schemas, "
        "and verify their operational availability."
    )
    input_schema = {
        "required": [],
        "properties": {
            "query": {"type": "string", "description": "Optional search keyword"},
            "only_available": {
                "type": "boolean",
                "description": "Filter to only currently AVAILABLE capabilities",
            },
        },
    }
    output_format = {"count": "int", "capabilities": "list[dict]"}

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        query = params.get("query")
        only_available = bool(params.get("only_available", False))
        if query:
            caps = context.registry.find_capabilities(str(query))
            if only_available:
                caps = [
                    c for c in caps if c.get("availability") == CapabilityAvailability.AVAILABLE.value
                ]
        else:
            caps = context.registry.list_capabilities(only_available=only_available)
        return CapabilityResult(
            capability_id=self.identifier,
            success=True,
            data={"count": len(caps), "capabilities": caps},
        )


class CheckModelConnectivityCapability(Capability):
    identifier = "model.connectivity"
    name = "Check AI Model Connectivity"
    purpose = (
        "Check actual connectivity to the configured AI model provider without fabricating "
        "a live connection."
    )
    input_schema = {"required": [], "properties": {}}
    output_format = {
        "status": "str",
        "provider": "str",
        "model_id": "str",
        "provider_reached": "bool",
        "is_live_provider": "bool",
        "details": "str",
    }

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        report = context.model_interface.check_connectivity()
        # Note: checking connectivity is a diagnostic capability that succeeds in returning
        # the honest report, while recording whether the live provider was reached.
        return CapabilityResult(
            capability_id=self.identifier,
            success=True,
            data=report.to_dict(),
            metadata={
                "provider_reached": report.provider_reached,
                "connectivity_status": report.status.value,
            },
        )


class QueryModelCapability(Capability):
    identifier = "model.query"
    name = "Query AI Model"
    purpose = "Send a reasoning or analysis prompt to the configured AI model provider."
    input_schema = {
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "User prompt to send to the AI model"},
            "system_prompt": {
                "type": "string",
                "description": "Optional system instruction",
            },
            "expect_json": {
                "type": "boolean",
                "description": "Whether to parse the model response as a JSON object",
            },
        },
    }
    output_format = {
        "content": "str",
        "structured_data": "dict | None",
        "provider_reached": "bool",
        "is_simulated": "bool",
    }
    required_config = ["model_api_key"]

    def check_availability(self, config: AgentConfig) -> CapabilityAvailability:
        if config.model.provider == "local_only":
            return CapabilityAvailability.UNAVAILABLE
        if config.model.provider in ("deterministic", "ollama"):
            return CapabilityAvailability.AVAILABLE
        if not config.model.resolve_api_key():
            return CapabilityAvailability.REQUIRES_CONFIG
        return CapabilityAvailability.AVAILABLE

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        prompt = str(params["prompt"])
        sys_prompt = str(
            params.get(
                "system_prompt",
                "You are Agent 911's reasoning engine. Provide concise, accurate output.",
            )
        )
        expect_json = bool(params.get("expect_json", False))

        try:
            resp = context.model_interface.generate(
                system_prompt=sys_prompt,
                user_prompt=prompt,
                expect_json=expect_json,
            )
            return CapabilityResult(
                capability_id=self.identifier,
                success=True,
                data={
                    "content": resp.content,
                    "structured_data": resp.structured_data,
                    "provider": resp.provider,
                    "model_id": resp.model_id,
                    "provider_reached": resp.provider_reached,
                    "is_simulated": resp.is_simulated,
                    "latency_ms": resp.latency_ms,
                },
                metadata={
                    "provider_reached": resp.provider_reached,
                    "is_simulated": resp.is_simulated,
                },
            )
        except ModelProviderError as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=str(exc),
                error_type=exc.error_type,
                metadata={"status_code": exc.status_code, "retryable": exc.retryable},
            )
