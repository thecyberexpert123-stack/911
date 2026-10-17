"""Base capability contract and execution context for Agent 911."""

from __future__ import annotations

import importlib.util
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent911.config import AgentConfig
from agent911.memory import PersistentMemoryStore
from agent911.model_provider import ModelInterface
from agent911.models import CapabilityAvailability, CapabilityResult


@dataclass
class CapabilityContext:
    """Runtime execution context passed to invoked capabilities."""

    config: AgentConfig
    memory_store: PersistentMemoryStore
    model_interface: ModelInterface
    registry: Any  # CapabilityRegistry
    task_id: Optional[str] = None
    step_id: Optional[str] = None

    @property
    def workspace_path(self) -> Path:
        return self.config.workspace_path


class Capability(ABC):
    """
    Contract that every registered capability in Agent 911 must implement.
    Capabilities must perform genuine work, declare side effects, dependencies,
    and permission boundaries, and never conceal errors behind fake success responses.
    """

    identifier: str = ""
    name: str = ""
    purpose: str = ""
    input_schema: Dict[str, Any] = {}
    output_format: Dict[str, Any] = {}
    error_behavior: str = "Returns CapabilityResult(success=False, error=..., error_type=...)"
    required_config: List[str] = []
    required_dependencies: List[str] = []
    operational_limitations: List[str] = []
    side_effects: List[str] = ["read_only"]
    permission_boundary: str = "workspace_confined"
    verification_evidence: Dict[str, Any] = {}
    is_custom_grown: bool = False
    verified: bool = True
    live_tested: bool = True

    def validate_input(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate input parameters against declared input_schema."""
        if not isinstance(params, dict):
            return False, "Capability parameters must be a dictionary."
        required_keys = self.input_schema.get("required", [])
        properties = self.input_schema.get("properties", {})

        for req_key in required_keys:
            if req_key not in params or params[req_key] is None:
                return (
                    False,
                    f"Missing required parameter '{req_key}' for capability '{self.identifier}'.",
                )

        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for key, val in params.items():
            if val is None or key not in properties:
                continue
            expected_type_name = properties[key].get("type")
            if expected_type_name in type_map:
                py_type = type_map[expected_type_name]
                if expected_type_name == "integer" and isinstance(val, bool):
                    return False, f"Parameter '{key}' must be an integer, got boolean."
                if not isinstance(val, py_type):
                    return (
                        False,
                        f"Parameter '{key}' for '{self.identifier}' expected type '{expected_type_name}', "
                        f"got '{type(val).__name__}'.",
                    )
        return True, None

    def check_missing_dependency(self) -> Optional[str]:
        """Return the name of the first missing Python package or binary dependency, if any."""
        for dep in self.required_dependencies:
            if dep.startswith("bin:"):
                binary = dep.split(":", 1)[1]
                if shutil.which(binary) is None:
                    return dep
            else:
                if importlib.util.find_spec(dep) is None and shutil.which(dep) is None:
                    return dep
        return None

    def check_availability(self, config: AgentConfig) -> CapabilityAvailability:
        """Check whether the capability is currently operational in the configured environment."""
        if self.is_custom_grown and not (self.verified and self.live_tested):
            return CapabilityAvailability.UNAVAILABLE

        missing_dep = self.check_missing_dependency()
        if missing_dep is not None:
            return CapabilityAvailability.MISSING_DEPENDENCY

        for cfg_key in self.required_config:
            if cfg_key == "model_api_key":
                if not config.model.resolve_api_key(
                    config.credential_store
                ) and config.model.provider not in (
                    "deterministic",
                    "ollama",
                ):
                    return CapabilityAvailability.REQUIRES_CONFIG
        return CapabilityAvailability.AVAILABLE

    def describe(self, config: AgentConfig) -> Dict[str, Any]:
        """Return a structured description of the capability, its contract, and its availability."""
        return {
            "identifier": self.identifier,
            "name": self.name,
            "purpose": self.purpose,
            "input_schema": self.input_schema,
            "output_format": self.output_format,
            "error_behavior": self.error_behavior,
            "required_config": self.required_config,
            "required_dependencies": list(self.required_dependencies),
            "operational_limitations": list(self.operational_limitations),
            "side_effects": list(self.side_effects),
            "permission_boundary": self.permission_boundary,
            "verification_evidence": dict(self.verification_evidence)
            if self.verification_evidence
            else {
                "verified": self.verified,
                "live_tested": self.live_tested,
                "is_custom_grown": self.is_custom_grown,
            },
            "availability": self.check_availability(config).value,
            "is_custom_grown": self.is_custom_grown,
            "verified": self.verified,
            "live_tested": self.live_tested,
        }

    @abstractmethod
    def execute(
        self, params: Dict[str, Any], context: CapabilityContext
    ) -> CapabilityResult:
        """Perform the capability's operation and return a structured CapabilityResult."""
        raise NotImplementedError
