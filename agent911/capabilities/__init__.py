"""Built-in capabilities and registry initialization for Agent 911."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agent911.capabilities.base import Capability, CapabilityContext
from agent911.capabilities.command_execution import RunCommandCapability
from agent911.capabilities.experience_ops import (
    RecordExperienceCapability,
    RetrieveExperienceCapability,
)
from agent911.capabilities.filesystem import (
    ListDirectoryCapability,
    MakeDirectoryCapability,
    ReadFileCapability,
    StatFileCapability,
    UpdateFileCapability,
    WorkspaceBoundaryError,
    WriteFileCapability,
    resolve_in_workspace,
)
from agent911.capabilities.growth import DevelopCapability
from agent911.capabilities.memory_ops import (
    RetrieveMemoryCapability,
    StoreMemoryCapability,
    UpdateMemoryCapability,
)
from agent911.capabilities.model_ops import (
    CheckModelConnectivityCapability,
    DiscoverCapabilitiesCapability,
    QueryModelCapability,
)
from agent911.capabilities.project_inspection import InspectProjectCapability
from agent911.capabilities.registry import (
    CapabilityRegistrationError,
    CapabilityRegistry,
)
from agent911.capabilities.test_execution import RunTestsCapability
from agent911.memory import PersistentMemoryStore


def register_builtin_capabilities(
    registry: CapabilityRegistry,
    memory_store: Optional[PersistentMemoryStore] = None,
) -> None:
    """Register all built-in capabilities and reload persisted custom capabilities."""
    builtins = [
        ListDirectoryCapability(),
        ReadFileCapability(),
        WriteFileCapability(),
        UpdateFileCapability(),
        MakeDirectoryCapability(),
        StatFileCapability(),
        InspectProjectCapability(),
        RunCommandCapability(),
        RunTestsCapability(),
        StoreMemoryCapability(),
        RetrieveMemoryCapability(),
        UpdateMemoryCapability(),
        RecordExperienceCapability(),
        RetrieveExperienceCapability(),
        DiscoverCapabilitiesCapability(),
        CheckModelConnectivityCapability(),
        QueryModelCapability(),
        DevelopCapability(),
    ]
    for cap in builtins:
        registry.register(cap, allow_replace=True)

    if memory_store is not None:
        for custom in memory_store.list_custom_capabilities():
            if custom.get("verified") and custom.get("live_tested"):
                mod_path = Path(custom["module_path"])
                if mod_path.exists():
                    try:
                        registry.load_capability_from_module(
                            module_path=mod_path,
                            class_name=custom.get("class_name"),
                            verified=True,
                            live_tested=True,
                            register_now=True,
                        )
                    except Exception:
                        pass


__all__ = [
    "Capability",
    "CapabilityContext",
    "CapabilityRegistry",
    "CapabilityRegistrationError",
    "WorkspaceBoundaryError",
    "resolve_in_workspace",
    "register_builtin_capabilities",
    "ListDirectoryCapability",
    "ReadFileCapability",
    "WriteFileCapability",
    "UpdateFileCapability",
    "MakeDirectoryCapability",
    "StatFileCapability",
    "InspectProjectCapability",
    "RunCommandCapability",
    "RunTestsCapability",
    "StoreMemoryCapability",
    "RetrieveMemoryCapability",
    "UpdateMemoryCapability",
    "RecordExperienceCapability",
    "RetrieveExperienceCapability",
    "DiscoverCapabilitiesCapability",
    "CheckModelConnectivityCapability",
    "QueryModelCapability",
    "DevelopCapability",
]
