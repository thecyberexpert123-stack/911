"""Capability registry, discovery, dynamic loading, and invocation pipeline."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent911.capabilities.base import Capability, CapabilityContext
from agent911.config import AgentConfig
from agent911.models import CapabilityAvailability, CapabilityResult


class CapabilityRegistrationError(ValueError):
    """Raised when attempting to register an invalid or unimplemented capability."""


class CapabilityRegistry:
    """Genuine registry for discovering, validating, and invoking capabilities."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._capabilities: Dict[str, Capability] = {}

    def register(self, capability: Capability, allow_replace: bool = True) -> None:
        """Register a working Capability instance after checking its contract."""
        if not isinstance(capability, Capability):
            raise CapabilityRegistrationError(
                f"Object {capability!r} does not inherit from Capability."
            )
        if not capability.identifier or not isinstance(capability.identifier, str):
            raise CapabilityRegistrationError("Capability must have a non-empty string identifier.")
        if not capability.name or not capability.purpose:
            raise CapabilityRegistrationError(
                f"Capability '{capability.identifier}' must declare both name and purpose."
            )
        if getattr(capability.execute, "__isabstractmethod__", False):
            raise CapabilityRegistrationError(
                f"Capability '{capability.identifier}' has an unimplemented abstract execute() method."
            )
        # Inspect source of execute to reject empty pass/NotImplementedError stubs
        try:
            src = inspect.getsource(capability.execute).strip()
            lines = [
                line.strip()
                for line in src.splitlines()
                if line.strip() and not line.strip().startswith(("#", '"""', "'''", "def "))
            ]
            if lines in (["pass"], ["..."], ["raise NotImplementedError"], ["raise NotImplementedError()"]):
                raise CapabilityRegistrationError(
                    f"Capability '{capability.identifier}' has a placeholder stub execute() method."
                )
        except OSError:
            pass

        if capability.identifier in self._capabilities and not allow_replace:
            raise CapabilityRegistrationError(
                f"Capability '{capability.identifier}' is already registered."
            )
        self._capabilities[capability.identifier] = capability

    def get(self, identifier: str) -> Optional[Capability]:
        """Look up a registered capability by identifier."""
        return self._capabilities.get(identifier)

    def has(self, identifier: str) -> bool:
        return identifier in self._capabilities

    def get_availability(self, identifier: str) -> CapabilityAvailability:
        """Distinguish available, unavailable, requires_config, and not_implemented capabilities."""
        cap = self._capabilities.get(identifier)
        if cap is None:
            return CapabilityAvailability.NOT_IMPLEMENTED
        return cap.check_availability(self.config)

    def list_capabilities(self, only_available: bool = False) -> List[Dict[str, Any]]:
        """List all registered capabilities with their metadata and availability."""
        items: List[Dict[str, Any]] = []
        for ident in sorted(self._capabilities.keys()):
            cap = self._capabilities[ident]
            avail = cap.check_availability(self.config)
            if only_available and avail != CapabilityAvailability.AVAILABLE:
                continue
            items.append(cap.describe(self.config))
        return items

    def find_capabilities(self, query: str) -> List[Dict[str, Any]]:
        """Search registered capabilities by keyword against identifier, name, and purpose."""
        q_lower = query.lower().strip()
        if not q_lower:
            return self.list_capabilities()
        tokens = [t for t in q_lower.split() if t]
        scored: List[tuple[int, Dict[str, Any]]] = []
        for cap in self._capabilities.values():
            desc = cap.describe(self.config)
            haystack = f"{cap.identifier} {cap.name} {cap.purpose}".lower()
            score = 0
            if q_lower in haystack:
                score += 5
            for tok in tokens:
                if tok in haystack:
                    score += 2
            if score > 0:
                scored.append((score, desc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [d for _, d in scored]

    def invoke(
        self,
        identifier: str,
        params: Dict[str, Any],
        context: CapabilityContext,
    ) -> CapabilityResult:
        """
        Validate and execute a registered capability through the actual execution pipeline.
        Never conceals a failure inside a success result.
        """
        t0 = time.monotonic()
        cap = self._capabilities.get(identifier)
        if cap is None:
            return CapabilityResult(
                capability_id=identifier,
                success=False,
                error=f"Capability '{identifier}' does not exist in the registry.",
                error_type="CapabilityNotFoundError",
                metadata={"availability": CapabilityAvailability.NOT_IMPLEMENTED.value},
            )

        avail = cap.check_availability(self.config)
        if avail != CapabilityAvailability.AVAILABLE:
            return CapabilityResult(
                capability_id=identifier,
                success=False,
                error=(
                    f"Capability '{identifier}' is currently not operational "
                    f"(availability={avail.value})."
                ),
                error_type="CapabilityUnavailableError",
                metadata={"availability": avail.value},
            )

        valid, validation_err = cap.validate_input(params)
        if not valid:
            return CapabilityResult(
                capability_id=identifier,
                success=False,
                error=validation_err or "Invalid input parameters.",
                error_type="InputValidationError",
                metadata={"parameters": params},
            )

        try:
            result = cap.execute(params, context)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if not isinstance(result, CapabilityResult):
                return CapabilityResult(
                    capability_id=identifier,
                    success=False,
                    error=(
                        f"Capability '{identifier}' returned invalid result type "
                        f"'{type(result).__name__}' instead of CapabilityResult."
                    ),
                    error_type="ContractViolationError",
                    metadata={"duration_ms": elapsed_ms},
                )
            result.metadata.setdefault("duration_ms", elapsed_ms)
            return result
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            return CapabilityResult(
                capability_id=identifier,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
                metadata={"duration_ms": elapsed_ms},
            )

    def load_capability_from_module(
        self,
        module_path: str | Path,
        class_name: Optional[str] = None,
        verified: bool = False,
        live_tested: bool = False,
        register_now: bool = True,
    ) -> Capability:
        """Dynamically import a capability class from a Python file and optionally register it."""
        path = Path(module_path).resolve()
        if not path.exists() or not path.is_file():
            raise CapabilityRegistrationError(f"Capability module file not found: {path}")

        mod_name = f"agent911_custom_{path.stem}_{int(path.stat().st_mtime_ns)}"
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec is None or spec.loader is None:
            raise CapabilityRegistrationError(
                f"Could not create module spec for capability file: {path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

        target_cls = None
        if class_name and hasattr(module, class_name):
            cand = getattr(module, class_name)
            if inspect.isclass(cand) and issubclass(cand, Capability) and cand is not Capability:
                target_cls = cand
        else:
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Capability) and obj is not Capability:
                    target_cls = obj
                    break

        if target_cls is None:
            raise CapabilityRegistrationError(
                f"No concrete Capability subclass found in module '{path}'."
            )

        instance: Capability = target_cls()
        instance.is_custom_grown = True
        instance.verified = verified
        instance.live_tested = live_tested
        if register_now:
            self.register(instance, allow_replace=True)
        return instance
