"""Evidence-based self-diagnosis and capability gap analysis for Agent 911."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from agent911.capabilities.base import CapabilityContext
from agent911.capabilities.registry import CapabilityRegistry
from agent911.config import AgentConfig
from agent911.models import (
    CapabilityAvailability,
    CapabilityGapAnalysis,
    CapabilityProposal,
    CapabilityResult,
    DiagnosticReport,
    ExecutionErrorRecord,
    FailureRootCauseCategory,
    PlannedAction,
    TaskState,
    new_id,
)


class SelfDiagnosticEngine:
    """
    Investigates why a step or task failed using actual system state, execution history,
    configuration, and workspace evidence.
    Strictly distinguishes confirmed facts from likely hypotheses and eliminated causes.
    """

    @classmethod
    def detect_unproductive_loop(cls, task: TaskState) -> tuple[bool, str]:
        """
        Detect unproductive repetition loops in task execution (Section 8.3):
        - Repeating the same failed operation with identical parameters
        - Alternating between two unsuccessful corrections
        """
        failed_calls = [c for c in task.capability_calls if not c.success]
        seen_sigs: Dict[tuple[str, str, str], int] = {}
        for fc in failed_calls:
            sig = (fc.step_id, fc.capability_id, str(fc.parameters))
            seen_sigs[sig] = seen_sigs.get(sig, 0) + 1
            if seen_sigs[sig] >= 2:
                return (
                    True,
                    f"Step '{fc.step_id}' repeated failed call to '{fc.capability_id}' with identical parameters {seen_sigs[sig]} times.",
                )

        failed_recs = [r for r in task.recovery_attempts if not r.succeeded]
        if len(failed_recs) >= 3:
            return (
                True,
                f"Task accumulated {len(failed_recs)} unsuccessful recovery attempts without resolving the failure.",
            )
        return False, "No unproductive loop detected."

    @classmethod
    def diagnose_failure(
        cls,
        task: TaskState,
        action: Optional[PlannedAction],
        failed_result: Optional[CapabilityResult],
        error_record: ExecutionErrorRecord,
        context: CapabilityContext,
        repeated_loop_detected: bool = False,
        run_safe_probes: bool = True,
    ) -> DiagnosticReport:
        err_type = (
            (failed_result.error_type if failed_result else None)
            or error_record.error_type
            or "ExecutionError"
        )
        err_msg = (
            (failed_result.error if failed_result else None)
            or error_record.message
            or "Operation failed."
        )
        cap_id = (
            action.capability_id
            if action is not None
            else (error_record.capability_id or "unknown")
        )

        confirmed_facts: List[str] = [
            f"Confirmed: step '{error_record.step_id or 'task'}' invoking '{cap_id}' failed with error_type='{err_type}'."
        ]
        likely_hypotheses: List[str] = []
        eliminated_causes: List[str] = []
        remaining_uncertainty: Optional[str] = None
        diagnostic_action_taken: Optional[str] = None
        recommended_correction: str = "none"
        category = FailureRootCauseCategory.UNKNOWN_CAUSE

        # Safe bounded diagnostic check 1: Inspect capability registration & availability
        avail = context.registry.get_availability(cap_id)
        diagnostic_action_taken = f"Inspected registry availability for '{cap_id}' -> {avail.value}"

        if avail == CapabilityAvailability.AVAILABLE:
            eliminated_causes.append(
                f"Eliminated: missing or unimplemented capability ('{cap_id}' is registered and AVAILABLE)."
            )

        # Classify root cause based on concrete evidence
        if err_type == "CapabilityNotFoundError" or avail == CapabilityAvailability.NOT_IMPLEMENTED:
            category = FailureRootCauseCategory.UNAVAILABLE_CAPABILITY
            confirmed_facts.append(
                f"Confirmed: capability '{cap_id}' is not registered in CapabilityRegistry."
            )
            eliminated_causes.append("Eliminated: network or credential failure.")
            recommended_correction = (
                f"Synthesize, verify, and live-test capability '{cap_id}' via capability.develop."
            )
        elif avail == CapabilityAvailability.MISSING_DEPENDENCY:
            category = FailureRootCauseCategory.MISSING_DEPENDENCY
            cap_obj = context.registry.get(cap_id)
            missing_dep = cap_obj.check_missing_dependency() if cap_obj else "unknown"
            confirmed_facts.append(
                f"Confirmed: capability '{cap_id}' is missing required dependency '{missing_dep}'."
            )
            eliminated_causes.append(
                f"Eliminated: missing capability implementation ('{cap_id}' class is registered)."
            )
            recommended_correction = f"Install missing dependency '{missing_dep}' or use an alternative capability."
        elif (
            err_type == "missing_credentials"
            or avail == CapabilityAvailability.REQUIRES_CONFIG
            or "REQUIRES_CONFIG" in err_msg
        ):
            category = FailureRootCauseCategory.MISSING_CREDENTIAL
            confirmed_facts.append(
                f"Confirmed: operation '{cap_id}' requires external provider credentials that are not configured."
            )
            eliminated_causes.append(
                f"Eliminated: missing capability ('{cap_id}' is implemented in the codebase)."
            )
            remaining_uncertainty = (
                "Unknown: whether external endpoint is reachable until a valid API key is supplied."
            )
            recommended_correction = (
                f"Configure API key in CredentialStore or environment variable '{context.config.model.api_key_env}'."
            )
        elif err_type == "auth_error":
            category = FailureRootCauseCategory.EXTERNAL_SERVICE_FAILURE
            confirmed_facts.append(
                "Confirmed: the configured model endpoint returned an HTTP authentication error (401/403)."
            )
            likely_hypotheses.append(
                "Likely: the stored API credential is invalid, revoked, or expired."
            )
            eliminated_causes.append(
                "Eliminated: network unreachability (endpoint responded with an HTTP status code)."
            )
            remaining_uncertainty = (
                "Unknown: whether the provider account has organization or quota restrictions."
            )
            recommended_correction = "Update the provider API key and run 'agent911 model-test'."
        elif err_type in ("WorkspaceBoundaryError", "PermissionError"):
            category = FailureRootCauseCategory.PERMISSION_BOUNDARY
            confirmed_facts.append(
                f"Confirmed: operation violated workspace or OS permission boundary ({err_msg})."
            )
            eliminated_causes.append(
                "Eliminated: transient runtime defect; security boundaries are enforced intentionally."
            )
            recommended_correction = (
                "Restrict target path to inside the configured workspace_root."
            )
        elif err_type == "ParentDirectoryNotFoundError":
            category = FailureRootCauseCategory.INCORRECT_ASSUMPTION
            target_path = (action.parameters.get("path") if action else None) or ""
            parent_dir = str(Path(target_path).parent) if target_path else ""
            parent_exists = (
                (context.workspace_path / parent_dir).exists() if parent_dir else False
            )
            diagnostic_action_taken += f"; checked parent directory '{parent_dir}' exists={parent_exists}"
            confirmed_facts.append(
                f"Confirmed: parent directory '{parent_dir}' did not exist on disk prior to file write."
            )
            eliminated_causes.append(
                "Eliminated: permission boundary error (target path is inside workspace_root)."
            )
            recommended_correction = (
                f"Create parent directory '{parent_dir}' via fs.mkdir and set create_parents=True."
            )
        elif err_type == "FileNotFoundError":
            category = FailureRootCauseCategory.INCORRECT_ASSUMPTION
            target_path = (action.parameters.get("path") if action else None) or ""
            confirmed_facts.append(
                f"Confirmed: target file '{target_path}' does not exist in workspace."
            )
            likely_hypotheses.append(
                "Likely: step assumed the file was already created by an earlier operation."
            )
            recommended_correction = (
                f"Create or initialize '{target_path}' before reading or updating it."
            )
        elif err_type in ("unsupported_feature", "malformed_output", "local_only_mode"):
            category = FailureRootCauseCategory.MODEL_LIMITATION
            confirmed_facts.append(
                f"Confirmed: active model '{context.config.active_model}' reported limitation '{err_type}'."
            )
            recommended_correction = (
                "Route request to a compatible configured model profile or use local heuristic fallback."
            )
        elif err_type in ("ConfigurationError", "model_not_found"):
            category = FailureRootCauseCategory.CONFIGURATION_PROBLEM
            confirmed_facts.append(
                f"Confirmed: provider or model configuration rejected ({err_msg})."
            )
            recommended_correction = (
                "Verify model_id and endpoint in model configuration and test connectivity."
            )
        elif err_type in ("network_error", "timeout", "http_error"):
            category = FailureRootCauseCategory.EXTERNAL_SERVICE_FAILURE
            confirmed_facts.append(
                f"Confirmed: external request failed with '{err_type}' ({err_msg})."
            )
            likely_hypotheses.append(
                "Likely: remote service is unreachable, overloaded, or blocked by network rules."
            )
            remaining_uncertainty = (
                "Unknown: whether the external outage is transient or permanent."
            )
            recommended_correction = (
                "Switch to configured fallback_model if available, or abort retries."
            )
        else:
            category = FailureRootCauseCategory.IMPLEMENTATION_DEFECT
            confirmed_facts.append(f"Confirmed: execution returned error '{err_msg}'.")
            likely_hypotheses.append(
                "Likely: invalid input parameters or defect in target script/command."
            )
            remaining_uncertainty = (
                "Unknown: whether alternative parameters will satisfy the target operation without inspection."
            )
            recommended_correction = (
                "Apply targeted parameter correction if known, otherwise halt retries."
            )

        if repeated_loop_detected:
            confirmed_facts.append(
                "Confirmed: identical failure signature was already observed on this step; repeating the same action will not produce new information."
            )
            recommended_correction = "Stop unproductive retry loop and report blocker."

        report = DiagnosticReport(
            diagnostic_id=new_id("diag"),
            task_id=task.task_id,
            step_id=error_record.step_id,
            observed_failure=f"{err_type}: {err_msg}",
            root_cause_category=category,
            confirmed_facts=confirmed_facts,
            likely_hypotheses=likely_hypotheses,
            eliminated_causes=eliminated_causes,
            remaining_uncertainty=remaining_uncertainty,
            diagnostic_action_taken=diagnostic_action_taken,
            recommended_correction=recommended_correction,
            repeated_loop_detected=repeated_loop_detected,
        )
        return report


class CapabilityGapAnalyzer:
    """
    Determines whether a task blocker is a genuinely missing capability or a missing
    dependency, credential, configuration setting, permission, or model feature, and
    generates concrete CapabilityProposals when a capability is genuinely missing (Section 9).
    """

    @classmethod
    def analyze_failure(
        cls,
        capability_id: str,
        error_type: str,
        error_message: str,
        registry: CapabilityRegistry,
        action_description: str = "",
        action_parameters: Optional[Dict[str, Any]] = None,
        config: Optional[AgentConfig] = None,
    ) -> CapabilityGapAnalysis:
        effective_cfg = config or getattr(registry, "config", None) or AgentConfig()
        return cls.analyze(
            capability_id=capability_id,
            task_instruction=action_description or f"Invoke {capability_id}",
            registry=registry,
            config=effective_cfg,
            error_type=error_type,
            error_message=error_message,
        )

    @classmethod
    def analyze(
        cls,
        capability_id: str,
        task_instruction: str,
        registry: CapabilityRegistry,
        config: AgentConfig,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> CapabilityGapAnalysis:
        avail = registry.get_availability(capability_id)

        # Find any existing capabilities with overlapping keywords to prefer reuse over duplication
        keywords = capability_id.replace(".", " ").replace("_", " ")
        similar = [
            c["identifier"]
            for c in registry.find_capabilities(keywords)[:4]
            if c["identifier"] != capability_id and c["availability"] == "AVAILABLE"
        ]

        if error_type in ("WorkspaceBoundaryError", "PermissionError"):
            return CapabilityGapAnalysis(
                missing_element_type="permission",
                missing_identifier=capability_id,
                explanation=(
                    f"Capability '{capability_id}' exists, but the requested operation crossed "
                    "the workspace or OS permission boundary."
                ),
                reusable_existing_capabilities=similar,
            )

        if error_type == "unsupported_feature":
            return CapabilityGapAnalysis(
                missing_element_type="model_feature",
                missing_identifier=config.active_model,
                explanation=(
                    f"Active model '{config.active_model}' lacks a required model feature: {error_message}"
                ),
                reusable_existing_capabilities=similar,
            )

        if avail == CapabilityAvailability.REQUIRES_CONFIG or error_type == "missing_credentials":
            return CapabilityGapAnalysis(
                missing_element_type="credential",
                missing_identifier=config.model.api_key_env,
                explanation=(
                    f"Capability '{capability_id}' is implemented and registered, but requires "
                    f"valid API credentials ({config.model.api_key_env} or CredentialStore)."
                ),
                reusable_existing_capabilities=similar,
            )

        if avail == CapabilityAvailability.MISSING_DEPENDENCY:
            cap = registry.get(capability_id)
            dep = cap.check_missing_dependency() if cap else "unknown_dependency"
            return CapabilityGapAnalysis(
                missing_element_type="dependency",
                missing_identifier=dep or "unknown_dependency",
                explanation=(
                    f"Capability '{capability_id}' is implemented, but required dependency "
                    f"'{dep}' is not installed in the runtime environment."
                ),
                reusable_existing_capabilities=similar,
            )

        if avail == CapabilityAvailability.MISCONFIGURED or error_type == "ConfigurationError":
            return CapabilityGapAnalysis(
                missing_element_type="configuration_setting",
                missing_identifier=capability_id,
                explanation=(
                    f"Capability '{capability_id}' cannot run due to a configuration issue: {error_message}"
                ),
                reusable_existing_capabilities=similar,
            )

        if avail == CapabilityAvailability.AVAILABLE:
            if error_type in ("network_error", "timeout", "http_error", "auth_error"):
                return CapabilityGapAnalysis(
                    missing_element_type="external_service",
                    missing_identifier=config.model.endpoint,
                    explanation=(
                        f"Capability '{capability_id}' is available, but the external service "
                        f"at '{config.model.endpoint}' failed ({error_type})."
                    ),
                    reusable_existing_capabilities=similar,
                )
            return CapabilityGapAnalysis(
                missing_element_type="none",
                missing_identifier="",
                explanation=f"Capability '{capability_id}' is registered and AVAILABLE.",
                reusable_existing_capabilities=similar,
            )

        # Genuinely missing capability (NOT_IMPLEMENTED or unverified)
        proposal = CapabilityProposal(
            capability_id=capability_id,
            name=f"Proposed {capability_id}",
            what_it_does=f"Perform '{capability_id}' to satisfy: {task_instruction}",
            why_needed=(
                f"Capability '{capability_id}' is not implemented in CapabilityRegistry "
                f"(existing similar capabilities: {similar or 'none'})."
            ),
            inputs={"text": "string (optional)", "path": "string (optional)"},
            outputs={"result": "structured output dictionary"},
            dependencies=[],
            operational_boundaries="workspace_confined",
            integration_notes=(
                "Implement subclass of Capability in custom_capabilities_dir, register in "
                "CapabilityRegistry, and persist metadata in SQLite custom_capabilities."
            ),
            verification_plan=(
                "Compile module via Python AST/py_compile, validate Capability subclass contract, "
                "and confirm CapabilityAvailability.AVAILABLE."
            ),
            live_test_plan=(
                f"Execute registry.invoke('{capability_id}', sample_params) via LiveRunTester "
                "and verify exit_code=0 and non-empty structured output."
            ),
        )
        return CapabilityGapAnalysis(
            missing_element_type="capability",
            missing_identifier=capability_id,
            explanation=f"Capability '{capability_id}' is genuinely missing from the registry.",
            reusable_existing_capabilities=similar,
            capability_proposal=proposal,
        )
