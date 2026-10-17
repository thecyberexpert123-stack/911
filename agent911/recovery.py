"""Error diagnosis and targeted recovery engine for Agent 911."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from agent911.capabilities.base import CapabilityContext
from agent911.diagnostics import CapabilityGapAnalyzer, SelfDiagnosticEngine
from agent911.models import (
    CapabilityResult,
    DiagnosticReport,
    ExecutionErrorRecord,
    FailureRootCauseCategory,
    PlannedAction,
    ProblemSolvingStrategy,
    RecoveryAttemptRecord,
    TaskState,
    new_id,
)
from agent911.strategy import StrategySelector


_UNRECOVERABLE_ERROR_TYPES = {
    "WorkspaceBoundaryError",
    "PermissionError",
    "CapabilityUnavailableError",
    "missing_credentials",
    "auth_error",
    "ConfigurationError",
}


@dataclass
class DiagnosisAndRemedy:
    """Structured diagnosis and targeted recovery plan for a failed step."""

    recoverable: bool
    probable_cause: str
    selected_correction: str
    root_cause_category: FailureRootCauseCategory = FailureRootCauseCategory.UNKNOWN_CAUSE
    diagnostic_report: Optional[DiagnosticReport] = None
    pre_action: Optional[Tuple[str, Dict[str, Any]]] = None  # (capability_id, params) to run first
    updated_parameters: Optional[Dict[str, Any]] = None
    matched_experience_id: Optional[str] = None
    notes: str = ""


class RecoveryEngine:
    """
    Diagnoses execution and evaluation failures, selects targeted corrections,
    prevents infinite retry loops, and re-executes only the affected step.
    """

    def __init__(self, max_attempts_per_step: int = 2) -> None:
        self.max_attempts_per_step = max_attempts_per_step
        self._attempt_counts: Dict[str, int] = {}
        self._seen_failure_signatures: Set[Tuple[str, str, str, str]] = set()
        self.diagnostic_engine = SelfDiagnosticEngine()
        self.gap_analyzer = CapabilityGapAnalyzer()
        self.strategy_selector = StrategySelector()

    def can_attempt_recovery(
        self,
        action: PlannedAction,
        error_record: ExecutionErrorRecord,
        task: Optional[TaskState] = None,
    ) -> Tuple[bool, str]:
        """Check retry limits, repeated-failure signatures, unproductive loops, and unrecoverable error types."""
        if task is not None:
            loop_detected, loop_desc = self.diagnostic_engine.detect_unproductive_loop(task)
            if loop_detected:
                return False, f"Unproductive repetition loop detected: {loop_desc}"

        if error_record.error_type in _UNRECOVERABLE_ERROR_TYPES:
            return (
                False,
                f"Error type '{error_record.error_type}' is an unrecoverable boundary or configuration condition.",
            )

        count = self._attempt_counts.get(action.step_id, 0)
        if count >= self.max_attempts_per_step:
            return (
                False,
                f"Step '{action.step_id}' reached maximum recovery limit ({self.max_attempts_per_step}).",
            )

        signature = (
            action.step_id,
            action.capability_id,
            error_record.error_type,
            error_record.message[:120],
        )
        if signature in self._seen_failure_signatures and count > 0:
            return (
                False,
                f"Repeated identical failure detected on step '{action.step_id}'; stopping retries.",
            )

        return True, "Recovery eligible."

    def diagnose(
        self,
        task: TaskState,
        action: PlannedAction,
        failed_result: CapabilityResult,
        error_record: ExecutionErrorRecord,
        context: CapabilityContext,
    ) -> DiagnosisAndRemedy:
        """Analyze failure evidence and historical experiences to select a targeted correction."""
        diag_report = self.diagnostic_engine.diagnose_failure(
            task=task,
            action=action,
            failed_result=failed_result,
            error_record=error_record,
            context=context,
            run_safe_probes=True,
        )
        task.diagnostic_reports.append(diag_report.to_dict())

        err_type = failed_result.error_type or error_record.error_type
        err_msg = failed_result.error or error_record.message

        # Perform capability gap analysis when capability/dependency/config/model failures occur
        if err_type in (
            "CapabilityNotFoundError",
            "CapabilityUnavailableError",
            "ImportError",
            "ModuleNotFoundError",
            "missing_credentials",
            "auth_error",
            "ConfigurationError",
            "PermissionError",
            "WorkspaceBoundaryError",
            "ModelCapabilityLimitationError",
            "connection_error",
            "timeout",
            "http_error",
        ):
            gap = self.gap_analyzer.analyze_failure(
                capability_id=action.capability_id,
                error_type=err_type,
                error_message=err_msg,
                registry=context.registry,
                action_description=action.description,
                action_parameters=action.parameters,
            )
            task.capability_gap_analyses.append(gap.to_dict())

        # Adapt task strategy based on diagnosed root cause category (Section 4.3)
        prev_strat = task.selected_strategy
        new_strat, strat_reason = self.strategy_selector.adapt_strategy_on_failure(
            task=task,
            root_cause=diag_report.root_cause_category,
            failed_action=action,
            reason=f"{err_type}: {err_msg}",
        )
        if new_strat.value != prev_strat:
            task.decisions.append(f"Adapted strategy to {new_strat.value}: {strat_reason}")

        can_try, reason = self.can_attempt_recovery(action, error_record, task=task)
        if not can_try:
            return DiagnosisAndRemedy(
                recoverable=False,
                probable_cause=error_record.message,
                selected_correction="none",
                root_cause_category=diag_report.root_cause_category,
                diagnostic_report=diag_report,
                notes=reason,
            )

        # 1. Missing parent directory when writing a file -> create directory via fs.mkdir
        if err_type == "ParentDirectoryNotFoundError":
            missing_parent = failed_result.metadata.get("missing_parent")
            if not missing_parent and "path" in action.parameters:
                p_str = str(action.parameters["path"])
                if "/" in p_str:
                    missing_parent = p_str.rsplit("/", 1)[0]
            if missing_parent:
                new_params = dict(action.parameters)
                new_params["create_parents"] = True
                return DiagnosisAndRemedy(
                    recoverable=True,
                    probable_cause=f"Parent directory '{missing_parent}' did not exist before fs.write.",
                    selected_correction=f"Create missing parent directory '{missing_parent}' via fs.mkdir and set create_parents=True.",
                    pre_action=("fs.mkdir", {"path": missing_parent, "parents": True}),
                    updated_parameters=new_params,
                )

        # 2. Missing capability in registry -> develop capability via capability.develop
        if err_type == "CapabilityNotFoundError":
            cap_id = action.capability_id
            return DiagnosisAndRemedy(
                recoverable=True,
                probable_cause=f"Capability '{cap_id}' is not yet implemented or registered.",
                selected_correction=f"Develop, verify, and live-test missing capability '{cap_id}' via capability.develop.",
                pre_action=(
                    "capability.develop",
                    {
                        "capability_id": cap_id,
                        "name": f"Auto-developed {cap_id}",
                        "purpose": action.description or f"Execute {cap_id}",
                        "behavior_spec": f"{cap_id} {action.description}",
                        "test_params": dict(action.parameters),
                    },
                ),
                updated_parameters=dict(action.parameters),
            )

        # 3. Updating a file that does not yet exist -> create initial file first if append_text is used
        if action.capability_id == "fs.update" and err_type == "FileNotFoundError":
            target_path = action.parameters.get("path")
            append_text = action.parameters.get("append_text")
            if target_path and append_text is not None:
                return DiagnosisAndRemedy(
                    recoverable=True,
                    probable_cause=f"Target file '{target_path}' did not exist before fs.update.",
                    selected_correction=f"Initialize '{target_path}' via fs.write before applying update.",
                    pre_action=(
                        "fs.write",
                        {"path": target_path, "content": "", "create_parents": True},
                    ),
                    updated_parameters=dict(action.parameters),
                )

        # 4. Consult persistent Experience Memory for a previously recorded parameter fix
        past_experiences = context.memory_store.retrieve_experiences(
            query=f"{action.capability_id} {err_type} {err_msg}", limit=3
        )
        for exp in past_experiences:
            if exp.successful_remedies:
                remedy = exp.successful_remedies[0]
                if "create_parents" in remedy and action.capability_id == "fs.write":
                    new_p = dict(action.parameters)
                    new_p["create_parents"] = True
                    return DiagnosisAndRemedy(
                        recoverable=True,
                        probable_cause=f"Matched historical failure ({err_type}).",
                        selected_correction=remedy,
                        root_cause_category=diag_report.root_cause_category,
                        diagnostic_report=diag_report,
                        updated_parameters=new_p,
                        matched_experience_id=exp.experience_id,
                    )

        # 5. If model.generate failed and a fallback model is configured, switch active model to fallback
        if (
            action.capability_id == "model.generate"
            and context.config.fallback_model
            and context.config.fallback_model != context.config.active_model
            and context.config.fallback_model in context.config.models
            and context.config.models[context.config.fallback_model].enabled
        ):
            fb = context.config.fallback_model
            return DiagnosisAndRemedy(
                recoverable=True,
                probable_cause=f"Primary model '{context.config.active_model}' failed with '{err_type}': {err_msg}",
                selected_correction=f"Switch active model to configured fallback model '{fb}' and retry step.",
                updated_parameters=dict(action.parameters),
                notes=f"switch_to_fallback_model:{fb}",
            )

        # 6. Check if action parameters contain a fallback_parameters or recovery_parameters hint
        if isinstance(action.parameters.get("_recovery_parameters"), dict):
            fallback_p = dict(action.parameters["_recovery_parameters"])
            return DiagnosisAndRemedy(
                recoverable=True,
                probable_cause=f"Operation '{action.capability_id}' failed with '{err_type}': {err_msg}",
                selected_correction="Apply targeted parameter correction for step re-execution.",
                updated_parameters=fallback_p,
            )

        return DiagnosisAndRemedy(
            recoverable=False,
            probable_cause=f"{err_type}: {err_msg}",
            selected_correction="none",
            notes="No automated deterministic correction applicable for this failure.",
        )

    def attempt_step_recovery(
        self,
        task: TaskState,
        action: PlannedAction,
        failed_result: CapabilityResult,
        error_record: ExecutionErrorRecord,
        context: CapabilityContext,
    ) -> Tuple[bool, Optional[CapabilityResult]]:
        """
        Execute targeted recovery for a failed step:
        1. Diagnose probable cause and select correction
        2. Record failure signature and increment step retry counter
        3. Execute pre-action (if any) and update step parameters
        4. Re-execute only the affected step
        5. Record RecoveryAttemptRecord on task
        """
        signature = (
            action.step_id,
            action.capability_id,
            error_record.error_type,
            error_record.message[:120],
        )
        diagnosis = self.diagnose(task, action, failed_result, error_record, context)
        self._seen_failure_signatures.add(signature)
        self._attempt_counts[action.step_id] = self._attempt_counts.get(action.step_id, 0) + 1

        if not diagnosis.recoverable:
            attempt = RecoveryAttemptRecord(
                attempt_id=new_id("rec"),
                task_id=task.task_id,
                step_id=action.step_id,
                failed_operation=f"{action.capability_id}({action.parameters})",
                probable_cause=diagnosis.probable_cause,
                selected_correction="none",
                succeeded=False,
                notes=diagnosis.notes,
            )
            task.recovery_attempts.append(attempt)
            return False, None

        # Run pre-action if needed (e.g., fs.mkdir or capability.develop)
        if diagnosis.pre_action is not None:
            pre_cap, pre_params = diagnosis.pre_action
            pre_res = context.registry.invoke(pre_cap, pre_params, context)
            context.memory_store.record_execution_log(
                task_id=task.task_id,
                step_id=action.step_id,
                capability_id=pre_cap,
                operation=f"recovery_pre_action:{pre_cap}",
                input_data=pre_params,
                output_data=pre_res.data,
                success=pre_res.success,
                error=pre_res.error,
                duration_ms=float(pre_res.metadata.get("duration_ms", 0.0)),
            )
            if not pre_res.success:
                attempt = RecoveryAttemptRecord(
                    attempt_id=new_id("rec"),
                    task_id=task.task_id,
                    step_id=action.step_id,
                    failed_operation=f"{action.capability_id}({action.parameters})",
                    probable_cause=diagnosis.probable_cause,
                    selected_correction=diagnosis.selected_correction,
                    correction_parameters=pre_params,
                    succeeded=False,
                    notes=f"Pre-action '{pre_cap}' failed: {pre_res.error}",
                )
                task.recovery_attempts.append(attempt)
                return False, pre_res

        if diagnosis.notes.startswith("switch_to_fallback_model:"):
            fb_id = diagnosis.notes.split(":", 1)[1]
            context.model_interface.switch_active_model(fb_id, set_as_default=False)

        if diagnosis.updated_parameters is not None:
            action.parameters = diagnosis.updated_parameters

        # Re-execute the affected step
        retry_res = context.registry.invoke(action.capability_id, action.parameters, context)
        succeeded = retry_res.success
        if succeeded:
            error_record.recovered = True

        if diagnosis.matched_experience_id:
            context.memory_store.record_experience_usefulness(
                diagnosis.matched_experience_id,
                helpful=succeeded,
                prevented_failure=succeeded,
                caused_problem=not succeeded,
            )

        attempt = RecoveryAttemptRecord(
            attempt_id=new_id("rec"),
            task_id=task.task_id,
            step_id=action.step_id,
            failed_operation=f"{action.capability_id}",
            probable_cause=diagnosis.probable_cause,
            selected_correction=diagnosis.selected_correction,
            correction_parameters=dict(action.parameters),
            succeeded=succeeded,
            notes="Step re-execution succeeded." if succeeded else f"Retry failed: {retry_res.error}",
        )
        task.recovery_attempts.append(attempt)
        return succeeded, retry_res
