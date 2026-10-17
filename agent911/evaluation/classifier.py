"""Task outcome classification combining Verification and Live Run Testing evidence."""

from __future__ import annotations

from typing import Tuple

from agent911.models import (
    ActionStatus,
    LiveTestStatus,
    TaskOutcome,
    TaskState,
    VerificationStatus,
)


_BLOCKING_ERROR_TYPES = {
    "CapabilityUnavailableError",
    "missing_credentials",
    "auth_error",
    "WorkspaceBoundaryError",
    "PermissionError",
    "ConfigurationError",
}


class OutcomeClassifier:
    """
    Classifies a task's final outcome strictly from execution evidence and the
    two formal evaluation mechanisms (Verification and Live Run Testing).
    """

    @staticmethod
    def classify(task: TaskState) -> Tuple[TaskOutcome, str]:
        # 1. Check for unrecovered blocking conditions (missing credentials, permissions, workspace boundary)
        unrecovered_errors = [e for e in task.errors if not e.recovered]
        blocking_errors = [
            e for e in unrecovered_errors if e.error_type in _BLOCKING_ERROR_TYPES
        ]

        required_verifs = [v for v in task.verification_results if v.required]
        required_live = [l for l in task.live_run_results if l.required]

        passed_verifs = [
            v for v in required_verifs if v.status == VerificationStatus.PASSED
        ]
        failed_verifs = [
            v for v in required_verifs if v.status == VerificationStatus.FAILED
        ]
        inconclusive_verifs = [
            v
            for v in required_verifs
            if v.status in (VerificationStatus.INCONCLUSIVE, VerificationStatus.UNAVAILABLE)
        ]

        passed_live = [l for l in required_live if l.status == LiveTestStatus.PASSED]
        failed_live = [
            l
            for l in required_live
            if l.status in (LiveTestStatus.FAILED, LiveTestStatus.TIMED_OUT)
        ]
        unavailable_live = [
            l for l in required_live if l.status == LiveTestStatus.UNAVAILABLE
        ]

        completed_req_steps = [
            a
            for a in task.planned_actions
            if a.required and a.status == ActionStatus.COMPLETED
        ]
        total_req_steps = [a for a in task.planned_actions if a.required]

        # Check explicit requirement satisfaction (excluding the generic step-completion check)
        satisfied_explicit_reqs = [r for r in task.requirements if r.required and r.satisfied]
        unsatisfied_explicit_reqs = [
            r for r in task.requirements if r.required and r.satisfied is False
        ]

        # If blocked and no explicit requirement was satisfied
        if blocking_errors and not satisfied_explicit_reqs and not completed_req_steps:
            reasons = "; ".join(f"{e.error_type}: {e.message}" for e in blocking_errors)
            summary = f"Task BLOCKED due to environmental or configuration condition: {reasons}"
            task.final_outcome = TaskOutcome.BLOCKED
            task.outcome_summary = summary
            return TaskOutcome.BLOCKED, summary

        # If no verification or live test ran at all
        if not required_verifs and not required_live:
            summary = "Task UNVERIFIED: neither Verification nor Live Run Testing produced evidence."
            task.final_outcome = TaskOutcome.UNVERIFIED
            task.outcome_summary = summary
            return TaskOutcome.UNVERIFIED, summary

        # Check if there are failures in either formal evaluation mechanism
        has_failures = bool(failed_verifs or failed_live or unsatisfied_explicit_reqs)
        has_meaningful_progress = bool(
            satisfied_explicit_reqs
            or (len(completed_req_steps) > 0 and len(completed_req_steps) < len(total_req_steps))
            or (len(passed_verifs) > 1 and has_failures)
        )

        if has_failures:
            if has_meaningful_progress and (satisfied_explicit_reqs or passed_live):
                summary = (
                    f"Task achieved PARTIAL_SUCCESS: {len(satisfied_explicit_reqs)} requirement(s) "
                    f"satisfied, but {len(failed_verifs)} verification check(s) and "
                    f"{len(failed_live)} live test(s) failed."
                )
                task.final_outcome = TaskOutcome.PARTIAL_SUCCESS
                task.outcome_summary = summary
                return TaskOutcome.PARTIAL_SUCCESS, summary

            if blocking_errors:
                reasons = "; ".join(f"{e.error_type}: {e.message}" for e in blocking_errors)
                summary = f"Task BLOCKED: {reasons}"
                task.final_outcome = TaskOutcome.BLOCKED
                task.outcome_summary = summary
                return TaskOutcome.BLOCKED, summary

            fail_details = [
                v.failure_explanation or v.observed_result for v in failed_verifs
            ] + [l.stderr or f"exit_code={l.exit_code}" for l in failed_live]
            summary = (
                f"Task FAILED: {len(failed_verifs)} verification check(s) and "
                f"{len(failed_live)} live test(s) failed. Details: {'; '.join(fail_details[:3])}"
            )
            task.final_outcome = TaskOutcome.FAILED
            task.outcome_summary = summary
            return TaskOutcome.FAILED, summary

        # No hard failures, check if required checks were inconclusive or unavailable
        if inconclusive_verifs or unavailable_live or not required_live:
            summary = (
                f"Task UNVERIFIED: {len(inconclusive_verifs)} verification check(s) inconclusive/unavailable "
                f"and {len(unavailable_live)} live test(s) unavailable (live tests run: {len(required_live)})."
            )
            task.final_outcome = TaskOutcome.UNVERIFIED
            task.outcome_summary = summary
            return TaskOutcome.UNVERIFIED, summary

        # All required verification checks PASSED and all required live tests PASSED
        summary = (
            f"Task SUCCEEDED: all {len(passed_verifs)} required Verification check(s) "
            f"and {len(passed_live)} Live Run Test(s) passed with objective evidence."
        )
        task.final_outcome = TaskOutcome.SUCCESS
        task.outcome_summary = summary
        return TaskOutcome.SUCCESS, summary
