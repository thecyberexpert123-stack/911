"""Improvement Proposal System and Controlled Self-Improvement Engine for Agent 911."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent911.models import (
    ImprovementProposal,
    ImprovementStatus,
    LiveTestStatus,
    MemoryCategory,
    MemoryConfidence,
    SelfImprovementRunRecord,
    TaskRequirement,
    VerificationStatus,
    new_id,
    utc_now_iso,
)


_FORBIDDEN_SECURITY_TARGETS = {
    "workspace_root",
    "credentials_vault_path",
    "allow_outside_workspace",
    "disable_verification",
    "disable_live_testing",
    "disable_secret_redaction",
}


class ImprovementEngine:
    """
    Detects recurring or significant operational weaknesses from actual SQLite execution
    records, generates prioritized ImprovementProposals, and executes Controlled
    Self-Improvement with Verification, Live Run Testing, and automatic rollback.
    """

    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator
        self.memory_store = orchestrator.memory_store

    def detect_and_propose(self) -> List[ImprovementProposal]:
        """
        Inspect actual task history, execution logs, and experience usefulness records
        to generate and persist evidence-backed ImprovementProposals.
        """
        existing_proposals = {
            p.title: p for p in self.memory_store.list_improvement_proposals()
        }
        new_or_updated: List[ImprovementProposal] = []

        tasks_summary = self.memory_store.list_tasks(limit=50)
        loaded_tasks = []
        for t_row in tasks_summary:
            t_obj = self.memory_store.load_task(t_row["task_id"])
            if t_obj is not None:
                loaded_tasks.append(t_obj)

        # 1. Check for repeated ParentDirectoryNotFoundError across tasks/steps
        parent_dir_errors: List[str] = []
        for t in loaded_tasks:
            for err in t.errors:
                if err.error_type == "ParentDirectoryNotFoundError":
                    parent_dir_errors.append(f"{t.task_id}:{err.step_id}")

        if len(parent_dir_errors) >= 1:
            is_recurring = len(parent_dir_errors) >= 2
            qual = (
                f"Recurring across {len(parent_dir_errors)} step(s)"
                if is_recurring
                else "Observed in 1 task execution (single-event qualification)"
            )
            title = "Enable proactive create_parents=True on nested fs.write paths"
            if title not in existing_proposals:
                prop = ImprovementProposal(
                    proposal_id=new_id("prop"),
                    title=title,
                    category="recovery_rule",
                    observed_problem=(
                        f"{qual}: fs.write failed with ParentDirectoryNotFoundError "
                        f"on {parent_dir_errors}, requiring reactive recovery retries."
                    ),
                    supporting_evidence={
                        "error_type": "ParentDirectoryNotFoundError",
                        "occurrence_count": len(parent_dir_errors),
                        "affected_task_steps": parent_dir_errors,
                        "is_recurring": is_recurring,
                    },
                    likely_cause=(
                        "Default fs.write planning leaves create_parents=False unless a prior "
                        "experience happens to match the query."
                    ),
                    proposed_change=(
                        "Persist verified operational policy 'policy.fs_write.create_parents=true' "
                        "so TaskPlanner proactively sets create_parents=True for nested paths."
                    ),
                    expected_benefits=[
                        "Eliminates avoidable ParentDirectoryNotFoundError step failures.",
                        "Reduces retry count and task execution latency on nested file creation.",
                    ],
                    potential_regressions=[
                        "May create intermediate directories when a caller intended to fail if parent was absent."
                    ],
                    required_dependencies=[],
                    implementation_scope="bounded_planner_policy",
                    test_plan=(
                        "Verify policy memory is persisted in SQLite and live-test planning a nested "
                        "fs.write action to confirm create_parents=True is set proactively."
                    ),
                    success_conditions=[
                        "Verification check 'memory_stored' for 'policy.fs_write.create_parents' passes.",
                        "Live Run Test confirms nested fs.write step succeeds on first attempt with 0 retries.",
                    ],
                    priority="HIGH" if is_recurring else "MEDIUM",
                    priority_rationale=(
                        f"Directly prevents avoidable step failure ({len(parent_dir_errors)} observed occurrence(s)) "
                        "with low regression risk."
                    ),
                    status=ImprovementStatus.PROPOSED,
                    action_payload={
                        "action_type": "enable_default_create_parents",
                        "policy_key": "policy.fs_write.create_parents",
                        "policy_value": "true",
                    },
                )
                self.memory_store.save_improvement_proposal(prop)
                new_or_updated.append(prop)

        # 2. Check forexperiences that caused problems or were repeatedly irrelevant
        all_exps = self.memory_store.retrieve_experiences(
            limit=50, include_superseded=False
        )
        for exp in all_exps:
            if exp.times_caused_problem > 0 or exp.times_irrelevant >= 2:
                title = f"Supersede counterproductive experience '{exp.experience_id}'"
                if title not in existing_proposals:
                    prop = ImprovementProposal(
                        proposal_id=new_id("prop"),
                        title=title,
                        category="retrieval_ranking",
                        observed_problem=(
                            f"Experience '{exp.experience_id}' (from task {exp.task_id}) was retrieved "
                            f"{exp.times_retrieved} time(s) and recorded times_caused_problem={exp.times_caused_problem}, "
                            f"times_irrelevant={exp.times_irrelevant}."
                        ),
                        supporting_evidence={
                            "experience_id": exp.experience_id,
                            "task_id": exp.task_id,
                            "times_retrieved": exp.times_retrieved,
                            "times_caused_problem": exp.times_caused_problem,
                            "times_irrelevant": exp.times_irrelevant,
                        },
                        likely_cause="Historical lesson does not generalize to subsequent tasks.",
                        proposed_change=(
                            f"Mark experience '{exp.experience_id}' as superseded=True so it is excluded "
                            "from active retrieval while preserving its audit record."
                        ),
                        expected_benefits=[
                            "Prevents misleading historical lesson from polluting future task context and strategy selection."
                        ],
                        potential_regressions=["None; historical record remains inspectable."],
                        required_dependencies=[],
                        implementation_scope="experience_store",
                        test_plan=(
                            f"Verify experience '{exp.experience_id}' is marked superseded and excluded from "
                            "retrieve_experiences(include_superseded=False)."
                        ),
                        success_conditions=[
                            "Active experience retrieval no longer returns the superseded record."
                        ],
                        priority="HIGH" if exp.times_caused_problem > 0 else "MEDIUM",
                        priority_rationale="Counterproductive experience directly degrades planning quality.",
                        status=ImprovementStatus.PROPOSED,
                        action_payload={
                            "action_type": "supersede_bad_experience",
                            "experience_id": exp.experience_id,
                        },
                    )
                    self.memory_store.save_improvement_proposal(prop)
                    new_or_updated.append(prop)

        # 3. Check if fallback_model is unset while local-planner is available and model errors occurred
        model_errors = [
            f"{t.task_id}:{e.error_type}"
            for t in loaded_tasks
            for e in t.errors
            if e.error_type in ("missing_credentials", "network_error", "timeout", "http_error")
        ]
        if (
            self.orchestrator.config.fallback_model is None
            and "local-planner" in self.orchestrator.config.models
            and len(model_errors) >= 1
        ):
            title = "Configure 'local-planner' as fallback_model for resilience"
            if title not in existing_proposals:
                prop = ImprovementProposal(
                    proposal_id=new_id("prop"),
                    title=title,
                    category="model_routing",
                    observed_problem=(
                        f"Observed {len(model_errors)} model provider error(s) ({model_errors[:5]}) "
                        "while fallback_model is None."
                    ),
                    supporting_evidence={
                        "model_errors": model_errors,
                        "current_fallback_model": None,
                        "available_fallback": "local-planner",
                    },
                    likely_cause="No fallback_model profile was configured in AgentConfig.",
                    proposed_change="Set fallback_model='local-planner' in AgentConfig and persist.",
                    expected_benefits=[
                        "Enables automatic fallback routing when the primary external provider is unreachable."
                    ],
                    potential_regressions=[
                        "Fallback uses local heuristic model rather than external cloud LLM."
                    ],
                    required_dependencies=[],
                    implementation_scope="model_configuration",
                    test_plan="Verify config_reloadable and effective_model_config for fallback_model='local-planner'.",
                    success_conditions=[
                        "AgentConfig reloads cleanly with fallback_model == 'local-planner'."
                    ],
                    priority="MEDIUM",
                    priority_rationale="Improves resilience against external provider outages.",
                    status=ImprovementStatus.PROPOSED,
                    action_payload={
                        "action_type": "set_fallback_model",
                        "fallback_model": "local-planner",
                    },
                )
                self.memory_store.save_improvement_proposal(prop)
                new_or_updated.append(prop)

        return self.list_prioritized_proposals()

    def list_prioritized_proposals(
        self, status: Optional[ImprovementStatus | str] = None
    ) -> List[ImprovementProposal]:
        """Return all persisted proposals ordered by priority (HIGH > MEDIUM > LOW) and recency."""
        proposals = self.memory_store.list_improvement_proposals(status=status)
        rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        proposals.sort(
            key=lambda p: (rank.get(p.priority, 0), p.updated_at), reverse=True
        )
        return proposals

    def apply_and_evaluate(
        self, proposal_id: str
    ) -> SelfImprovementRunRecord:
        """
        Execute a proposed improvement under controlled conditions:
        1. Verify security boundaries (refuse to alter workspace confinement, credential permissions, or evaluation guards)
        2. Record baseline state checkpoint
        3. Apply bounded change (status -> IMPLEMENTED_PENDING_EVAL)
        4. Run formal Verification and Live Run Testing
        5. Commit if both pass (status -> VALIDATED_SUCCESS) or roll back to checkpoint if either fails (status -> ROLLED_BACK_FAILED)
        """
        proposal = self.memory_store.get_improvement_proposal(proposal_id)
        if proposal is None:
            raise KeyError(f"Improvement proposal '{proposal_id}' not found.")

        payload = proposal.action_payload or {}
        action_type = payload.get("action_type", "")
        target_param = payload.get("target_parameter", "")

        # Security & User Control Boundary Guard (Section 7.4)
        if (
            action_type == "modify_security_boundary"
            or target_param in _FORBIDDEN_SECURITY_TARGETS
        ):
            proposal.status = ImprovementStatus.REJECTED
            proposal.updated_at = utc_now_iso()
            self.memory_store.save_improvement_proposal(proposal)
            run_rec = SelfImprovementRunRecord(
                run_id=new_id("simpr"),
                proposal_id=proposal.proposal_id,
                baseline_behavior={"security_boundaries_enforced": True},
                change_applied=payload,
                expected_result="Reject forbidden modification to core security or evaluation boundary.",
                actual_result="Rejected modification to protected security/evaluation setting.",
                verification_passed=False,
                live_test_passed=False,
                regressions_observed=["Attempted to alter protected boundary."],
                rolled_back=False,
                final_disposition="REJECTED_SECURITY_BOUNDARY",
            )
            self.memory_store.save_self_improvement_run(run_rec)
            return run_rec

        # Mark as IMPLEMENTED_PENDING_EVAL prior to verification/live-test
        proposal.status = ImprovementStatus.IMPLEMENTED_PENDING_EVAL
        proposal.updated_at = utc_now_iso()
        self.memory_store.save_improvement_proposal(proposal)

        if action_type == "enable_default_create_parents":
            return self._apply_enable_create_parents(proposal)
        if action_type == "supersede_bad_experience":
            return self._apply_supersede_experience(proposal)
        if action_type == "set_fallback_model":
            return self._apply_set_fallback_model(proposal)
        if action_type == "code_patch":
            return self._apply_controlled_code_patch(proposal)

        raise ValueError(f"Unsupported self-improvement action_type: '{action_type}'.")

    def _apply_enable_create_parents(
        self, proposal: ImprovementProposal
    ) -> SelfImprovementRunRecord:
        existing_mem = self.memory_store.retrieve_memories(
            key="policy.fs_write.create_parents", limit=1
        )
        baseline = {
            "policy_present": len(existing_mem) > 0,
            "previous_content": existing_mem[0].content if existing_mem else None,
        }

        mem_rec = self.memory_store.store_memory(
            category=MemoryCategory.INSTRUCTION,
            key="policy.fs_write.create_parents",
            content="true",
            source=f"self_improvement:{proposal.proposal_id}",
            confidence=MemoryConfidence.VERIFIED,
        )

        # Formal Evaluation 1: Verification
        dummy_task = self.orchestrator.create_task(
            f"Verify self-improvement {proposal.proposal_id}"
        )
        v_chk = self.orchestrator.verifier.verify_requirement(
            dummy_task,
            TaskRequirement(
                requirement_id="req-policy-mem",
                description="Policy memory stored in SQLite",
                category="memory_stored",
                target="policy.fs_write.create_parents",
                expected="true",
            ),
        )
        verif_ok = v_chk.status == VerificationStatus.PASSED

        # Formal Evaluation 2: Live Run Testing (verify planner proactively sets create_parents=True)
        test_task = self.orchestrator.create_task(
            "Write a file 'nested_self_impr_check/hello.txt' with content 'ok'"
        )
        ctx = self.orchestrator.context_manager.retrieve_task_context(
            test_task.original_instruction
        )
        self.orchestrator.planner.understand_and_plan(test_task, ctx)
        write_acts = [
            a for a in test_task.planned_actions if a.capability_id == "fs.write"
        ]
        live_ok = bool(
            write_acts and write_acts[0].parameters.get("create_parents") is True
        )

        if verif_ok and live_ok:
            proposal.status = ImprovementStatus.VALIDATED_SUCCESS
            proposal.updated_at = utc_now_iso()
            self.memory_store.save_improvement_proposal(proposal)
            run_rec = SelfImprovementRunRecord(
                run_id=new_id("simpr"),
                proposal_id=proposal.proposal_id,
                baseline_behavior=baseline,
                change_applied={"memory_id": mem_rec.memory_id, "key": mem_rec.key, "content": "true"},
                expected_result="TaskPlanner sets create_parents=True on nested fs.write actions.",
                actual_result="Verified in SQLite and live-tested in TaskPlanner.",
                verification_passed=True,
                live_test_passed=True,
                rolled_back=False,
                final_disposition="VALIDATED_SUCCESS",
            )
            self.memory_store.save_self_improvement_run(run_rec)
            return run_rec

        # Rollback on failure
        self.memory_store.mark_memory_obsolete(
            mem_rec.memory_id, reason="Self-improvement validation failed"
        )
        proposal.status = ImprovementStatus.ROLLED_BACK_FAILED
        proposal.updated_at = utc_now_iso()
        self.memory_store.save_improvement_proposal(proposal)
        run_rec = SelfImprovementRunRecord(
            run_id=new_id("simpr"),
            proposal_id=proposal.proposal_id,
            baseline_behavior=baseline,
            change_applied={"memory_id": mem_rec.memory_id},
            expected_result="TaskPlanner sets create_parents=True on nested fs.write actions.",
            actual_result=f"Validation failed (verif_ok={verif_ok}, live_ok={live_ok}); rolled back.",
            verification_passed=verif_ok,
            live_test_passed=live_ok,
            regressions_observed=["Planner did not apply policy as expected."],
            rolled_back=True,
            final_disposition="ROLLED_BACK_FAILED",
        )
        self.memory_store.save_self_improvement_run(run_rec)
        return run_rec

    def _apply_supersede_experience(
        self, proposal: ImprovementProposal
    ) -> SelfImprovementRunRecord:
        exp_id = str(proposal.action_payload.get("experience_id", ""))
        baseline = {"experience_id": exp_id, "superseded": False}
        ok = self.memory_store.mark_experience_superseded(exp_id)
        active_after = self.memory_store.retrieve_experiences(
            limit=50, include_superseded=False
        )
        verif_ok = ok and all(e.experience_id != exp_id for e in active_after)
        live_ok = verif_ok

        proposal.status = (
            ImprovementStatus.VALIDATED_SUCCESS
            if verif_ok
            else ImprovementStatus.ROLLED_BACK_FAILED
        )
        proposal.updated_at = utc_now_iso()
        self.memory_store.save_improvement_proposal(proposal)
        run_rec = SelfImprovementRunRecord(
            run_id=new_id("simpr"),
            proposal_id=proposal.proposal_id,
            baseline_behavior=baseline,
            change_applied={"experience_id": exp_id, "superseded": True},
            expected_result=f"Experience '{exp_id}' excluded from active retrieval.",
            actual_result=f"mark_experience_superseded={ok}, excluded_from_active={verif_ok}",
            verification_passed=verif_ok,
            live_test_passed=live_ok,
            rolled_back=not verif_ok,
            final_disposition=(
                "VALIDATED_SUCCESS" if verif_ok else "ROLLED_BACK_FAILED"
            ),
        )
        self.memory_store.save_self_improvement_run(run_rec)
        return run_rec

    def _apply_set_fallback_model(
        self, proposal: ImprovementProposal
    ) -> SelfImprovementRunRecord:
        prev_fb = self.orchestrator.config.fallback_model
        target_fb = str(proposal.action_payload.get("fallback_model", "local-planner"))
        baseline = {"fallback_model": prev_fb}

        try:
            self.orchestrator.set_fallback_model(target_fb, save=True)
            dummy_task = self.orchestrator.create_task(
                f"Verify fallback model {proposal.proposal_id}"
            )
            v_chk = self.orchestrator.verifier.verify_requirement(
                dummy_task,
                TaskRequirement(
                    requirement_id="req-fb",
                    description="Effective fallback_model updated",
                    category="effective_model_config",
                    expected={"fallback_model": target_fb},
                ),
            )
            verif_ok = v_chk.status == VerificationStatus.PASSED
            conn_rep = self.orchestrator.test_model_connection(profile_id=target_fb)
            live_ok = conn_rep.config_valid
        except Exception as exc:
            self.orchestrator.config.fallback_model = prev_fb
            self.orchestrator.config.save()
            proposal.status = ImprovementStatus.ROLLED_BACK_FAILED
            proposal.updated_at = utc_now_iso()
            self.memory_store.save_improvement_proposal(proposal)
            run_rec = SelfImprovementRunRecord(
                run_id=new_id("simpr"),
                proposal_id=proposal.proposal_id,
                baseline_behavior=baseline,
                change_applied={"fallback_model": target_fb},
                expected_result=f"Set fallback_model='{target_fb}'",
                actual_result=f"Failed with {exc}; rolled back to {prev_fb!r}.",
                verification_passed=False,
                live_test_passed=False,
                regressions_observed=[str(exc)],
                rolled_back=True,
                final_disposition="ROLLED_BACK_FAILED",
            )
            self.memory_store.save_self_improvement_run(run_rec)
            return run_rec

        if verif_ok and live_ok:
            proposal.status = ImprovementStatus.VALIDATED_SUCCESS
            proposal.updated_at = utc_now_iso()
            self.memory_store.save_improvement_proposal(proposal)
            run_rec = SelfImprovementRunRecord(
                run_id=new_id("simpr"),
                proposal_id=proposal.proposal_id,
                baseline_behavior=baseline,
                change_applied={"fallback_model": target_fb},
                expected_result=f"Set fallback_model='{target_fb}'",
                actual_result=f"Verified effective fallback_model='{target_fb}'.",
                verification_passed=True,
                live_test_passed=True,
                rolled_back=False,
                final_disposition="VALIDATED_SUCCESS",
            )
            self.memory_store.save_self_improvement_run(run_rec)
            return run_rec

        # Rollback
        self.orchestrator.set_fallback_model(prev_fb, save=True)
        proposal.status = ImprovementStatus.ROLLED_BACK_FAILED
        proposal.updated_at = utc_now_iso()
        self.memory_store.save_improvement_proposal(proposal)
        run_rec = SelfImprovementRunRecord(
            run_id=new_id("simpr"),
            proposal_id=proposal.proposal_id,
            baseline_behavior=baseline,
            change_applied={"fallback_model": target_fb},
            expected_result=f"Set fallback_model='{target_fb}'",
            actual_result="Verification or live test failed; rolled back.",
            verification_passed=verif_ok,
            live_test_passed=live_ok,
            rolled_back=True,
            final_disposition="ROLLED_BACK_FAILED",
        )
        self.memory_store.save_self_improvement_run(run_rec)
        return run_rec

    def _apply_controlled_code_patch(
        self, proposal: ImprovementProposal
    ) -> SelfImprovementRunRecord:
        """
        Apply a bounded code change to a workspace file with automatic checkpoint rollback
        if Verification (AST/compile) or Live Run Testing (execution/unit test) fails (Section 7.2 & 7.3).
        """
        from agent911.capabilities.base import CapabilityContext
        from agent911.capabilities.filesystem import resolve_in_workspace
        from agent911.evaluation.live_testing import LiveRunTester

        rel_path = str(proposal.action_payload.get("path", ""))
        new_content = str(proposal.action_payload.get("new_content", ""))
        test_target = proposal.action_payload.get("test_target")

        target_file = resolve_in_workspace(
            self.orchestrator.config.workspace_path, rel_path
        )
        existed_before = target_file.exists()
        original_content = (
            target_file.read_text(encoding="utf-8") if existed_before else None
        )
        baseline = {
            "path": rel_path,
            "existed_before": existed_before,
            "original_size": len(original_content) if original_content is not None else 0,
        }

        # Apply experimental patch
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(new_content, encoding="utf-8")

        # 1. Formal Verification: check AST/bytecode compilation
        dummy_task = self.orchestrator.create_task(
            f"Verify code patch {proposal.proposal_id}"
        )
        v_chk = self.orchestrator.verifier.verify_requirement(
            dummy_task,
            TaskRequirement(
                requirement_id="req-compile",
                description=f"Patched file '{rel_path}' compiles cleanly",
                category="code_compiles" if rel_path.endswith(".py") else "file_exists",
                target=rel_path,
            ),
        )
        verif_ok = v_chk.status == VerificationStatus.PASSED

        # 2. Formal Live Run Testing: run patched Python file or specified test suite
        eval_ctx = CapabilityContext(
            config=self.orchestrator.config,
            memory_store=self.memory_store,
            model_interface=self.orchestrator.model_interface,
            registry=self.orchestrator.registry,
            task_id=dummy_task.task_id,
        )
        tester = LiveRunTester(eval_ctx)
        if test_target:
            test_res = self.orchestrator.registry.invoke(
                "test.run", {"target": str(test_target), "runner": "unittest"}, eval_ctx
            )
            live_ok = test_res.success
            live_detail = test_res.error or f"Tests passed: {test_res.data}"
        elif rel_path.endswith(".py"):
            l_rec = tester.test_python_file_execution(dummy_task.task_id, rel_path)
            live_ok = l_rec.status == LiveTestStatus.PASSED
            live_detail = l_rec.stderr or l_rec.stdout
        else:
            live_ok = target_file.exists()
            live_detail = "File exists"

        if verif_ok and live_ok:
            proposal.status = ImprovementStatus.VALIDATED_SUCCESS
            proposal.updated_at = utc_now_iso()
            self.memory_store.save_improvement_proposal(proposal)
            run_rec = SelfImprovementRunRecord(
                run_id=new_id("simpr"),
                proposal_id=proposal.proposal_id,
                baseline_behavior=baseline,
                change_applied={"path": rel_path, "new_size": len(new_content)},
                expected_result="Patched code compiles and passes live execution/tests.",
                actual_result=f"Verification PASSED; Live Run Test PASSED ({live_detail[:200]}).",
                verification_passed=True,
                live_test_passed=True,
                rolled_back=False,
                final_disposition="VALIDATED_SUCCESS",
            )
            self.memory_store.save_self_improvement_run(run_rec)
            return run_rec

        # Rollback experimental code change to preserve previous working state!
        if existed_before and original_content is not None:
            target_file.write_text(original_content, encoding="utf-8")
        elif target_file.exists():
            target_file.unlink()

        proposal.status = ImprovementStatus.ROLLED_BACK_FAILED
        proposal.updated_at = utc_now_iso()
        self.memory_store.save_improvement_proposal(proposal)
        run_rec = SelfImprovementRunRecord(
            run_id=new_id("simpr"),
            proposal_id=proposal.proposal_id,
            baseline_behavior=baseline,
            change_applied={"path": rel_path, "new_size": len(new_content)},
            expected_result="Patched code compiles and passes live execution/tests.",
            actual_result=(
                f"Failed validation (verif_ok={verif_ok}, live_ok={live_ok}: "
                f"{v_chk.failure_explanation or live_detail[:200]}). Restored previous file state."
            ),
            verification_passed=verif_ok,
            live_test_passed=live_ok,
            regressions_observed=[v_chk.failure_explanation or live_detail[:200]],
            rolled_back=True,
            final_disposition="ROLLED_BACK_FAILED",
        )
        self.memory_store.save_self_improvement_run(run_rec)
        return run_rec
