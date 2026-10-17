"""Main task lifecycle orchestrator and runtime model/provider manager for Agent 911."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent911.capabilities import (
    CapabilityContext,
    CapabilityRegistry,
    register_builtin_capabilities,
)
from agent911.config import AgentConfig, ModelProfileConfig, ProviderConfig
from agent911.context import ContextManager
from agent911.diagnostics import CapabilityGapAnalyzer, SelfDiagnosticEngine
from agent911.evaluation import LiveRunTester, OutcomeClassifier, Verifier
from agent911.improvement import ImprovementEngine
from agent911.memory import PersistentMemoryStore
from agent911.model_provider import ConnectivityReport, ModelInterface
from agent911.models import (
    ActionStatus,
    CapabilityCallRecord,
    ExecutionErrorRecord,
    ExperienceRecord,
    ImprovementStatus,
    MemoryCategory,
    MemoryConfidence,
    PlannedAction,
    TaskLifecycleStage,
    TaskOutcome,
    TaskState,
    new_id,
)
from agent911.planner import TaskPlanner
from agent911.recovery import RecoveryEngine


class Agent911Orchestrator:
    """
    Coordinates the end-to-end task execution lifecycle of Agent 911:
    Instruction -> Intent & Context Retrieval -> Task Representation & Planning ->
    Capability Discovery & Selection -> Actual Execution (& Targeted Recovery) ->
    Verification -> Live Run Testing -> Result Classification -> Experience & Memory Update.
    Also manages runtime provider/model configuration, active model selection, and testing.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        deterministic_responder: Optional[
            Callable[[str, Dict[str, Any]], str | Dict[str, Any]]
        ] = None,
    ) -> None:
        self.config = config or AgentConfig.load()
        self.config.validate()
        self.config.ensure_directories()

        api_key = self.config.model.resolve_api_key(self.config.credential_store)
        secrets = [api_key] if api_key else []
        self.memory_store = PersistentMemoryStore(
            db_path=self.config.db_path, secrets_to_redact=secrets
        )
        self.model_interface = ModelInterface(
            config=self.config.model,
            max_context_chars=self.config.max_context_chars,
            deterministic_responder=deterministic_responder,
            credential_store=self.config.credential_store,
            agent_config=self.config,
        )
        self.registry = CapabilityRegistry(config=self.config)
        register_builtin_capabilities(self.registry, self.memory_store)

        self.context_manager = ContextManager(
            memory_store=self.memory_store,
            max_context_chars=self.config.max_context_chars,
        )
        self.planner = TaskPlanner(
            config=self.config,
            registry=self.registry,
            model_interface=self.model_interface,
        )
        self.verifier = Verifier(
            workspace_root=self.config.workspace_path,
            registry=self.registry,
            memory_store=self.memory_store,
        )
        self.diagnostic_engine = SelfDiagnosticEngine()
        self.gap_analyzer = CapabilityGapAnalyzer()
        self.improvement_engine = ImprovementEngine(self)

    # ------------------------------------------------------------------
    # Provider & Model Configuration Operations (Section 2)
    # ------------------------------------------------------------------

    def list_providers(self) -> List[Dict[str, Any]]:
        """List all configured providers with safe masked credential status."""
        store = self.config.credential_store
        out = []
        for pid, pcfg in sorted(self.config.providers.items()):
            has_key = store.has_credential(pid) if store else False
            masked = store.get_masked_credential(pid) if store else None
            if not has_key and pcfg.api_key_env:
                import os

                env_val = os.environ.get(pcfg.api_key_env, "").strip()
                if env_val:
                    has_key = True
                    masked = (
                        f"env:{pcfg.api_key_env} ({env_val[:3]}...{env_val[-4:]})"
                        if len(env_val) > 7
                        else f"env:{pcfg.api_key_env} (***)"
                    )
            d = pcfg.to_dict()
            d["has_credentials"] = has_key
            d["credential_preview"] = masked
            out.append(d)
        return out

    def upsert_provider(
        self,
        provider_data: Dict[str, Any] | ProviderConfig,
        api_key: Optional[str] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Add or update a provider configuration and optionally store its API key securely."""
        pcfg = self.config.add_or_update_provider(provider_data, api_key=api_key)
        self.model_interface.config = self.config.model
        if save:
            self.config.save()
        return pcfg.to_dict()

    def remove_provider(self, provider_id: str, save: bool = True) -> None:
        """Remove a provider configuration if not referenced by the active/default model."""
        self.config.remove_provider(provider_id)
        if save:
            self.config.save()

    def list_models(self) -> List[Dict[str, Any]]:
        """List all configured model profiles with active/default/fallback markers."""
        out = []
        for mid, mcfg in sorted(self.config.models.items()):
            d = mcfg.to_dict()
            d["is_active"] = mid == self.config.active_model
            d["is_default"] = mid == self.config.default_model
            d["is_fallback"] = mid == self.config.fallback_model
            out.append(d)
        return out

    def upsert_model(
        self, model_data: Dict[str, Any] | ModelProfileConfig, save: bool = True
    ) -> Dict[str, Any]:
        """Add or update a model profile configuration."""
        mcfg = self.config.add_or_update_model(model_data)
        self.model_interface.config = self.config.model
        if save:
            self.config.save()
        return mcfg.to_dict()

    def remove_model(self, profile_id: str, save: bool = True) -> None:
        """Remove a model profile if it is not the currently active or default model."""
        self.config.remove_model(profile_id)
        if save:
            self.config.save()

    def select_active_model(
        self,
        profile_id: str,
        set_as_default: bool = False,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Switch the active model at runtime without restarting the application."""
        self.model_interface.switch_active_model(
            profile_id, set_as_default=set_as_default
        )
        if save:
            self.config.save()
        return {
            "active_model": self.config.active_model,
            "default_model": self.config.default_model,
            "fallback_model": self.config.fallback_model,
            "effective_model": self.config.model.to_safe_dict(),
        }

    def set_fallback_model(
        self, profile_id: Optional[str], save: bool = True
    ) -> Dict[str, Any]:
        """Configure or clear the fallback model profile."""
        if profile_id is not None and profile_id not in self.config.models:
            from agent911.config import ConfigurationError

            raise ConfigurationError(f"Fallback model '{profile_id}' is not configured.")
        self.config.fallback_model = profile_id
        if save:
            self.config.save()
        return {
            "active_model": self.config.active_model,
            "default_model": self.config.default_model,
            "fallback_model": self.config.fallback_model,
        }

    def test_model_connection(
        self, profile_id: Optional[str] = None
    ) -> ConnectivityReport:
        """Execute a genuine connection test against the active or specified model profile."""
        return self.model_interface.check_connectivity(target_profile_id=profile_id)

    # ------------------------------------------------------------------
    # Task Execution Pipeline
    # ------------------------------------------------------------------

    def create_task(self, instruction: str) -> TaskState:
        """Create and persist a new TaskState preserving the original instruction."""
        if not instruction or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Task instruction must be a non-empty string.")
        task = TaskState(
            task_id=new_id("task"),
            original_instruction=instruction,
            interpreted_objective=instruction.strip(),
            current_execution_state=TaskLifecycleStage.CREATED,
        )
        self.memory_store.save_task(task)
        return task

    def execute_task(
        self,
        instruction: str,
        explicit_plan: Optional[List[Dict[str, Any]]] = None,
        explicit_requirements: Optional[List[Dict[str, Any]]] = None,
        stop_after_step: Optional[int] = None,
    ) -> TaskState:
        """
        Execute a user instruction through the full Agent 911 task lifecycle.
        Supports optional explicit_plan/explicit_requirements for structured workflows
        and stop_after_step for testing interruption and resumption.
        """
        task = self.create_task(instruction)

        # 1. Intent Understanding & Context/Memory Retrieval
        task.transition_to(TaskLifecycleStage.RETRIEVING_CONTEXT)
        retrieved_ctx = self.context_manager.retrieve_task_context(instruction)
        for exp_dict in retrieved_ctx.get("relevant_experiences", []):
            if exp_dict.get("experience_id"):
                self.memory_store.record_experience_usefulness(
                    str(exp_dict["experience_id"]), retrieved=True
                )
        self.memory_store.record_execution_log(
            task_id=task.task_id,
            operation="retrieve_task_context",
            success=True,
            input_data={
                "instruction": instruction,
                "active_model": self.config.active_model,
            },
            output_data={
                "factual_count": len(retrieved_ctx.get("factual_memories", [])),
                "experience_count": len(retrieved_ctx.get("relevant_experiences", [])),
            },
        )

        # 2. Task Representation, Planning, and Capability Selection
        task.transition_to(TaskLifecycleStage.PLANNING)
        self.planner.understand_and_plan(
            task=task,
            retrieved_context=retrieved_ctx,
            explicit_plan=explicit_plan,
            explicit_requirements=explicit_requirements,
        )

        # Capability-Aware Model Routing evaluation (Section 10)
        routing_decision = self.config.select_model_for_operation(
            operation_category=task.intent.complexity_level if task.intent else "general",
            required_context_tokens=max(len(instruction) // 4, 256),
            require_tool_calling=False,
            require_structured_output=False,
            allow_capability_routing=bool(self.config.model_routing_rules),
        )
        task.model_routing_decisions.append(routing_decision)

        task.transition_to(TaskLifecycleStage.SELECTING_CAPABILITIES)
        self.memory_store.save_task(task)

        # 3. Run execution, evaluation, classification, and experience recording
        return self._run_task_pipeline(
            task,
            stop_after_step=stop_after_step,
            retrieved_experiences=retrieved_ctx.get("relevant_experiences", []),
        )

    def resume_task(self, task_id: str) -> TaskState:
        """
        Restore a persisted task from SQLite, verify integrity of completed artifacts,
        and resume remaining steps without blindly repeating valid completed work.
        """
        task = self.memory_store.load_task(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found in persistent storage.")

        resumption_notes = self.context_manager.prepare_task_for_resumption(
            task, self.config.workspace_path
        )
        for note in resumption_notes:
            task.decisions.append(note)
        self.memory_store.save_task(task)

        return self._run_task_pipeline(task)

    def _run_task_pipeline(
        self,
        task: TaskState,
        stop_after_step: Optional[int] = None,
        retrieved_experiences: Optional[List[Dict[str, Any]]] = None,
    ) -> TaskState:
        pipeline_t0 = time.perf_counter()
        recovery_engine = RecoveryEngine(
            max_attempts_per_step=self.config.max_recovery_attempts_per_step
        )
        task.transition_to(TaskLifecycleStage.EXECUTING)
        self.memory_store.save_task(task)

        # Resource-aware cache to avoid redundant unchanged file inspections within a task (Section 11.1)
        read_cache: Dict[str, Any] = {}
        avoided_redundant_ops = 0
        model_calls_count = 0

        executed_in_this_run = 0
        for action in task.planned_actions:
            if action.status == ActionStatus.COMPLETED:
                continue

            # Resource bound check: max_task_duration_seconds (Section 11.3)
            elapsed_sec = time.perf_counter() - pipeline_t0
            if elapsed_sec > getattr(self.config, "max_task_duration_seconds", 120.0):
                action.status = ActionStatus.FAILED
                action.error = (
                    f"Task exceeded configured max_task_duration_seconds "
                    f"({self.config.max_task_duration_seconds}s)."
                )
                task.refresh_action_lists()
                break

            # Resource bound check: max_model_calls_per_task (Section 11.3)
            if action.capability_id in ("model.query", "model.generate"):
                if model_calls_count >= getattr(self.config, "max_model_calls_per_task", 10):
                    action.status = ActionStatus.FAILED
                    action.error = (
                        f"Task reached configured max_model_calls_per_task "
                        f"({self.config.max_model_calls_per_task})."
                    )
                    task.refresh_action_lists()
                    continue
                model_calls_count += 1

            # Check if any dependency step failed and remained unrecovered
            if action.depends_on:
                dep_failed = any(
                    dep_act.status == ActionStatus.FAILED
                    for dep_act in task.planned_actions
                    if dep_act.step_id in action.depends_on
                )
                if dep_failed:
                    action.status = ActionStatus.SKIPPED
                    action.error = f"Skipped because dependency {action.depends_on} failed."
                    task.refresh_action_lists()
                    continue

            action.status = ActionStatus.IN_PROGRESS
            task.refresh_action_lists()

            cap_ctx = CapabilityContext(
                config=self.config,
                memory_store=self.memory_store,
                model_interface=self.model_interface,
                registry=self.registry,
                task_id=task.task_id,
                step_id=action.step_id,
            )

            clean_params = {
                k: v for k, v in action.parameters.items() if not k.startswith("_")
            }

            # Check if this is a redundant read of an unchanged file within the same task
            cache_key: Optional[str] = None
            if action.capability_id in ("fs.read", "fs.stat") and "path" in clean_params:
                cache_key = f"{action.capability_id}:{clean_params.get('path')}"

            if cache_key and cache_key in read_cache:
                result = read_cache[cache_key]
                avoided_redundant_ops += 1
                task.decisions.append(
                    f"Reused cached result for unchanged '{cache_key}' to avoid redundant I/O."
                )
            else:
                result = self.registry.invoke(action.capability_id, clean_params, cap_ctx)
                if result.success and cache_key:
                    read_cache[cache_key] = result
                elif action.capability_id in ("fs.write", "fs.update", "cmd.run"):
                    # Invalidate cache when state may have mutated
                    read_cache.clear()

            duration_ms = float(result.metadata.get("duration_ms", 0.0))

            call_record = CapabilityCallRecord(
                call_id=new_id("call"),
                task_id=task.task_id,
                step_id=action.step_id,
                capability_id=action.capability_id,
                parameters=clean_params,
                success=result.success,
                output_data=result.data,
                error=result.error,
                error_type=result.error_type,
                duration_ms=duration_ms,
                intended_action=action.description,
                retry_occurred=False,
                attempt_number=1,
            )
            task.capability_calls.append(call_record)
            self.memory_store.record_execution_log(
                task_id=task.task_id,
                step_id=action.step_id,
                capability_id=action.capability_id,
                operation=action.description,
                input_data=clean_params,
                output_data=result.data,
                success=result.success,
                error=result.error,
                duration_ms=duration_ms,
            )

            if result.success:
                action.status = ActionStatus.COMPLETED
                action.output_summary = self._summarize_output(result.data)
                task.relevant_outputs[action.step_id] = result.data
            else:
                err_rec = ExecutionErrorRecord(
                    error_id=new_id("err"),
                    task_id=task.task_id,
                    step_id=action.step_id,
                    capability_id=action.capability_id,
                    error_type=result.error_type or "ExecutionError",
                    message=result.error or "Capability execution failed.",
                    diagnostics=dict(result.metadata),
                    recovered=False,
                )
                task.errors.append(err_rec)

                # Attempt Targeted Recovery
                task.transition_to(TaskLifecycleStage.RECOVERING)
                recovered, retry_res = recovery_engine.attempt_step_recovery(
                    task=task,
                    action=action,
                    failed_result=result,
                    error_record=err_rec,
                    context=cap_ctx,
                )
                task.transition_to(TaskLifecycleStage.EXECUTING)

                if recovered and retry_res is not None:
                    read_cache.clear()
                    retry_duration = float(retry_res.metadata.get("duration_ms", 0.0))
                    clean_retry_params = {
                        k: v for k, v in action.parameters.items() if not k.startswith("_")
                    }
                    retry_call = CapabilityCallRecord(
                        call_id=new_id("call"),
                        task_id=task.task_id,
                        step_id=action.step_id,
                        capability_id=action.capability_id,
                        parameters=clean_retry_params,
                        success=True,
                        output_data=retry_res.data,
                        duration_ms=retry_duration,
                        intended_action=f"recovered:{action.description}",
                        retry_occurred=True,
                        attempt_number=2,
                    )
                    task.capability_calls.append(retry_call)
                    self.memory_store.record_execution_log(
                        task_id=task.task_id,
                        step_id=action.step_id,
                        capability_id=action.capability_id,
                        operation=f"recovered:{action.description}",
                        input_data=clean_retry_params,
                        output_data=retry_res.data,
                        success=True,
                        duration_ms=retry_duration,
                    )
                    action.status = ActionStatus.COMPLETED
                    action.error = None
                    action.output_summary = self._summarize_output(retry_res.data)
                    task.relevant_outputs[action.step_id] = retry_res.data
                else:
                    action.status = ActionStatus.FAILED
                    action.error = result.error

            task.refresh_action_lists()

            # Compress context if outputs exceed configured window
            if len(json.dumps(task.relevant_outputs, default=str)) > self.config.max_context_chars:
                self.context_manager.compress_context(task)

            self.memory_store.save_task(task)
            executed_in_this_run += 1

            # Support controlled interruption for testing long-running task resumption
            if stop_after_step is not None and executed_in_this_run >= stop_after_step:
                if task.pending_actions:
                    task.transition_to(TaskLifecycleStage.INTERRUPTED)
                    self.memory_store.save_task(task)
                    return task

        # Record Resource-Aware Execution Metrics (Section 11)
        total_duration_ms = round((time.perf_counter() - pipeline_t0) * 1000.0, 2)
        task.resource_metrics = {
            "total_duration_ms": total_duration_ms,
            "subtasks_planned": len(task.planned_actions),
            "subtasks_executed": executed_in_this_run,
            "capability_calls_count": len(task.capability_calls),
            "model_calls_count": model_calls_count,
            "recovery_attempts_count": len(task.recovery_attempts),
            "avoided_redundant_operations": avoided_redundant_ops,
            "context_chars_used": len(json.dumps(task.relevant_outputs, default=str)),
        }

        # Formal Evaluation Mechanism 1: Verification
        task.transition_to(TaskLifecycleStage.VERIFYING)
        self.verifier.verify_task(task)
        self.memory_store.save_task(task)

        # Formal Evaluation Mechanism 2: Live Run Testing
        task.transition_to(TaskLifecycleStage.LIVE_TESTING)
        eval_ctx = CapabilityContext(
            config=self.config,
            memory_store=self.memory_store,
            model_interface=self.model_interface,
            registry=self.registry,
            task_id=task.task_id,
        )
        live_tester = LiveRunTester(eval_ctx)
        live_tester.run_for_task(task)
        self.memory_store.save_task(task)

        # Result Classification
        task.transition_to(TaskLifecycleStage.CLASSIFYING)
        OutcomeClassifier.classify(task)
        self._populate_assistant_response(task)

        # Update usefulness feedback for retrieved experiences (Section 5.4)
        applied_proactive_exp = any(
            "Applied prior experience remedy" in d for d in task.decisions
        )
        for exp_dict in retrieved_experiences or []:
            eid = exp_dict.get("experience_id")
            if not eid:
                continue
            if applied_proactive_exp and task.final_outcome == TaskOutcome.SUCCESS:
                self.memory_store.record_experience_usefulness(
                    str(eid), helpful=True, prevented_failure=True
                )
            elif task.final_outcome == TaskOutcome.SUCCESS:
                self.memory_store.record_experience_usefulness(str(eid), helpful=True)

        # Experience and Memory Update
        exp = self._record_task_experience(task)
        if exp.experience_id not in task.associated_experience_ids:
            task.associated_experience_ids.append(exp.experience_id)

        self.memory_store.store_memory(
            category=MemoryCategory.EPISODIC,
            key=f"task:{task.task_id}",
            content=f"{task.original_instruction} -> {task.final_outcome.value}: {task.outcome_summary}",
            task_id=task.task_id,
            source="orchestrator",
            evidence={
                "outcome": task.final_outcome.value,
                "verifications": len(task.verification_results),
                "live_tests": len(task.live_run_results),
            },
            confidence=(
                MemoryConfidence.VERIFIED
                if task.final_outcome == TaskOutcome.SUCCESS
                else MemoryConfidence.OBSERVED
            ),
        )

        task.transition_to(TaskLifecycleStage.COMPLETED)
        self.memory_store.save_task(task)
        return task

    def _record_task_experience(self, task: TaskState) -> ExperienceRecord:
        caps_used = []
        for c in task.capability_calls:
            if c.capability_id not in caps_used:
                caps_used.append(c.capability_id)

        ops_performed = [
            f"{a.step_id}:{a.capability_id}({a.status.value})" for a in task.planned_actions
        ]
        unexpected_failures = [
            f"{e.error_type}: {e.message}" for e in task.errors
        ]
        identified_causes = [
            r.probable_cause for r in task.recovery_attempts if r.probable_cause
        ]
        confirmed_causes: List[str] = []
        hypothesized_causes: List[str] = []
        for dr in task.diagnostic_reports:
            for cf in dr.get("confirmed_facts", []):
                if cf not in confirmed_causes:
                    confirmed_causes.append(cf)
            for lh in dr.get("likely_hypotheses", []):
                if lh not in hypothesized_causes:
                    hypothesized_causes.append(lh)
        for e in task.errors:
            fact_str = f"Confirmed error [{e.error_type}]: {e.message}"
            if fact_str not in confirmed_causes:
                confirmed_causes.append(fact_str)

        corrections_attempted = [
            r.selected_correction
            for r in task.recovery_attempts
            if r.selected_correction and r.selected_correction != "none"
        ]
        successful_remedies = [
            r.selected_correction
            for r in task.recovery_attempts
            if r.succeeded and r.selected_correction
        ]
        limitations = [
            l.limitations for l in task.live_run_results if l.limitations
        ]
        reusable_procs = []
        if task.final_outcome == TaskOutcome.SUCCESS:
            reusable_procs.append(
                " -> ".join(f"{a.capability_id}" for a in task.planned_actions)
            )

        verif_summary = {
            "total": len(task.verification_results),
            "passed": sum(1 for v in task.verification_results if v.status.value == "PASSED"),
            "failed": sum(1 for v in task.verification_results if v.status.value == "FAILED"),
        }
        live_summary = {
            "total": len(task.live_run_results),
            "passed": sum(1 for l in task.live_run_results if l.status.value == "PASSED"),
            "failed": sum(1 for l in task.live_run_results if l.status.value == "FAILED"),
        }

        has_verified_evidence = (
            task.final_outcome == TaskOutcome.SUCCESS
            and verif_summary["passed"] > 0
            and verif_summary["failed"] == 0
        )

        exp = ExperienceRecord(
            experience_id=new_id("exp"),
            task_id=task.task_id,
            objective=task.interpreted_objective or task.original_instruction,
            plan_summary="; ".join(f"{a.step_id}:{a.description}" for a in task.planned_actions),
            capabilities_used=caps_used,
            operations_performed=ops_performed,
            actual_results={
                k: self._summarize_output(v) for k, v in task.relevant_outputs.items()
            },
            verification_summary=verif_summary,
            live_test_summary=live_summary,
            unexpected_failures=unexpected_failures,
            identified_causes=identified_causes,
            confirmed_causes=confirmed_causes,
            hypothesized_causes=hypothesized_causes,
            corrections_attempted=corrections_attempted,
            successful_remedies=successful_remedies,
            remaining_limitations=limitations,
            reusable_procedures=reusable_procs,
            outcome=task.final_outcome,
            tags=caps_used + [task.final_outcome.value.lower()],
            task_type=caps_used[0].split(".")[0] if caps_used else "general",
            evidence_supporting_outcome={
                "verification_passed": verif_summary["passed"],
                "live_tests_passed": live_summary["passed"],
                "active_model": self.config.active_model,
                "selected_strategy": task.selected_strategy,
            },
            applicable_conditions=[f"capabilities:{','.join(caps_used)}"],
            lesson_confidence="VERIFIED_EXECUTION" if has_verified_evidence else "OBSERVED",
        )
        return self.memory_store.record_experience(exp)

    @staticmethod
    def _summarize_output(data: Any) -> str:
        if data is None:
            return "none"
        text = json.dumps(data, default=str)
        return text if len(text) <= 240 else text[:240] + "..."

    def _populate_assistant_response(self, task: TaskState) -> None:
        """
        Synthesize a natural, conversational assistant_response for both pure chat
        interactions ('chat.respond') and autonomous task executions.
        """
        caps = [a.capability_id for a in task.planned_actions]
        has_chat = "chat.respond" in caps
        has_tools = any(c != "chat.respond" for c in caps)

        if has_chat and not has_tools:
            task.interaction_mode = "conversational"
        elif has_chat and has_tools:
            task.interaction_mode = "hybrid"
        else:
            task.interaction_mode = "task_execution"

        parts: List[str] = []
        for act in task.planned_actions:
            out = task.relevant_outputs.get(act.step_id)
            cid = act.capability_id
            if cid == "chat.respond" and isinstance(out, dict) and out.get("reply"):
                parts.append(str(out["reply"]))
            elif cid in ("fs.write", "fs.update") and isinstance(out, dict):
                p = out.get("path", act.parameters.get("path", "file"))
                sz = out.get("bytes_written", "")
                verb = "Created and wrote" if cid == "fs.write" else "Updated"
                parts.append(f"{verb} **`{p}`**" + (f" ({sz} bytes)." if sz != "" else "."))
            elif cid == "fs.read" and isinstance(out, dict):
                p = out.get("path", act.parameters.get("path", "file"))
                content = str(out.get("content", ""))
                preview = content if len(content) <= 600 else content[:600] + "\n... (truncated)"
                parts.append(f"Here is the content of **`{p}`**:\n```\n{preview}\n```")
            elif cid == "fs.list" and isinstance(out, dict):
                p = out.get("path", ".")
                entries = out.get("entries", [])
                names = [e.get("name", str(e)) if isinstance(e, dict) else str(e) for e in entries[:25]]
                parts.append(
                    f"Listed directory **`{p}`** ({len(entries)} entries): "
                    + (", ".join(f"`{n}`" for n in names) if names else "empty")
                )
            elif cid == "project.inspect" and isinstance(out, dict):
                pfiles = out.get("project_files", [])
                manifests = out.get("manifests", [])
                tests = out.get("test_files", [])
                parts.append(
                    f"I inspected the workspace project structure: found **{len(pfiles)} project files**, "
                    f"**{len(manifests)} manifest(s)** ({', '.join(f'`{m}`' for m in manifests[:5]) or 'none'}), "
                    f"and **{len(tests)} test file(s)** ({', '.join(f'`{t}`' for t in tests[:5]) or 'none'})."
                )
            elif cid == "test.run" and isinstance(out, dict):
                passed = out.get("passed", act.status == ActionStatus.COMPLETED)
                runner = out.get("runner", "test runner")
                parts.append(
                    f"Executed test suite via **`{runner}`**: **{'PASSED' if passed else 'FAILED'}**."
                )
            elif cid == "cmd.run" and isinstance(out, dict):
                rc = out.get("exit_code", 0)
                stdout = str(out.get("stdout", "")).strip()
                msg = f"Executed command (exit code `{rc}`)."
                if stdout:
                    msg += f"\n```\n{stdout[:400]}\n```"
                parts.append(msg)
            elif cid == "memory.store" and isinstance(out, dict):
                k = out.get("key", act.parameters.get("key", ""))
                v = out.get("content", act.parameters.get("content", ""))
                parts.append(f"Saved to persistent SQLite memory: **`{k}`** = `{v}`.")
            elif cid == "memory.retrieve" and isinstance(out, dict):
                recs = out.get("records", [])
                if recs:
                    lines = [f"- **{r.get('key')}**: {r.get('content')}" for r in recs[:8]]
                    parts.append("Retrieved from persistent memory:\n" + "\n".join(lines))
                else:
                    parts.append("No matching records were found in persistent memory.")
            elif cid == "capability.develop" and isinstance(out, dict):
                cap_id = out.get("capability_id", act.parameters.get("capability_id", ""))
                parts.append(
                    f"Developed, verified, live-tested, and registered new capability **`{cap_id}`**."
                )
            elif out is not None and act.status == ActionStatus.COMPLETED:
                parts.append(
                    f"Completed **`{cid}`**: `{self._summarize_output(out)}`"
                )

        if not parts:
            if task.errors:
                err = task.errors[-1]
                parts.append(
                    f"I encountered an issue while executing `{err.capability_id}` (`{err.error_type}`): {err.message}"
                )
            else:
                parts.append(task.outcome_summary)
        elif task.interaction_mode != "conversational":
            verif_pass = sum(1 for v in task.verification_results if v.status.value == "PASSED")
            live_pass = sum(1 for l in task.live_run_results if l.status.value == "PASSED")
            parts.append(
                f"\n*(Outcome: **{task.final_outcome.value}** — {verif_pass}/{len(task.verification_results)} Verification checks passed, "
                f"{live_pass}/{len(task.live_run_results)} Live Run Tests passed.)*"
            )

        task.assistant_response = "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Phase 3: Diagnostics, Capability Gap Analysis, & Controlled Self-Improvement
    # ------------------------------------------------------------------

    def analyze_capability_gap(
        self,
        capability_id: str,
        error_type: str = "CapabilityNotFoundError",
        error_message: str = "",
        action_description: str = "",
        action_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Distinguish missing capabilities from dependencies, credentials, config, or model limits."""
        gap = self.gap_analyzer.analyze_failure(
            capability_id=capability_id,
            error_type=error_type,
            error_message=error_message or f"Capability '{capability_id}' failure.",
            registry=self.registry,
            action_description=action_description,
            action_parameters=action_parameters,
        )
        return gap.to_dict()

    def diagnose_task(self, task_id: str) -> Dict[str, Any]:
        """Return structured diagnostic reports, capability gap analyses, and loop detection for a task."""
        task = self.memory_store.load_task(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found in persistent storage.")
        loop_detected, loop_desc = self.diagnostic_engine.detect_unproductive_loop(task)
        return {
            "task_id": task.task_id,
            "outcome": task.final_outcome.value,
            "selected_strategy": task.selected_strategy,
            "strategy_history": task.strategy_history,
            "unproductive_loop_detected": loop_detected,
            "unproductive_loop_description": loop_desc,
            "diagnostic_reports": task.diagnostic_reports,
            "capability_gap_analyses": task.capability_gap_analyses,
            "resource_metrics": task.resource_metrics,
        }

    def detect_and_propose_improvements(self) -> List[Dict[str, Any]]:
        """Analyze persistent task/experience history and generate ImprovementProposals."""
        proposals = self.improvement_engine.detect_and_propose()
        return [p.to_dict() for p in proposals]

    def list_improvement_proposals(
        self, status: Optional[ImprovementStatus | str] = None
    ) -> List[Dict[str, Any]]:
        """List persisted ImprovementProposals ordered by priority."""
        proposals = self.improvement_engine.list_prioritized_proposals(status=status)
        return [p.to_dict() for p in proposals]

    def run_controlled_self_improvement(self, proposal_id: str) -> Dict[str, Any]:
        """
        Execute controlled, bounded self-improvement for a proposal with Verification,
        Live Run Testing, security boundary enforcement, and automatic rollback on failure.
        """
        run_record = self.improvement_engine.apply_and_evaluate(proposal_id)
        return run_record.to_dict()

    def list_self_improvement_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return persisted controlled self-improvement run records."""
        return self.memory_store.list_self_improvement_runs(limit=limit)
