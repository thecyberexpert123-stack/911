"""Problem-solving strategy specifications, evidence-based selection, and strategy adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent911.models import (
    FailureRootCauseCategory,
    InstructionIntent,
    PlannedAction,
    ProblemSolvingStrategy,
    TaskState,
    utc_now_iso,
)


@dataclass
class StrategySpec:
    """Specification of a problem-solving strategy, its intended use, and its boundaries."""

    strategy: ProblemSolvingStrategy
    intended_use: str
    inappropriate_when: List[str] = field(default_factory=list)
    execution_cost: str = "low"  # low, medium, high
    expected_reliability: str = "high"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["strategy"] = self.strategy.value
        return d


STRATEGY_CATALOG: Dict[ProblemSolvingStrategy, StrategySpec] = {
    ProblemSolvingStrategy.DIRECT_EXECUTION: StrategySpec(
        strategy=ProblemSolvingStrategy.DIRECT_EXECUTION,
        intended_use="Single-operation queries, memory lookups/stores, or read-only inspections.",
        inappropriate_when=[
            "Task requires multi-file modifications or dependency ordering.",
            "Required capability is missing or unverified.",
        ],
        execution_cost="low",
        expected_reliability="high",
    ),
    ProblemSolvingStrategy.INSPECT_BEFORE_ACT: StrategySpec(
        strategy=ProblemSolvingStrategy.INSPECT_BEFORE_ACT,
        intended_use="Modifying existing files or state where prior content and structure must be verified first.",
        inappropriate_when=[
            "Simple read-only query that needs no workspace mutation.",
        ],
        execution_cost="medium",
        expected_reliability="high",
    ),
    ProblemSolvingStrategy.STEP_BY_STEP_PLANNING: StrategySpec(
        strategy=ProblemSolvingStrategy.STEP_BY_STEP_PLANNING,
        intended_use="Multi-step software construction, file creation, and test execution with explicit dependencies.",
        inappropriate_when=[
            "Trivial single-step factual or status request where multi-stage planning wastes resources.",
        ],
        execution_cost="medium",
        expected_reliability="high",
    ),
    ProblemSolvingStrategy.REUSE_PROVEN_PROCEDURE: StrategySpec(
        strategy=ProblemSolvingStrategy.REUSE_PROVEN_PROCEDURE,
        intended_use="Tasks matching a verified past ExperienceRecord whose applicable conditions hold in the current workspace.",
        inappropriate_when=[
            "Past experience was unverified, superseded, or caused problems.",
            "Current task constraints or workspace conditions differ from the historical task.",
        ],
        execution_cost="low",
        expected_reliability="high",
    ),
    ProblemSolvingStrategy.CAPABILITY_DELEGATION_OR_GROWTH: StrategySpec(
        strategy=ProblemSolvingStrategy.CAPABILITY_DELEGATION_OR_GROWTH,
        intended_use="Tasks requiring specialized registered capabilities or organic synthesis of a missing capability.",
        inappropriate_when=[
            "Existing built-in filesystem or inspection capabilities already satisfy the task.",
        ],
        execution_cost="high",
        expected_reliability="medium",
    ),
    ProblemSolvingStrategy.CONTROLLED_EXPERIMENTATION: StrategySpec(
        strategy=ProblemSolvingStrategy.CONTROLLED_EXPERIMENTATION,
        intended_use="Tasks centered on running tests, validating code behavior, or benchmarking commands in isolation.",
        inappropriate_when=[
            "Operation has irreversible external side effects.",
        ],
        execution_cost="medium",
        expected_reliability="high",
    ),
    ProblemSolvingStrategy.FAILURE_DIAGNOSIS_AND_ALTERNATIVE: StrategySpec(
        strategy=ProblemSolvingStrategy.FAILURE_DIAGNOSIS_AND_ALTERNATIVE,
        intended_use="Diagnosing an observed execution failure using evidence and switching to an alternative remedy.",
        inappropriate_when=[
            "Failure is caused by a hard security/workspace boundary or missing user credentials.",
        ],
        execution_cost="medium",
        expected_reliability="medium",
    ),
    ProblemSolvingStrategy.INTERRUPTED_TASK_RECOVERY: StrategySpec(
        strategy=ProblemSolvingStrategy.INTERRUPTED_TASK_RECOVERY,
        intended_use="Resuming a previously interrupted or partially completed task after validating existing artifacts.",
        inappropriate_when=[
            "Fresh task with no prior execution state.",
        ],
        execution_cost="low",
        expected_reliability="high",
    ),
}


class StrategySelector:
    """
    Selects and adapts problem-solving strategies using task complexity,
    explicit requirements, available capabilities, and verified past experience.
    """

    @staticmethod
    def list_strategies() -> List[Dict[str, Any]]:
        return [spec.to_dict() for spec in STRATEGY_CATALOG.values()]

    @classmethod
    def select_initial_strategy(
        cls,
        task: TaskState,
        intent: InstructionIntent,
        available_capability_ids: set[str],
        relevant_experiences: List[Dict[str, Any]],
        is_resumption: bool = False,
    ) -> Tuple[ProblemSolvingStrategy, str]:
        """
        Determine the most appropriate initial strategy based on task evidence.
        Records the selection on task.selected_strategy and task.strategy_history.
        """
        if is_resumption:
            strat = ProblemSolvingStrategy.INTERRUPTED_TASK_RECOVERY
            rationale = "Task is being resumed from persisted state; verifying completed artifacts before continuing."
            cls._record_strategy(task, strat, rationale, trigger="resumption")
            return strat, rationale

        req_caps = list(intent.required_capabilities)

        # 1. Check if capability growth or specialized custom capability is needed
        if "capability.develop" in req_caps or any(
            c not in available_capability_ids for c in req_caps
        ):
            strat = ProblemSolvingStrategy.CAPABILITY_DELEGATION_OR_GROWTH
            rationale = (
                f"Task requires capability development or delegation ({req_caps}); "
                "using CAPABILITY_DELEGATION_OR_GROWTH."
            )
            cls._record_strategy(task, strat, rationale, trigger="capability_requirement")
            return strat, rationale

        # 2. Check if a verified, high-usefulness past experience applies under matching conditions
        matching_exp = cls._find_applicable_verified_experience(
            req_caps, relevant_experiences
        )
        if matching_exp is not None:
            strat = ProblemSolvingStrategy.REUSE_PROVEN_PROCEDURE
            exp_id = matching_exp.get("experience_id", "unknown")
            rationale = (
                f"Matched verified past experience '{exp_id}' with compatible capability conditions "
                f"({req_caps}) and zero caused problems."
            )
            cls._record_strategy(task, strat, rationale, trigger=f"experience:{exp_id}")
            return strat, rationale

        # 3. Check if modifying an existing file -> INSPECT_BEFORE_ACT
        if "fs.update" in req_caps or (
            intent.complexity_level == "moderate_inspect_act"
            and any(c in req_caps for c in ("fs.update", "fs.read"))
        ):
            strat = ProblemSolvingStrategy.INSPECT_BEFORE_ACT
            rationale = "Task modifies or inspects workspace artifacts; using INSPECT_BEFORE_ACT."
            cls._record_strategy(task, strat, rationale, trigger="artifact_modification")
            return strat, rationale

        # 4. Check if controlled test/command experimentation
        if req_caps == ["test.run"] or (
            "test.run" in req_caps and "fs.write" not in req_caps
        ):
            strat = ProblemSolvingStrategy.CONTROLLED_EXPERIMENTATION
            rationale = "Task exercises test suite or runtime validation; using CONTROLLED_EXPERIMENTATION."
            cls._record_strategy(task, strat, rationale, trigger="test_execution")
            return strat, rationale

        # 5. Simple single-capability operation -> DIRECT_EXECUTION
        if len(req_caps) <= 1 and intent.complexity_level == "simple_direct":
            strat = ProblemSolvingStrategy.DIRECT_EXECUTION
            rationale = (
                f"Single-operation task ({req_caps[0] if req_caps else 'inspect'}) "
                "requires no multi-stage planning overhead."
            )
            cls._record_strategy(task, strat, rationale, trigger="simple_complexity")
            return strat, rationale

        # 6. Multi-step task -> STEP_BY_STEP_PLANNING
        strat = ProblemSolvingStrategy.STEP_BY_STEP_PLANNING
        rationale = (
            f"Multi-step task involving capabilities {req_caps}; "
            "using dependency-aware STEP_BY_STEP_PLANNING."
        )
        cls._record_strategy(task, strat, rationale, trigger="multi_step_complexity")
        return strat, rationale

    @classmethod
    def adapt_strategy_on_failure(
        cls,
        task: TaskState,
        root_cause: FailureRootCauseCategory,
        failed_action: PlannedAction,
        reason: str,
    ) -> Tuple[ProblemSolvingStrategy, str]:
        """
        Switch strategy when the current approach fails, based on diagnosed root cause.
        """
        if root_cause == FailureRootCauseCategory.UNAVAILABLE_CAPABILITY:
            new_strat = ProblemSolvingStrategy.CAPABILITY_DELEGATION_OR_GROWTH
            rationale = (
                f"Switched strategy to {new_strat.value} because capability "
                f"'{failed_action.capability_id}' was missing: {reason}"
            )
        elif root_cause in (
            FailureRootCauseCategory.INCORRECT_ASSUMPTION,
            FailureRootCauseCategory.IMPLEMENTATION_DEFECT,
        ):
            new_strat = ProblemSolvingStrategy.FAILURE_DIAGNOSIS_AND_ALTERNATIVE
            rationale = (
                f"Switched strategy to {new_strat.value} after diagnosing "
                f"{root_cause.value} on step '{failed_action.step_id}': {reason}"
            )
        else:
            new_strat = ProblemSolvingStrategy.FAILURE_DIAGNOSIS_AND_ALTERNATIVE
            rationale = (
                f"Adapted strategy to {new_strat.value} due to {root_cause.value}: {reason}"
            )

        cls._record_strategy(
            task,
            new_strat,
            rationale,
            trigger=f"failure_adaptation:{failed_action.step_id}:{root_cause.value}",
        )
        return new_strat, rationale

    @staticmethod
    def _find_applicable_verified_experience(
        required_caps: List[str], experiences: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not required_caps or not experiences:
            return None
        req_set = set(required_caps)
        for exp in experiences:
            if exp.get("superseded"):
                continue
            if exp.get("outcome") != "SUCCESS":
                continue
            if exp.get("lesson_confidence") != "VERIFIED_EXECUTION":
                continue
            if int(exp.get("times_caused_problem", 0)) > 0:
                continue
            exp_caps = set(exp.get("capabilities_used") or [])
            # Require that the past experience's capabilities cover the current task's required capabilities
            if req_set and req_set.issubset(exp_caps):
                return exp
            # Or if the past experience has a verified remedy for the exact capability being used
            if exp.get("successful_remedies") and (req_set & exp_caps):
                return exp
        return None

    @staticmethod
    def _record_strategy(
        task: TaskState,
        strategy: ProblemSolvingStrategy,
        rationale: str,
        trigger: str,
    ) -> None:
        task.selected_strategy = strategy.value
        task.strategy_rationale = rationale
        task.strategy_history.append(
            {
                "strategy": strategy.value,
                "rationale": rationale,
                "trigger": trigger,
                "timestamp": utc_now_iso(),
            }
        )
        task.decisions.append(f"Strategy [{strategy.value}]: {rationale}")
