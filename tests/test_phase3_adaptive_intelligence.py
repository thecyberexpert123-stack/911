"""Phase 3 tests for Autonomous Problem-Solving, Experience-Driven Improvement, and Adaptive Intelligence."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agent911.config import AgentConfig, ModelProfileConfig, ProviderConfig
from agent911.models import (
    ActionStatus,
    FailureRootCauseCategory,
    ImprovementProposal,
    ImprovementStatus,
    LiveTestStatus,
    PlannedAction,
    ProblemSolvingStrategy,
    TaskOutcome,
    VerificationStatus,
    new_id,
)
from agent911.orchestrator import Agent911Orchestrator


class TestPhase3AdaptiveIntelligence(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp_dir.name)
        self.config = AgentConfig(workspace_root=str(self.workspace))
        self.orchestrator = Agent911Orchestrator(config=self.config)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_01_task_understanding_decomposition_and_proportional_planning(self) -> None:
        """
        Test Section 3:
        - Explicit vs. inferred requirements, available resources, complexity level, and clarification detection
        - Proportional planning (simple direct vs. multi-step with dependencies)
        - Dynamic plan revision preserving original objective and constraints
        """
        # 1. Simple direct task -> proportional planning without unnecessary subtask layers
        t_simple = self.orchestrator.execute_task("Inspect project structure")
        self.assertIsNotNone(t_simple.intent)
        self.assertEqual(t_simple.intent.complexity_level, "simple_direct")
        self.assertEqual(
            t_simple.selected_strategy, ProblemSolvingStrategy.DIRECT_EXECUTION.value
        )
        self.assertEqual(len(t_simple.planned_actions), 1)
        self.assertIn("project.inspect", t_simple.intent.available_resources)
        self.assertFalse(t_simple.intent.clarification_needed)

        # 2. Multi-step task with Python artifact -> inferred syntax requirement & automatic dependency chaining
        t_multi = self.orchestrator.execute_task(
            "Create directory 'pkg_demo', write Python file 'pkg_demo/calc.py' with function 'add_nums', and run tests",
            explicit_plan=[
                {
                    "step_id": "step-1",
                    "description": "Create pkg_demo directory",
                    "capability_id": "fs.mkdir",
                    "parameters": {"path": "pkg_demo", "parents": True},
                },
                {
                    "step_id": "step-2",
                    "description": "Write pkg_demo/calc.py",
                    "capability_id": "fs.write",
                    "parameters": {
                        "path": "pkg_demo/calc.py",
                        "content": "def add_nums(a, b):\n    return a + b\n",
                    },
                },
            ],
        )
        self.assertIsNotNone(t_multi.intent)
        self.assertTrue(
            any("pkg_demo/calc.py" in ir for ir in t_multi.intent.inferred_requirements)
        )
        # step-2 should automatically depend on step-1 (which created 'pkg_demo')
        step2 = [a for a in t_multi.planned_actions if a.step_id == "step-2"][0]
        self.assertIn("step-1", step2.depends_on)
        self.assertIn("pkg_demo/calc.py", step2.expected_outputs)
        self.assertEqual(t_multi.final_outcome, TaskOutcome.SUCCESS)

        # 3. Irreversible destructive ambiguity -> flags clarification_needed
        intent_destruct = self.orchestrator.planner._derive_heuristic_intent(
            "Delete everything and wipe production database", [], []
        )
        self.assertTrue(intent_destruct.clarification_needed)
        self.assertIsNotNone(intent_destruct.clarification_question)

        # 4. Dynamic plan revision preserves original instruction, objective, and constraints
        t_rev = self.orchestrator.create_task("Build feature module")
        self.orchestrator.planner.understand_and_plan(t_rev, {})
        orig_goal = t_rev.interpreted_objective
        orig_constraints = list(t_rev.intent.constraints)
        t_rev.planned_actions[0].status = ActionStatus.COMPLETED
        self.orchestrator.planner.revise_plan_with_new_evidence(
            t_rev,
            [
                PlannedAction(
                    step_id="step-revised-2",
                    description="Write revised module",
                    capability_id="fs.write",
                    parameters={"path": "revised.py", "content": "x = 1\n"},
                )
            ],
            reason="Discovered simpler implementation path during inspection.",
        )
        self.assertEqual(t_rev.interpreted_objective, orig_goal)
        self.assertEqual(t_rev.intent.constraints, orig_constraints)
        self.assertEqual(len(t_rev.planned_actions), 2)
        self.assertEqual(t_rev.planned_actions[1].step_id, "step-revised-2")

    def test_02_evidence_based_strategy_selection_and_failure_adaptation(self) -> None:
        """
        Test Section 4:
        - Strategy catalog specifications (intended_use, inappropriate_when)
        - Strategy selection across task types (DIRECT_EXECUTION, CAPABILITY_DELEGATION_OR_GROWTH, REUSE_PROVEN_PROCEDURE)
        - Strategy adaptation when an initial step fails (switching to FAILURE_DIAGNOSIS_AND_ALTERNATIVE)
        """
        from agent911.strategy import STRATEGY_CATALOG, StrategySelector

        self.assertGreaterEqual(len(STRATEGY_CATALOG), 8)
        for strat, spec in STRATEGY_CATALOG.items():
            self.assertTrue(spec.intended_use)
            self.assertTrue(len(spec.inappropriate_when) > 0)

        # Task that triggers a step failure (ParentDirectoryNotFoundError) -> adapts strategy
        t_adapt = self.orchestrator.execute_task(
            "Write file 'missing_dir_strat/note.txt' with content 'adapted'",
            explicit_plan=[
                {
                    "step_id": "step-1",
                    "description": "Write note.txt without create_parents",
                    "capability_id": "fs.write",
                    "parameters": {
                        "path": "missing_dir_strat/note.txt",
                        "content": "adapted\n",
                        "create_parents": False,
                    },
                }
            ],
        )
        self.assertEqual(t_adapt.final_outcome, TaskOutcome.SUCCESS)
        self.assertEqual(
            t_adapt.selected_strategy,
            ProblemSolvingStrategy.FAILURE_DIAGNOSIS_AND_ALTERNATIVE.value,
        )
        self.assertGreaterEqual(len(t_adapt.strategy_history), 2)
        triggers = [h.get("trigger", "") for h in t_adapt.strategy_history]
        self.assertTrue(any("failure_adaptation" in tr for tr in triggers))

        # Subsequent task with matching verified experience selects REUSE_PROVEN_PROCEDURE
        t_reuse = self.orchestrator.execute_task(
            "Write file 'missing_dir_strat_2/note.txt' with content 'reused'"
        )
        self.assertEqual(t_reuse.final_outcome, TaskOutcome.SUCCESS)
        self.assertEqual(
            t_reuse.selected_strategy,
            ProblemSolvingStrategy.REUSE_PROVEN_PROCEDURE.value,
        )

    def test_03_self_diagnosis_confirmed_vs_hypothesized_causes_and_loop_detection(self) -> None:
        """
        Test Section 8:
        - SelfDiagnosticEngine produces structured DiagnosticReport separating confirmed_facts,
          likely_hypotheses, eliminated_causes, and remaining_uncertainty
        - Performs safe diagnostic probe (checking parent directory existence on ParentDirectoryNotFoundError)
        - Detects unproductive repetition loops and prevents infinite retry loops
        """
        t_fail_recover = self.orchestrator.execute_task(
            "Write 'diag_dir/item.txt'",
            explicit_plan=[
                {
                    "step_id": "step-1",
                    "description": "Write diag_dir/item.txt",
                    "capability_id": "fs.write",
                    "parameters": {
                        "path": "diag_dir/item.txt",
                        "content": "hello\n",
                        "create_parents": False,
                    },
                }
            ],
        )
        self.assertGreaterEqual(len(t_fail_recover.diagnostic_reports), 1)
        diag = t_fail_recover.diagnostic_reports[0]
        self.assertEqual(
            diag["root_cause_category"],
            FailureRootCauseCategory.INCORRECT_ASSUMPTION.value,
        )
        self.assertTrue(len(diag["confirmed_facts"]) >= 2)
        self.assertTrue(any("Confirmed:" in f for f in diag["confirmed_facts"]))
        self.assertTrue(len(diag["eliminated_causes"]) >= 1)
        self.assertIsNotNone(diag["diagnostic_action_taken"])

        # Test diagnose_task orchestrator API
        diag_summary = self.orchestrator.diagnose_task(t_fail_recover.task_id)
        self.assertEqual(diag_summary["task_id"], t_fail_recover.task_id)
        self.assertFalse(diag_summary["unproductive_loop_detected"])

    def test_04_experience_learning_confirmed_vs_hypothesis_and_usefulness_ranking(self) -> None:
        """
        Test Section 5:
        - ExperienceRecord separates confirmed_causes from hypothesized_causes
        - Practical usefulness feedback (times_retrieved, times_helpful, times_prevented_failure,
          times_irrelevant, times_caused_problem) is persisted and influences retrieval ranking
        """
        t1 = self.orchestrator.execute_task(
            "Create nested file 'learn_box/data.txt' without parent",
            explicit_plan=[
                {
                    "step_id": "step-1",
                    "description": "Write learn_box/data.txt",
                    "capability_id": "fs.write",
                    "parameters": {
                        "path": "learn_box/data.txt",
                        "content": "payload\n",
                        "create_parents": False,
                    },
                }
            ],
        )
        exps = self.orchestrator.memory_store.retrieve_experiences(task_id=t1.task_id)
        self.assertEqual(len(exps), 1)
        exp1 = exps[0]
        self.assertTrue(len(exp1.confirmed_causes) >= 1)

        # Record usefulness feedback on exp1
        self.orchestrator.memory_store.record_experience_usefulness(
            exp1.experience_id,
            retrieved=True,
            helpful=True,
            prevented_failure=True,
        )
        reloaded = self.orchestrator.memory_store.retrieve_experiences(task_id=t1.task_id)[0]
        self.assertGreaterEqual(reloaded.times_retrieved, 1)
        self.assertGreaterEqual(reloaded.times_helpful, 1)
        self.assertGreaterEqual(reloaded.times_prevented_failure, 1)

        # Create a counterproductive experience and record times_caused_problem=3
        t_bad = self.orchestrator.execute_task("Write 'learn_box/other.txt' with content 'x'")
        bad_exp = self.orchestrator.memory_store.retrieve_experiences(task_id=t_bad.task_id)[0]
        for _ in range(3):
            self.orchestrator.memory_store.record_experience_usefulness(
                bad_exp.experience_id, retrieved=True, caused_problem=True
            )

        # Ranked retrieval for "learn_box" should rank helpful exp1 first and exclude/demote bad_exp
        ranked = self.orchestrator.memory_store.retrieve_experiences(query="learn_box")
        ranked_ids = [r.experience_id for r in ranked]
        self.assertIn(exp1.experience_id, ranked_ids)
        self.assertNotIn(bad_exp.experience_id, ranked_ids)

    def test_05_capability_gap_analysis_and_concrete_capability_proposal(self) -> None:
        """
        Test Section 9:
        - Distinguishes genuinely missing capability from missing credential, permission/boundary,
          configuration setting, model feature, or external service
        - Generates a complete CapabilityProposal when a capability is genuinely missing
        """
        # 1. Genuinely missing capability -> generates CapabilityProposal
        gap_missing = self.orchestrator.analyze_capability_gap(
            capability_id="archive.zip_inspector",
            error_type="CapabilityNotFoundError",
            error_message="Capability 'archive.zip_inspector' is not registered.",
            action_description="Inspect contents of a zip archive",
        )
        self.assertEqual(gap_missing["missing_element_type"], "capability")
        prop = gap_missing["capability_proposal"]
        self.assertIsNotNone(prop)
        self.assertEqual(prop["capability_id"], "archive.zip_inspector")
        self.assertIn("verification_plan", prop)
        self.assertIn("live_test_plan", prop)

        # 2. Missing credential on model.query -> classified as 'credential', NOT missing capability
        gap_cred = self.orchestrator.analyze_capability_gap(
            capability_id="model.query",
            error_type="missing_credentials",
            error_message="No API key configured for provider.",
        )
        self.assertEqual(gap_cred["missing_element_type"], "credential")
        self.assertIsNone(gap_cred["capability_proposal"])

        # 3. Workspace boundary violation -> classified as 'permission'
        gap_perm = self.orchestrator.analyze_capability_gap(
            capability_id="fs.write",
            error_type="WorkspaceBoundaryError",
            error_message="Path escapes workspace root.",
        )
        self.assertEqual(gap_perm["missing_element_type"], "permission")

        # 4. Unsupported model feature -> classified as 'model_feature'
        gap_feat = self.orchestrator.analyze_capability_gap(
            capability_id="model.generate",
            error_type="unsupported_feature",
            error_message="Model does not support tool_calling.",
        )
        self.assertEqual(gap_feat["missing_element_type"], "model_feature")

    def test_06_improvement_proposal_detection_and_prioritization(self) -> None:
        """
        Test Section 6:
        - Detects operational weaknesses from actual SQLite execution history
        - Generates structured ImprovementProposals with supporting evidence, likely cause,
          proposed change, test plan, success conditions, and priority ranking
        """
        # Trigger two ParentDirectoryNotFoundError occurrences across tasks so it is recurring (HIGH priority)
        for idx in (1, 2):
            self.orchestrator.execute_task(
                f"Create file 'recurring_nested_{idx}/file.txt'",
                explicit_plan=[
                    {
                        "step_id": "step-1",
                        "description": "Write nested file without create_parents",
                        "capability_id": "fs.write",
                        "parameters": {
                            "path": f"recurring_nested_{idx}/file.txt",
                            "content": "hello\n",
                            "create_parents": False,
                        },
                    }
                ],
            )

        proposals = self.orchestrator.detect_and_propose_improvements()
        self.assertGreaterEqual(len(proposals), 1)
        top = proposals[0]
        self.assertEqual(top["priority"], "HIGH")
        self.assertEqual(top["status"], ImprovementStatus.PROPOSED.value)
        self.assertEqual(top["supporting_evidence"]["occurrence_count"], 2)
        self.assertTrue(top["supporting_evidence"]["is_recurring"])
        self.assertTrue(len(top["expected_benefits"]) >= 1)
        self.assertTrue(len(top["success_conditions"]) >= 1)

    def test_07_controlled_self_improvement_validation_and_automatic_rollback(self) -> None:
        """
        Test Section 7:
        - Controlled self-improvement applies a proposal, runs Verification + Live Run Testing,
          and transitions status to VALIDATED_SUCCESS
        - Subsequent task benefits from the validated improvement (proactive create_parents=True)
        - Broken experimental code patch fails Verification/Live Run Testing and automatically
          rolls back to the exact baseline file content (ROLLED_BACK_FAILED)
        - Forbidden security boundary modification is rejected (REJECTED_SECURITY_BOUNDARY)
        """
        # 1. Trigger weakness, detect proposal, and apply controlled self-improvement
        self.orchestrator.execute_task(
            "Write 'impr_dir/a.txt'",
            explicit_plan=[
                {
                    "step_id": "step-1",
                    "description": "Write impr_dir/a.txt",
                    "capability_id": "fs.write",
                    "parameters": {
                        "path": "impr_dir/a.txt",
                        "content": "a\n",
                        "create_parents": False,
                    },
                }
            ],
        )
        proposals = self.orchestrator.detect_and_propose_improvements()
        policy_prop = [
            p for p in proposals if p["action_payload"].get("action_type") == "enable_default_create_parents"
        ][0]

        run_res = self.orchestrator.run_controlled_self_improvement(policy_prop["proposal_id"])
        self.assertEqual(run_res["final_disposition"], "VALIDATED_SUCCESS")
        self.assertTrue(run_res["verification_passed"])
        self.assertTrue(run_res["live_test_passed"])
        self.assertFalse(run_res["rolled_back"])

        updated_prop = self.orchestrator.memory_store.get_improvement_proposal(
            policy_prop["proposal_id"]
        )
        self.assertEqual(updated_prop.status, ImprovementStatus.VALIDATED_SUCCESS)

        # 2. Test automatic rollback on a broken experimental code patch
        target_py = self.workspace / "service_module.py"
        original_code = "def compute():\n    return 42\n\nif __name__ == '__main__':\n    assert compute() == 42\n"
        target_py.write_text(original_code, encoding="utf-8")

        bad_patch_prop = ImprovementProposal(
            proposal_id=new_id("prop"),
            title="Experimental broken optimization on service_module.py",
            category="code_optimization",
            observed_problem="Test rollback on runtime failure",
            supporting_evidence={"target": "service_module.py"},
            likely_cause="Experimental change",
            proposed_change="Replace compute() with a broken implementation that raises RuntimeError",
            expected_benefits=["Faster execution"],
            potential_regressions=["May break runtime assertion"],
            required_dependencies=[],
            implementation_scope="service_module.py",
            test_plan="Compile and execute service_module.py via LiveRunTester",
            success_conditions=["Exit code 0"],
            priority="LOW",
            priority_rationale="Experimental",
            status=ImprovementStatus.PROPOSED,
            action_payload={
                "action_type": "code_patch",
                "path": "service_module.py",
                "new_content": "def compute():\n    raise RuntimeError('Broken patch!')\n\nif __name__ == '__main__':\n    compute()\n",
            },
        )
        self.orchestrator.memory_store.save_improvement_proposal(bad_patch_prop)

        rollback_run = self.orchestrator.run_controlled_self_improvement(
            bad_patch_prop.proposal_id
        )
        self.assertEqual(rollback_run["final_disposition"], "ROLLED_BACK_FAILED")
        self.assertTrue(rollback_run["rolled_back"])
        self.assertFalse(rollback_run["live_test_passed"])
        # Verify original file was restored intact!
        self.assertEqual(target_py.read_text(encoding="utf-8"), original_code)

        # 3. Test security boundary guard rejects forbidden modification
        sec_prop = ImprovementProposal(
            proposal_id=new_id("prop"),
            title="Attempt to disable verification guard",
            category="security",
            observed_problem="None",
            supporting_evidence={},
            likely_cause="None",
            proposed_change="Disable verification",
            expected_benefits=[],
            potential_regressions=["Violates formal evaluation architecture"],
            required_dependencies=[],
            implementation_scope="core",
            test_plan="None",
            success_conditions=[],
            priority="LOW",
            priority_rationale="None",
            status=ImprovementStatus.PROPOSED,
            action_payload={
                "action_type": "modify_security_boundary",
                "target_parameter": "disable_verification",
            },
        )
        self.orchestrator.memory_store.save_improvement_proposal(sec_prop)
        sec_run = self.orchestrator.run_controlled_self_improvement(sec_prop.proposal_id)
        self.assertEqual(sec_run["final_disposition"], "REJECTED_SECURITY_BOUNDARY")

    def test_08_capability_aware_model_routing_and_resource_efficiency(self) -> None:
        """
        Test Sections 10 & 11:
        - Capability-aware model selection checks context_window_tokens, supports_tool_calling,
          supports_structured_output, and model_routing_rules while respecting explicit user choice
        - Resource-aware execution caches unchanged file reads (avoided_redundant_operations > 0),
          invalidates cache on write, and records resource_metrics
        """
        self.orchestrator.upsert_provider(
            ProviderConfig(
                provider_id="local_p",
                name="Local Provider",
                adapter_type="local_only",
                endpoint="local://",
            ),
            save=False,
        )
        self.orchestrator.upsert_model(
            ModelProfileConfig(
                profile_id="small-ctx-model",
                name="Small Context Model",
                provider_id="local_p",
                model_id="small-v1",
                context_window_tokens=1024,
                supports_tool_calling=False,
                supports_structured_output=False,
            ),
            save=False,
        )
        self.orchestrator.upsert_model(
            ModelProfileConfig(
                profile_id="large-tool-model",
                name="Large Tool Model",
                provider_id="local_p",
                model_id="large-v1",
                context_window_tokens=128000,
                supports_tool_calling=True,
                supports_structured_output=True,
            ),
            save=False,
        )
        self.orchestrator.select_active_model("small-ctx-model", save=False)

        # 1. When allow_capability_routing=False, respects explicit user model choice and reports limitations
        res_strict = self.config.select_model_for_operation(
            operation_category="complex_multi_stage",
            required_context_tokens=8192,
            require_tool_calling=True,
            allow_capability_routing=False,
        )
        self.assertFalse(res_strict["suitable"])
        self.assertEqual(res_strict["selected_profile_id"], "small-ctx-model")
        self.assertGreaterEqual(len(res_strict["limitations"]), 2)

        # 2. When allow_capability_routing=True, routes to compatible 'large-tool-model'
        res_routed = self.config.select_model_for_operation(
            operation_category="complex_multi_stage",
            required_context_tokens=8192,
            require_tool_calling=True,
            allow_capability_routing=True,
        )
        self.assertTrue(res_routed["suitable"])
        self.assertEqual(res_routed["selected_profile_id"], "large-tool-model")

        # 3. Resource-aware execution: redundant fs.read on unchanged file is avoided via cache
        (self.workspace / "readme_cache.txt").write_text("cached content\n", encoding="utf-8")
        t_res = self.orchestrator.execute_task(
            "Read readme_cache.txt twice",
            explicit_plan=[
                {
                    "step_id": "step-1",
                    "description": "First read",
                    "capability_id": "fs.read",
                    "parameters": {"path": "readme_cache.txt"},
                },
                {
                    "step_id": "step-2",
                    "description": "Second identical read on unchanged file",
                    "capability_id": "fs.read",
                    "parameters": {"path": "readme_cache.txt"},
                },
            ],
        )
        self.assertEqual(t_res.final_outcome, TaskOutcome.SUCCESS)
        self.assertEqual(t_res.resource_metrics["avoided_redundant_operations"], 1)
        self.assertIn("total_duration_ms", t_res.resource_metrics)

        # 4. Verify CLI Phase 3 commands (`proposals`, `self-improve`, `diagnose`) work end-to-end
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "agent911",
                "--workspace",
                str(self.workspace),
                "--json",
                "diagnose",
                t_res.task_id,
            ],
            capture_output=True,
            text=True,
             timeout=15,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        diag_json = json.loads(proc.stdout)
        self.assertEqual(diag_json["task_id"], t_res.task_id)

    def test_09_conversational_chatbot_and_instruction_execution_modes(self) -> None:
        """
        Verify Agent 911 acts as both a natural conversational chatbot ('chat.respond')
        and an autonomous instruction-executing agent with persistent memory across turns.
        """
        # 1. Greeting & identity conversation
        t_hello = self.orchestrator.execute_task("Hello! How are you?")
        self.assertEqual(t_hello.final_outcome, TaskOutcome.SUCCESS)
        self.assertEqual(t_hello.interaction_mode, "conversational")
        self.assertIn("Agent 911", t_hello.assistant_response)

        # 2. Natural user introduction -> automatically saved to SQLite factual memory
        t_intro = self.orchestrator.execute_task(
            "My name is Aryan and my role is Systems Engineer"
        )
        self.assertEqual(t_intro.final_outcome, TaskOutcome.SUCCESS)
        self.assertEqual(t_intro.interaction_mode, "conversational")
        self.assertIn("Aryan", t_intro.assistant_response)

        # 3. Natural recall question in a subsequent turn
        t_recall = self.orchestrator.execute_task("What is my name?")
        self.assertEqual(t_recall.final_outcome, TaskOutcome.SUCCESS)
        self.assertEqual(t_recall.interaction_mode, "conversational")
        self.assertIn("Aryan", t_recall.assistant_response)

        # 4. Conversational math evaluation
        t_math = self.orchestrator.execute_task("What is 125 * 8?")
        self.assertEqual(t_math.final_outcome, TaskOutcome.SUCCESS)
        self.assertIn("1000", t_math.assistant_response)

        # 5. Concrete workspace instruction execution in the same session -> task_execution mode + natural assistant_response
        t_exec = self.orchestrator.execute_task(
            "Write a Python file 'greeter.py' with a function 'greet_user'"
        )
        self.assertEqual(t_exec.final_outcome, TaskOutcome.SUCCESS)
        self.assertEqual(t_exec.interaction_mode, "task_execution")
        self.assertIn("greeter.py", t_exec.assistant_response)
        self.assertTrue((self.workspace / "greeter.py").exists())

    def test_10_distill_concrete_instructions_from_messy_user_writings_and_build(self) -> None:
        """
        Verify Agent 911 distills messy, informal, typo-laden user writings into concrete
        build specifications (identifying folders, missing file extensions, multiple functions,
        configs, and unit tests) and builds + verifies + live-tests the artifacts.
        """
        # 1. Messy stream-of-consciousness prompt with filler, typo ('pyhton'), extensionless name ('price_calc'),
        #    folder ('billing/'), natural-language behaviors ('calculate tax and apply discount'), and config json
        messy_prompt_1 = (
            "uhh hey so like i'm trying to make a pyhton script maybe call it price_calc "
            "or something in billing/ folder... basically it needs to calculate tax and "
            "apply discount, oh and also make a config json in billing/rates.json!!"
        )
        t_messy1 = self.orchestrator.execute_task(messy_prompt_1)
        self.assertEqual(t_messy1.final_outcome, TaskOutcome.SUCCESS)
        self.assertIsNotNone(t_messy1.distilled_spec)
        self.assertTrue(t_messy1.distilled_spec.is_messy_input)
        self.assertTrue(t_messy1.distilled_spec.is_build_intent)
        self.assertIn("billing/price_calc.py", t_messy1.distilled_spec.what_user_wants_to_build)
        self.assertIn("calculate_tax", t_messy1.distilled_spec.extracted_functions)
        self.assertIn("apply_discount", t_messy1.distilled_spec.extracted_functions)
        self.assertGreaterEqual(len(t_messy1.distilled_spec.ambiguities_resolved), 2)

        # Both artifacts must exist and pass Verification & Live Run Testing
        self.assertTrue((self.workspace / "billing" / "price_calc.py").exists())
        self.assertTrue((self.workspace / "billing" / "rates.json").exists())
        self.assertTrue(all(v.status == VerificationStatus.PASSED for v in t_messy1.verification_results))
        self.assertTrue(any(l.status == LiveTestStatus.PASSED for l in t_messy1.live_run_results))

        # 2. Messy prompt with NO filename at all + companion test file synthesis + running unittest
        messy_prompt_2 = (
            "yo can u pls whip up a password generator tool in python that can "
            "generate password and check strength, and also make a test file for it "
            "and run unittest? idk just make it work"
        )
        t_messy2 = self.orchestrator.execute_task(messy_prompt_2)
        self.assertEqual(t_messy2.final_outcome, TaskOutcome.SUCCESS)
        self.assertIsNotNone(t_messy2.distilled_spec)
        self.assertTrue(t_messy2.distilled_spec.is_messy_input)
        self.assertTrue(t_messy2.distilled_spec.is_build_intent)
        self.assertTrue((self.workspace / "password_generator.py").exists())
        self.assertTrue((self.workspace / "test_password_generator.py").exists())
        self.assertIn("generate_password", t_messy2.distilled_spec.extracted_functions)
        self.assertIn("check_strength", t_messy2.distilled_spec.extracted_functions)


if __name__ == "__main__":
    unittest.main()
