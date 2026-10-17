"""Formal Evaluation Mechanism 1: Evidence-based Verification."""

from __future__ import annotations

import ast
import json
import py_compile
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent911.capabilities.filesystem import WorkspaceBoundaryError, resolve_in_workspace
from agent911.capabilities.registry import CapabilityRegistry
from agent911.memory import PersistentMemoryStore
from agent911.models import (
    ActionStatus,
    CapabilityAvailability,
    TaskRequirement,
    TaskState,
    VerificationCheckResult,
    VerificationStatus,
    new_id,
)


class Verifier:
    """
    Formal Evaluation Mechanism 1: Verification.
    Examines task requirements and concrete execution artifacts/evidence to determine
    whether the intended outcome has been achieved.
    """

    def __init__(
        self,
        workspace_root: Path,
        registry: CapabilityRegistry,
        memory_store: PersistentMemoryStore,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.registry = registry
        self.memory_store = memory_store

    def verify_task(self, task: TaskState) -> List[VerificationCheckResult]:
        """Run all applicable verification checks for a task and update task state."""
        results: List[VerificationCheckResult] = []

        # 1. Verify completion of required planned steps
        step_check = self._verify_planned_steps(task)
        results.append(step_check)

        # 2. Verify each explicit task requirement against real evidence
        for req in task.requirements:
            chk = self.verify_requirement(task, req)
            if chk.status == VerificationStatus.PASSED:
                req.satisfied = True
            elif chk.status == VerificationStatus.FAILED:
                req.satisfied = False
            else:
                req.satisfied = None
            results.append(chk)

        task.verification_results = results
        task.touch()
        return results

    def _verify_planned_steps(self, task: TaskState) -> VerificationCheckResult:
        if not task.planned_actions:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task.task_id,
                requirement="Actionable plan steps executed",
                method="execution_record_inspection",
                observed_result="No planned actions were generated or executed.",
                status=VerificationStatus.INCONCLUSIVE,
                evidence={"evidence_source": "execution_record", "planned_count": 0},
                failure_explanation="Task had no actionable execution steps.",
                required=True,
            )

        required_actions = [a for a in task.planned_actions if a.required]
        completed = [a for a in required_actions if a.status == ActionStatus.COMPLETED]
        failed = [a for a in required_actions if a.status == ActionStatus.FAILED]
        pending = [
            a
            for a in required_actions
            if a.status in (ActionStatus.PENDING, ActionStatus.IN_PROGRESS)
        ]

        if len(completed) == len(required_actions):
            if len(task.capability_calls) >= len(completed):
                return VerificationCheckResult(
                    check_id=new_id("vchk"),
                    task_id=task.task_id,
                    requirement="All required planned steps completed",
                    method="execution_record_inspection",
                    observed_result=(
                        f"All {len(completed)}/{len(required_actions)} required steps completed "
                        f"with {len(task.capability_calls)} recorded capability call(s)."
                    ),
                    status=VerificationStatus.PASSED,
                    evidence={
                        "evidence_source": "execution_record",
                        "completed_steps": [a.step_id for a in completed],
                        "capability_calls_count": len(task.capability_calls),
                    },
                    required=True,
                )
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task.task_id,
                requirement="All required planned steps completed",
                method="execution_record_inspection",
                observed_result=(
                    f"Steps marked completed ({len(completed)}), but only "
                    f"{len(task.capability_calls)} capability call record(s) found."
                ),
                status=VerificationStatus.INCONCLUSIVE,
                evidence={
                    "evidence_source": "execution_record",
                    "completed_steps": [a.step_id for a in completed],
                    "capability_calls_count": len(task.capability_calls),
                },
                failure_explanation="Insufficient capability call records to verify step execution.",
                required=True,
            )

        return VerificationCheckResult(
            check_id=new_id("vchk"),
            task_id=task.task_id,
            requirement="All required planned steps completed",
            method="execution_record_inspection",
            observed_result=(
                f"Completed {len(completed)}/{len(required_actions)} required steps "
                f"(failed={len(failed)}, pending={len(pending)})."
            ),
            status=VerificationStatus.FAILED,
            evidence={
                "evidence_source": "execution_record",
                "completed_steps": [a.step_id for a in completed],
                "failed_steps": [a.step_id for a in failed],
                "pending_steps": [a.step_id for a in pending],
            },
            failure_explanation=(
                f"{len(failed)} step(s) failed and {len(pending)} step(s) remain unexecuted."
            ),
            required=True,
        )

    def verify_requirement(
        self, task: TaskState, req: TaskRequirement
    ) -> VerificationCheckResult:
        """Dispatch a single TaskRequirement to the appropriate evidence checker."""
        cat = req.category.lower().strip()
        if cat == "file_exists":
            return self._check_file_exists(task.task_id, req)
        if cat == "directory_exists":
            return self._check_directory_exists(task.task_id, req)
        if cat == "file_content":
            return self._check_file_content(task.task_id, req)
        if cat == "json_valid":
            return self._check_json_valid(task.task_id, req)
        if cat == "code_compiles":
            return self._check_code_compiles(task.task_id, req)
        if cat in ("function_exists", "interface_exists"):
            return self._check_interface_exists(task.task_id, req)
        if cat == "tests_pass":
            return self._check_tests_pass(task, req)
        if cat == "command_succeeds":
            return self._check_command_succeeds(task, req)
        if cat == "capability_operational":
            return self._check_capability_operational(task.task_id, req)
        if cat == "memory_stored":
            return self._check_memory_stored(task.task_id, req)
        if cat == "experience_recorded":
            return self._check_experience_recorded(task.task_id, req)
        if cat == "output_present":
            return self._check_output_present(task, req)
        if cat == "config_reloadable":
            return self._check_config_reloadable(task.task_id, req)
        if cat == "effective_model_config":
            return self._check_effective_model_config(task.task_id, req)
        if cat == "ai_assertion":
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task.task_id,
                requirement=req.description,
                method="model_interpretation",
                observed_result=str(req.expected or "AI assertion without operational evidence"),
                status=VerificationStatus.INCONCLUSIVE,
                evidence={"evidence_source": "model_interpretation"},
                failure_explanation=(
                    "AI assertion alone is insufficient to establish objective verification."
                ),
                remaining_uncertainty="No independent filesystem, subprocess, or state evidence was collected.",
                required=req.required,
                is_ai_assertion_only=True,
            )
        if cat == "unavailable_check":
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task.task_id,
                requirement=req.description,
                method="unavailable_inspector",
                observed_result="Verification method is unavailable in current environment.",
                status=VerificationStatus.UNAVAILABLE,
                evidence={"evidence_source": "none", "reason": str(req.expected or "unavailable")},
                failure_explanation="Required verification tool or resource is unavailable.",
                remaining_uncertainty="Cannot verify requirement while inspector is unavailable.",
                required=req.required,
            )

        # Default fallback for unknown category: never fake PASSED; mark INCONCLUSIVE
        return VerificationCheckResult(
            check_id=new_id("vchk"),
            task_id=task.task_id,
            requirement=req.description,
            method=f"unrecognized_category:{cat}",
            observed_result=f"No concrete verifier implemented for category '{cat}'.",
            status=VerificationStatus.INCONCLUSIVE,
            evidence={"evidence_source": "none", "category": cat},
            failure_explanation=f"Cannot objectively verify unknown requirement category '{cat}'.",
            remaining_uncertainty=f"Category '{cat}' has no registered objective evidence verifier.",
            required=req.required,
        )

    def _resolve_target_path(self, target: Optional[str]) -> Optional[Path]:
        if not target:
            return None
        try:
            return resolve_in_workspace(self.workspace_root, target)
        except (WorkspaceBoundaryError, ValueError):
            return None

    def _check_file_exists(
        self, task_id: str, req: TaskRequirement
    ) -> VerificationCheckResult:
        path = self._resolve_target_path(req.target)
        if path is None:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="filesystem_inspection",
                observed_result=f"Invalid or out-of-workspace path: {req.target!r}",
                status=VerificationStatus.FAILED,
                evidence={"evidence_source": "filesystem", "target": req.target},
                failure_explanation="Target path is invalid or outside workspace boundary.",
                required=req.required,
            )
        if path.exists() and path.is_file():
            size = path.stat().st_size
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="filesystem_inspection",
                observed_result=f"File '{req.target}' exists ({size} bytes).",
                status=VerificationStatus.PASSED,
                evidence={
                    "evidence_source": "filesystem",
                    "target": req.target,
                    "exists": True,
                    "size_bytes": size,
                },
                required=req.required,
            )
        return VerificationCheckResult(
            check_id=new_id("vchk"),
            task_id=task_id,
            requirement=req.description,
            method="filesystem_inspection",
            observed_result=f"File '{req.target}' does not exist.",
            status=VerificationStatus.FAILED,
            evidence={"evidence_source": "filesystem", "target": req.target, "exists": False},
            failure_explanation=f"Expected file '{req.target}' was not found on disk.",
            required=req.required,
        )

    def _check_directory_exists(
        self, task_id: str, req: TaskRequirement
    ) -> VerificationCheckResult:
        path = self._resolve_target_path(req.target)
        if path is not None and path.exists() and path.is_dir():
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="filesystem_inspection",
                observed_result=f"Directory '{req.target}' exists.",
                status=VerificationStatus.PASSED,
                evidence={"evidence_source": "filesystem", "target": req.target, "is_dir": True},
                required=req.required,
            )
        return VerificationCheckResult(
            check_id=new_id("vchk"),
            task_id=task_id,
            requirement=req.description,
            method="filesystem_inspection",
            observed_result=f"Directory '{req.target}' does not exist.",
            status=VerificationStatus.FAILED,
            evidence={"evidence_source": "filesystem", "target": req.target, "is_dir": False},
            failure_explanation=f"Expected directory '{req.target}' was not found.",
            required=req.required,
        )

    def _check_file_content(
        self, task_id: str, req: TaskRequirement
    ) -> VerificationCheckResult:
        path = self._resolve_target_path(req.target)
        if path is None or not path.exists() or not path.is_file():
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="file_content_inspection",
                observed_result=f"Cannot inspect content; file '{req.target}' does not exist.",
                status=VerificationStatus.FAILED,
                evidence={"evidence_source": "filesystem", "target": req.target, "exists": False},
                failure_explanation=f"File '{req.target}' does not exist.",
                required=req.required,
            )

        content = path.read_text(encoding="utf-8", errors="replace")
        expected = req.expected
        if expected is None:
            # Verify non-empty content
            if content.strip():
                return VerificationCheckResult(
                    check_id=new_id("vchk"),
                    task_id=task_id,
                    requirement=req.description,
                    method="file_content_inspection",
                    observed_result=f"File '{req.target}' has non-empty content ({len(content)} chars).",
                    status=VerificationStatus.PASSED,
                    evidence={
                        "evidence_source": "filesystem",
                        "target": req.target,
                        "length": len(content),
                    },
                    required=req.required,
                )
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="file_content_inspection",
                observed_result=f"File '{req.target}' is empty.",
                status=VerificationStatus.FAILED,
                evidence={"evidence_source": "filesystem", "target": req.target, "length": 0},
                failure_explanation=f"File '{req.target}' exists but contains no content.",
                required=req.required,
            )

        expected_substrings = [expected] if isinstance(expected, str) else list(expected)
        missing = [s for s in expected_substrings if str(s) not in content]
        if not missing:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="file_content_inspection",
                observed_result=f"File '{req.target}' contains all expected content.",
                status=VerificationStatus.PASSED,
                evidence={
                    "evidence_source": "filesystem",
                    "target": req.target,
                    "matched_substrings": expected_substrings,
                },
                required=req.required,
            )
        return VerificationCheckResult(
            check_id=new_id("vchk"),
            task_id=task_id,
            requirement=req.description,
            method="file_content_inspection",
            observed_result=f"File '{req.target}' is missing expected content: {missing}",
            status=VerificationStatus.FAILED,
            evidence={
                "evidence_source": "filesystem",
                "target": req.target,
                "missing_substrings": missing,
                "actual_preview": content[:200],
            },
            failure_explanation=f"File '{req.target}' did not contain required text {missing}.",
            required=req.required,
        )

    def _check_json_valid(
        self, task_id: str, req: TaskRequirement
    ) -> VerificationCheckResult:
        path = self._resolve_target_path(req.target)
        if path is None or not path.exists() or not path.is_file():
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="json_validation",
                observed_result=f"Target JSON file '{req.target}' does not exist.",
                status=VerificationStatus.FAILED,
                evidence={"evidence_source": "filesystem", "target": req.target},
                failure_explanation=f"JSON file '{req.target}' not found.",
                required=req.required,
            )
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="json_validation",
                observed_result=f"Invalid JSON in '{req.target}': {exc}",
                status=VerificationStatus.FAILED,
                evidence={"evidence_source": "json_parser", "error": str(exc)},
                failure_explanation=f"File '{req.target}' is not valid JSON: {exc}",
                required=req.required,
            )

        if isinstance(req.expected, dict) and isinstance(parsed, dict):
            for k, v in req.expected.items():
                if k not in parsed or (v is not None and parsed[k] != v):
                    return VerificationCheckResult(
                        check_id=new_id("vchk"),
                        task_id=task_id,
                        requirement=req.description,
                        method="json_validation",
                        observed_result=(
                            f"JSON key '{k}' mismatch in '{req.target}': "
                            f"expected {v!r}, got {parsed.get(k)!r}"
                        ),
                        status=VerificationStatus.FAILED,
                        evidence={"evidence_source": "json_parser", "parsed": parsed},
                        failure_explanation=f"Expected JSON key '{k}'={v!r} not satisfied.",
                        required=req.required,
                    )

        return VerificationCheckResult(
            check_id=new_id("vchk"),
            task_id=task_id,
            requirement=req.description,
            method="json_validation",
            observed_result=f"File '{req.target}' contains valid JSON structure.",
            status=VerificationStatus.PASSED,
            evidence={"evidence_source": "json_parser", "target": req.target},
            required=req.required,
        )

    def _check_code_compiles(
        self, task_id: str, req: TaskRequirement
    ) -> VerificationCheckResult:
        path = self._resolve_target_path(req.target)
        if path is None or not path.exists() or not path.is_file():
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="python_ast_and_bytecode_compile",
                observed_result=f"Source file '{req.target}' does not exist.",
                status=VerificationStatus.FAILED,
                evidence={"evidence_source": "ast_inspection", "target": req.target},
                failure_explanation=f"Source file '{req.target}' not found.",
                required=req.required,
            )

        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            py_compile.compile(str(path), doraise=True)
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="python_ast_and_bytecode_compile",
                observed_result=f"Python file '{req.target}' compiled without syntax errors.",
                status=VerificationStatus.PASSED,
                evidence={"evidence_source": "ast_inspection", "target": req.target},
                required=req.required,
            )
        except Exception as exc:
            err_str = f"{type(exc).__name__}: {exc}"
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="python_ast_and_bytecode_compile",
                observed_result=f"Compilation failed for '{req.target}': {err_str}",
                status=VerificationStatus.FAILED,
                evidence={
                    "evidence_source": "ast_inspection",
                    "target": req.target,
                    "error": err_str,
                },
                failure_explanation=f"Syntax or compilation error in '{req.target}': {err_str}",
                required=req.required,
            )

    def _check_interface_exists(
        self, task_id: str, req: TaskRequirement
    ) -> VerificationCheckResult:
        path = self._resolve_target_path(req.target)
        if path is None or not path.exists() or not path.is_file():
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="ast_symbol_inspection",
                observed_result=f"Source file '{req.target}' does not exist.",
                status=VerificationStatus.FAILED,
                evidence={"evidence_source": "ast_inspection", "target": req.target},
                failure_explanation=f"Source file '{req.target}' not found.",
                required=req.required,
            )

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="ast_symbol_inspection",
                observed_result=f"SyntaxError while parsing '{req.target}': {exc}",
                status=VerificationStatus.FAILED,
                evidence={"evidence_source": "ast_inspection", "error": str(exc)},
                failure_explanation=f"Cannot inspect symbols due to SyntaxError: {exc}",
                required=req.required,
            )

        defined_names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined_names.add(node.name)

        expected_symbols = (
            [req.expected]
            if isinstance(req.expected, str)
            else list(req.expected or [])
        )
        missing = [s for s in expected_symbols if s not in defined_names]
        if not missing:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="ast_symbol_inspection",
                observed_result=(
                    f"Required symbol(s) {expected_symbols} found in '{req.target}'."
                ),
                status=VerificationStatus.PASSED,
                evidence={
                    "evidence_source": "ast_inspection",
                    "target": req.target,
                    "found_symbols": expected_symbols,
                },
                required=req.required,
            )
        return VerificationCheckResult(
            check_id=new_id("vchk"),
            task_id=task_id,
            requirement=req.description,
            method="ast_symbol_inspection",
            observed_result=(
                f"Missing symbol(s) {missing} in '{req.target}' (found: {sorted(defined_names)})."
            ),
            status=VerificationStatus.FAILED,
            evidence={
                "evidence_source": "ast_inspection",
                "target": req.target,
                "missing_symbols": missing,
                "defined_symbols": sorted(defined_names),
            },
            failure_explanation=f"Required function/class {missing} not defined in '{req.target}'.",
            required=req.required,
        )

    def _check_tests_pass(
        self, task: TaskState, req: TaskRequirement
    ) -> VerificationCheckResult:
        test_calls = [c for c in task.capability_calls if c.capability_id == "test.run"]
        if not test_calls:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task.task_id,
                requirement=req.description,
                method="test_execution_record_inspection",
                observed_result="No test execution record found.",
                status=VerificationStatus.INCONCLUSIVE,
                evidence={"evidence_source": "test_execution", "calls": 0},
                failure_explanation="Automated tests were not executed during this task.",
                required=req.required,
            )

        latest = test_calls[-1]
        data = latest.output_data if isinstance(latest.output_data, dict) else {}
        tests_ran = int(data.get("tests_ran", 0))
        exit_code = data.get("exit_code")
        if latest.success and exit_code == 0 and tests_ran > 0:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task.task_id,
                requirement=req.description,
                method="test_execution_record_inspection",
                observed_result=f"Test suite passed ({tests_ran} test(s) ran, exit_code=0).",
                status=VerificationStatus.PASSED,
                evidence={
                    "evidence_source": "test_execution",
                    "tests_ran": tests_ran,
                    "exit_code": exit_code,
                    "target": data.get("target"),
                },
                required=req.required,
            )
        return VerificationCheckResult(
            check_id=new_id("vchk"),
            task_id=task.task_id,
            requirement=req.description,
            method="test_execution_record_inspection",
            observed_result=f"Tests failed or did not run (tests_ran={tests_ran}, exit_code={exit_code}).",
            status=VerificationStatus.FAILED,
            evidence={
                "evidence_source": "test_execution",
                "tests_ran": tests_ran,
                "exit_code": exit_code,
                "error": latest.error,
            },
            failure_explanation=latest.error or "Automated tests did not pass.",
            required=req.required,
        )

    def _check_command_succeeds(
        self, task: TaskState, req: TaskRequirement
    ) -> VerificationCheckResult:
        cmd_calls = [c for c in task.capability_calls if c.capability_id == "cmd.run"]
        if not cmd_calls:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task.task_id,
                requirement=req.description,
                method="command_execution_record_inspection",
                observed_result="No command execution record found.",
                status=VerificationStatus.INCONCLUSIVE,
                evidence={"evidence_source": "command_output", "calls": 0},
                failure_explanation="Command was never executed.",
                required=req.required,
            )
        latest = cmd_calls[-1]
        data = latest.output_data if isinstance(latest.output_data, dict) else {}
        stdout = str(data.get("stdout", ""))
        exit_code = data.get("exit_code")
        expected_out = req.expected

        if not latest.success or exit_code != 0:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task.task_id,
                requirement=req.description,
                method="command_execution_record_inspection",
                observed_result=f"Command failed with exit_code={exit_code}.",
                status=VerificationStatus.FAILED,
                evidence={
                    "evidence_source": "command_output",
                    "exit_code": exit_code,
                    "stderr": data.get("stderr", ""),
                },
                failure_explanation=latest.error or f"Command exited with {exit_code}.",
                required=req.required,
            )

        if expected_out and str(expected_out) not in stdout:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task.task_id,
                requirement=req.description,
                method="command_execution_record_inspection",
                observed_result=(
                    f"Command exited 0, but stdout did not contain '{expected_out}'."
                ),
                status=VerificationStatus.FAILED,
                evidence={
                    "evidence_source": "command_output",
                    "stdout": stdout[:300],
                    "expected": expected_out,
                },
                failure_explanation=f"Expected output '{expected_out}' not found in stdout.",
                required=req.required,
            )

        return VerificationCheckResult(
            check_id=new_id("vchk"),
            task_id=task.task_id,
            requirement=req.description,
            method="command_execution_record_inspection",
            observed_result=f"Command succeeded (exit_code=0, stdout_len={len(stdout)}).",
            status=VerificationStatus.PASSED,
            evidence={
                "evidence_source": "command_output",
                "exit_code": exit_code,
                "stdout_preview": stdout[:200],
            },
            required=req.required,
        )

    def _check_capability_operational(
        self, task_id: str, req: TaskRequirement
    ) -> VerificationCheckResult:
        cap_id = str(req.target or req.expected or "")
        cap = self.registry.get(cap_id)
        if cap is None:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="capability_registry_inspection",
                observed_result=f"Capability '{cap_id}' is not registered.",
                status=VerificationStatus.FAILED,
                evidence={"evidence_source": "capability_registry", "capability_id": cap_id},
                failure_explanation=f"Capability '{cap_id}' was not found in registry.",
                required=req.required,
            )
        avail = cap.check_availability(self.registry.config)
        if avail == CapabilityAvailability.AVAILABLE:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="capability_registry_inspection",
                observed_result=f"Capability '{cap_id}' is registered and AVAILABLE.",
                status=VerificationStatus.PASSED,
                evidence={
                    "evidence_source": "capability_registry",
                    "capability_id": cap_id,
                    "availability": avail.value,
                    "verified": cap.verified,
                    "live_tested": cap.live_tested,
                },
                required=req.required,
            )
        return VerificationCheckResult(
            check_id=new_id("vchk"),
            task_id=task_id,
            requirement=req.description,
            method="capability_registry_inspection",
            observed_result=f"Capability '{cap_id}' has availability={avail.value}.",
            status=VerificationStatus.FAILED,
            evidence={
                "evidence_source": "capability_registry",
                "capability_id": cap_id,
                "availability": avail.value,
            },
            failure_explanation=f"Capability '{cap_id}' is not operational ({avail.value}).",
            required=req.required,
        )

    def _check_memory_stored(
        self, task_id: str, req: TaskRequirement
    ) -> VerificationCheckResult:
        key = req.target
        records = self.memory_store.retrieve_memories(key=key, limit=5)
        if not records and req.expected:
            records = self.memory_store.retrieve_memories(query=str(req.expected), limit=5)
        if records:
            if req.expected and not any(str(req.expected) in r.content for r in records):
                return VerificationCheckResult(
                    check_id=new_id("vchk"),
                    task_id=task_id,
                    requirement=req.description,
                    method="sqlite_memory_inspection",
                    observed_result=f"Memory record found for key={key!r} but content did not match.",
                    status=VerificationStatus.FAILED,
                    evidence={
                        "evidence_source": "sqlite_memory",
                        "records": [r.to_dict() for r in records],
                    },
                    failure_explanation=f"Stored memory did not contain '{req.expected}'.",
                    required=req.required,
                )
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="sqlite_memory_inspection",
                observed_result=f"Confirmed {len(records)} persisted memory record(s) in SQLite.",
                status=VerificationStatus.PASSED,
                evidence={
                    "evidence_source": "sqlite_memory",
                    "memory_id": records[0].memory_id,
                    "key": records[0].key,
                },
                required=req.required,
            )
        return VerificationCheckResult(
            check_id=new_id("vchk"),
            task_id=task_id,
            requirement=req.description,
            method="sqlite_memory_inspection",
            observed_result=f"No matching memory record found for key={key!r}.",
            status=VerificationStatus.FAILED,
            evidence={"evidence_source": "sqlite_memory", "key": key},
            failure_explanation="Expected memory record was not persisted.",
            required=req.required,
        )

    def _check_experience_recorded(
        self, task_id: str, req: TaskRequirement
    ) -> VerificationCheckResult:
        records = self.memory_store.retrieve_experiences( query=req.target, limit=5 )
        if records:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="sqlite_experience_inspection",
                observed_result=f"Confirmed {len(records)} experience record(s) in SQLite.",
                status=VerificationStatus.PASSED,
                evidence={
                    "evidence_source": "sqlite_experience",
                    "experience_id": records[0].experience_id,
                },
                required=req.required,
            )
        return VerificationCheckResult(
            check_id=new_id("vchk"),
            task_id=task_id,
            requirement=req.description,
            method="sqlite_experience_inspection",
            observed_result="No matching experience record found.",
            status=VerificationStatus.FAILED,
            evidence={"evidence_source": "sqlite_experience"},
            failure_explanation="Expected experience record was not persisted.",
            required=req.required,
        )

    def _check_output_present(
        self, task: TaskState, req: TaskRequirement
    ) -> VerificationCheckResult:
        key = req.target
        if key and key in task.relevant_outputs and task.relevant_outputs[key] is not None:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task.task_id,
                requirement=req.description,
                method="task_output_inspection",
                observed_result=f"Task output '{key}' is present.",
                status=VerificationStatus.PASSED,
                evidence={"evidence_source": "execution_record", "output_key": key},
                required=req.required,
            )
        if not key and task.relevant_outputs:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task.task_id,
                requirement=req.description,
                method="task_output_inspection",
                observed_result=f"Task produced {len(task.relevant_outputs)} output entry(ies).",
                status=VerificationStatus.PASSED,
                evidence={
                    "evidence_source": "execution_record",
                    "keys": list(task.relevant_outputs.keys()),
                },
                required=req.required,
            )
        return VerificationCheckResult(
            check_id=new_id("vchk"),
            task_id=task.task_id,
            requirement=req.description,
            method="task_output_inspection",
            observed_result=f"Required task output {key!r} is missing.",
            status=VerificationStatus.FAILED,
            evidence={"evidence_source": "execution_record", "target_key": key},
            failure_explanation=f"Expected output '{key}' was not produced.",
            required=req.required,
        )

    def _check_config_reloadable(
        self, task_id: str, req: TaskRequirement
    ) -> VerificationCheckResult:
        from agent911.config import AgentConfig

        path = self._resolve_target_path(req.target) if req.target else None
        try:
            loaded = AgentConfig.load(
                config_path=path, workspace_override=self.workspace_root
            )
            loaded.validate()
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="config_reload_validation",
                observed_result=(
                    f"Configuration reloaded and validated (schema_version={loaded.schema_version}, "
                    f"active_model={loaded.active_model})."
                ),
                status=VerificationStatus.PASSED,
                evidence={
                    "evidence_source": "config_loader",
                    "schema_version": loaded.schema_version,
                    "active_model": loaded.active_model,
                    "default_model": loaded.default_model,
                    "warnings": loaded.validation_warnings,
                },
                required=req.required,
            )
        except Exception as exc:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="config_reload_validation",
                observed_result=f"Configuration failed to reload or validate: {exc}",
                status=VerificationStatus.FAILED,
                evidence={"evidence_source": "config_loader", "error": str(exc)},
                failure_explanation=f"Configuration could not be reloaded: {exc}",
                required=req.required,
            )

    def _check_effective_model_config(
        self, task_id: str, req: TaskRequirement
    ) -> VerificationCheckResult:
        from agent911.config import AgentConfig

        cfg = self.registry.config
        if req.target:
            path = self._resolve_target_path(req.target)
            if path and path.exists():
                cfg = AgentConfig.load(
                    config_path=path, workspace_override=self.workspace_root
                )

        expected = req.expected if isinstance(req.expected, dict) else {}
        mismatches = []
        observed = {
            "active_model": cfg.active_model,
            "default_model": cfg.default_model,
            "fallback_model": cfg.fallback_model,
            "provider": cfg.model.provider,
            "model_id": cfg.model.model_id,
            "endpoint": cfg.model.endpoint,
        }
        for key, exp_val in expected.items():
            act_val = observed.get(key)
            if act_val != exp_val:
                mismatches.append(f"{key}: expected {exp_val!r}, got {act_val!r}")

        if not mismatches:
            return VerificationCheckResult(
                check_id=new_id("vchk"),
                task_id=task_id,
                requirement=req.description,
                method="effective_model_config_inspection",
                observed_result=f"Effective model configuration matches expected settings ({observed}).",
                status=VerificationStatus.PASSED,
                evidence={
                    "evidence_source": "effective_config",
                    "observed": observed,
                },
                required=req.required,
            )
        return VerificationCheckResult(
            check_id=new_id("vchk"),
            task_id=task_id,
            requirement=req.description,
            method="effective_model_config_inspection",
            observed_result=f"Effective model configuration mismatch: {'; '.join(mismatches)}",
            status=VerificationStatus.FAILED,
            evidence={
                "evidence_source": "effective_config",
                "observed": observed,
                "mismatches": mismatches,
            },
            failure_explanation="; ".join(mismatches),
            required=req.required,
        )
