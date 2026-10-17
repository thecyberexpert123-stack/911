"""Formal Evaluation Mechanism 2: Live Run Testing of actual runtime behavior."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent911.capabilities.base import CapabilityContext
from agent911.capabilities.filesystem import WorkspaceBoundaryError, resolve_in_workspace
from agent911.memory import PersistentMemoryStore
from agent911.models import (
    LiveRunTestRecord,
    LiveTestStatus,
    TaskRequirement,
    TaskState,
    new_id,
)


class LiveRunTester:
    """
    Formal Evaluation Mechanism 2: Live Run Testing.
    Executes relevant target functionality at runtime in an appropriate environment
    to establish whether it actually behaves as expected.
    """

    def __init__(self, context: CapabilityContext) -> None:
        self.context = context

    def _base_environment(self, isolation: str = "workspace") -> Dict[str, Any]:
        return {
            "workspace": str(self.context.workspace_path),
            "python_version": platform.python_version(),
            "platform": sys.platform,
            "isolation": isolation,
        }

    def run_for_task(self, task: TaskState) -> List[LiveRunTestRecord]:
        """
        Determine and execute appropriate Live Run Tests for a task based on the
        operations performed and artifacts created.
        """
        records: List[LiveRunTestRecord] = []
        ws = self.context.workspace_path.resolve()

        # 1. Check if any Python scripts/modules were created or updated during the task
        written_py_files: List[str] = []
        written_json_files: List[str] = []
        for call in task.capability_calls:
            if call.success and call.capability_id in ("fs.write", "fs.update"):
                p = call.parameters.get("path", "")
                if isinstance(p, str):
                    if p.endswith(".py") and p not in written_py_files:
                        written_py_files.append(p)
                    elif p.endswith(".json") and p not in written_json_files:
                        written_json_files.append(p)

        for py_rel in written_py_files:
            # Find if there is a function_exists requirement for this file to exercise
            func_reqs = [
                r
                for r in task.requirements
                if r.category in ("function_exists", "interface_exists") and r.target == py_rel
            ]
            if func_reqs:
                for freq in func_reqs:
                    symbols = (
                        [freq.expected]
                        if isinstance(freq.expected, str)
                        else list(freq.expected or [])
                    )
                    for sym in symbols:
                        records.append(self.test_python_symbol_execution(task.task_id, py_rel, sym))
            else:
                records.append(self.test_python_file_execution(task.task_id, py_rel))

        # 2. Check if any JSON configuration/data artifacts were written: validate live load
        for json_rel in written_json_files:
            records.append(self.test_json_artifact_load(task.task_id, json_rel))

        # 3. Check if any new capability was developed or required to be operational
        for req in task.requirements:
            if req.category == "capability_operational" and req.target:
                test_inputs = req.expected if isinstance(req.expected, dict) else {}
                records.append(
                    self.test_capability_invocation(
                        task_id=task.task_id,
                        capability_id=req.target,
                        sample_params=test_inputs,
                    )
                )

        # 4. Check if memory persistence was part of the task: exercise live DB reopen read
        if any(c.capability_id in ("memory.store", "memory.update") for c in task.capability_calls):
            records.append(self.test_memory_persistence_reload(task.task_id))

        # 5. If none of the specific artifact tests triggered, exercise the primary capability or command
        if not records and task.capability_calls:
            last_call = task.capability_calls[-1]
            records.append(self._evaluate_executed_capability_live(task.task_id, last_call))

        task.live_run_results = records
        task.touch()
        return records

    def test_python_file_execution(
        self,
        task_id: str,
        rel_path: str,
        args: Optional[List[str]] = None,
        expected_stdout_substring: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> LiveRunTestRecord:
        """Execute a Python file in a live subprocess and inspect exit code and output."""
        timeout = timeout_seconds or self.context.config.command_timeout_seconds
        try:
            target = resolve_in_workspace(self.context.workspace_path, rel_path)
        except (WorkspaceBoundaryError, ValueError) as exc:
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation=f"python3 {rel_path}",
                target_component=rel_path,
                environment=self._base_environment(),
                exit_code=None,
                stdout="",
                stderr=str(exc),
                expected_behavior_occurred=False,
                status=LiveTestStatus.FAILED,
                limitations="Target path outside workspace boundary.",
            )

        if not target.exists():
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation=f"python3 {rel_path}",
                target_component=rel_path,
                environment=self._base_environment(),
                exit_code=None,
                stdout="",
                stderr=f"File not found: {rel_path}",
                expected_behavior_occurred=False,
                status=LiveTestStatus.FAILED,
            )

        cmd = [sys.executable, str(target)] + (args or [])
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.context.workspace_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = (time.monotonic() - t0) * 1000.0
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            ok = proc.returncode == 0
            if ok and expected_stdout_substring:
                ok = expected_stdout_substring in stdout
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation=" ".join(cmd),
                target_component=rel_path,
                environment=self._base_environment("subprocess"),
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                expected_behavior_occurred=ok,
                status=LiveTestStatus.PASSED if ok else LiveTestStatus.FAILED,
                reproduction_command=" ".join(cmd),
                is_mocked=False,
                duration_ms=elapsed,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = (time.monotonic() - t0) * 1000.0
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation=" ".join(cmd),
                target_component=rel_path,
                environment=self._base_environment("subprocess"),
                exit_code=None,
                stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                stderr=f"Process timed out after {timeout}s.",
                expected_behavior_occurred=False,
                status=LiveTestStatus.TIMED_OUT,
                limitations=f"Exceeded timeout of {timeout}s.",
                reproduction_command=" ".join(cmd),
                is_mocked=False,
                duration_ms=elapsed,
            )

    def test_python_symbol_execution(
        self,
        task_id: str,
        rel_path: str,
        symbol_name: str,
        call_args: Optional[List[Any]] = None,
        expected_return: Any = None,
    ) -> LiveRunTestRecord:
        """
        Import a Python file and actually invoke the target function/class at runtime
        in an isolated subprocess (never merely importing without exercising behavior).
        """
        try:
            target = resolve_in_workspace(self.context.workspace_path, rel_path)
        except (WorkspaceBoundaryError, ValueError) as exc:
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation=f"invoke {rel_path}:{symbol_name}",
                target_component=f"{rel_path}:{symbol_name}",
                environment=self._base_environment("subprocess"),
                exit_code=None,
                stdout="",
                stderr=str(exc),
                expected_behavior_occurred=False,
                status=LiveTestStatus.FAILED,
            )

        if not target.exists():
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation=f"invoke {rel_path}:{symbol_name}",
                target_component=f"{rel_path}:{symbol_name}",
                environment=self._base_environment("subprocess"),
                exit_code=None,
                stdout="",
                stderr=f"File not found: {rel_path}",
                expected_behavior_occurred=False,
                status=LiveTestStatus.FAILED,
            )

        harness = (
            "import importlib.util, inspect, json, sys\n"
            f"spec = importlib.util.spec_from_file_location('target_mod', {str(target)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            f"sym = getattr(mod, {symbol_name!r})\n"
            f"explicit_args = {json.dumps(call_args) if call_args is not None else 'None'}\n"
            "if explicit_args is not None:\n"
            "    res = sym(*explicit_args)\n"
            "elif inspect.isclass(sym):\n"
            "    inst = sym()\n"
            "    res = type(inst).__name__\n"
            "else:\n"
            "    sig = inspect.signature(sym)\n"
            "    sample = []\n"
            "    for p in sig.parameters.values():\n"
            "        if p.default is not inspect.Parameter.empty:\n"
            "            continue\n"
            "        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):\n"
            "            continue\n"
            "        name_l = p.name.lower()\n"
            "        if any(k in name_l for k in ('text', 'str', 'name', 'word', 'msg', 'path')):\n"
            "            sample.append('test_value')\n"
            "        elif any(k in name_l for k in ('items', 'list', 'values', 'nums', 'numbers', 'data')):\n"
            "            sample.append([1, 2, 3])\n"
            "        else:\n"
            "            sample.append(2)\n"
            "    res = sym(*sample)\n"
            "print(json.dumps({'invoked': True, 'result': res}, default=str))\n"
        )
        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-c", harness],
            cwd=str(self.context.workspace_path),
            capture_output=True,
            text=True,
            timeout=self.context.config.command_timeout_seconds,
        )
        elapsed = (time.monotonic() - t0) * 1000.0
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        ok = proc.returncode == 0 and '"invoked": true' in stdout.lower()
        if ok and expected_return is not None:
            try:
                parsed = json.loads(stdout.splitlines()[-1])
                ok = parsed.get("result") == expected_return
            except Exception:
                ok = False

        return LiveRunTestRecord(
            test_id=new_id("ltest"),
            task_id=task_id,
            executed_operation=f"python3 -c '<invoke {symbol_name} from {rel_path}>'",
            target_component=f"{rel_path}:{symbol_name}",
            environment=self._base_environment("subprocess"),
            exit_code=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            expected_behavior_occurred=ok,
            status=LiveTestStatus.PASSED if ok else LiveTestStatus.FAILED,
            reproduction_command=f'{sys.executable} -c "import importlib.util; s=importlib.util.spec_from_file_location(\'m\', {str(target)!r}); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); getattr(m, {symbol_name!r})"',
            is_mocked=False,
            duration_ms=elapsed,
        )

    def test_json_artifact_load(self, task_id: str, rel_path: str) -> LiveRunTestRecord:
        """Load and parse a JSON artifact at runtime to confirm it is usable by consumers."""
        t0 = time.monotonic()
        try:
            target = resolve_in_workspace(self.context.workspace_path, rel_path)
            raw = target.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            elapsed = (time.monotonic() - t0) * 1000.0
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation=f"json.loads(read({rel_path!r}))",
                target_component=rel_path,
                environment=self._base_environment(),
                exit_code=0,
                stdout=f"Loaded JSON type={type(parsed).__name__}, len={len(parsed)}",
                stderr="",
                expected_behavior_occurred=True,
                status=LiveTestStatus.PASSED,
                reproduction_command=f'{sys.executable} -m json.tool {rel_path}',
                is_mocked=False,
                duration_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000.0
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation=f"json.loads(read({rel_path!r}))",
                target_component=rel_path,
                environment=self._base_environment(),
                exit_code=1,
                stdout="",
                stderr=str(exc),
                expected_behavior_occurred=False,
                status=LiveTestStatus.FAILED,
                duration_ms=elapsed,
            )

    def test_capability_invocation(
        self,
        task_id: str,
        capability_id: str,
        sample_params: Dict[str, Any],
        expected_output_subset: Optional[Dict[str, Any]] = None,
    ) -> LiveRunTestRecord:
        """Invoke a registered capability with real parameters and check runtime behavior."""
        t0 = time.monotonic()
        res = self.context.registry.invoke(capability_id, sample_params, self.context)
        elapsed = (time.monotonic() - t0) * 1000.0

        if res.error_type == "CapabilityUnavailableError" and "REQUIRES_CONFIG" in (
            res.error or ""
        ):
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation=f"registry.invoke({capability_id!r}, {sample_params!r})",
                target_component=capability_id,
                environment=self._base_environment(),
                exit_code=None,
                stdout="",
                stderr=res.error or "",
                expected_behavior_occurred=False,
                status=LiveTestStatus.UNAVAILABLE,
                limitations="External credentials or configuration unavailable for live test.",
                duration_ms=elapsed,
            )

        ok = res.success
        if ok and expected_output_subset and isinstance(res.data, dict):
            for k, v in expected_output_subset.items():
                if res.data.get(k) != v:
                    ok = False
                    break

        return LiveRunTestRecord(
            test_id=new_id("ltest"),
            task_id=task_id,
            executed_operation=f"registry.invoke({capability_id!r}, {sample_params!r})",
            target_component=capability_id,
            environment=self._base_environment(),
            exit_code=0 if ok else 1,
            stdout=json.dumps(res.data, default=str)[:500] if res.data is not None else "",
            stderr=res.error or "",
            expected_behavior_occurred=ok,
            status=LiveTestStatus.PASSED if ok else LiveTestStatus.FAILED,
            duration_ms=elapsed,
        )

    def test_memory_persistence_reload(self, task_id: str) -> LiveRunTestRecord:
        """Open a fresh PersistentMemoryStore instance on the DB file to test restart persistence."""
        t0 = time.monotonic()
        try:
            fresh_store = PersistentMemoryStore(self.context.config.db_path)
            records = fresh_store.retrieve_memories(limit=5)
            elapsed = (time.monotonic() - t0) * 1000.0
            ok = len(records) > 0
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation="PersistentMemoryStore(db_path).retrieve_memories()",
                target_component="PersistentMemoryStore",
                environment=self._base_environment("fresh_sqlite_connection"),
                exit_code=0 if ok else 1,
                stdout=f"Retrieved {len(records)} persisted record(s) from fresh SQLite connection.",
                stderr="" if ok else "No records found after re-opening SQLite database.",
                expected_behavior_occurred=ok,
                status=LiveTestStatus.PASSED if ok else LiveTestStatus.FAILED,
                duration_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000.0
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation="PersistentMemoryStore(db_path).retrieve_memories()",
                target_component="PersistentMemoryStore",
                environment=self._base_environment("fresh_sqlite_connection"),
                exit_code=1,
                stdout="",
                stderr=str(exc),
                expected_behavior_occurred=False,
                status=LiveTestStatus.FAILED,
                duration_ms=elapsed,
            )

    def test_command_live(
        self,
        task_id: str,
        command: str | List[str],
        expected_stdout: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> LiveRunTestRecord:
        """Run a real command via cmd.run capability and record a LiveRunTestRecord."""
        params: Dict[str, Any] = {"command": command}
        if timeout_seconds is not None:
            params["timeout_seconds"] = timeout_seconds
        res = self.context.registry.invoke("cmd.run", params, self.context)
        data = res.data if isinstance(res.data, dict) else {}
        timed_out = bool(data.get("timed_out", False))
        stdout = str(data.get("stdout", ""))
        stderr = str(data.get("stderr", "") or (res.error or ""))
        exit_code = data.get("exit_code")
        duration = float(data.get("duration_ms", res.metadata.get("duration_ms", 0.0)))

        if timed_out:
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation=str(command),
                target_component="cmd.run",
                environment=self._base_environment("subprocess"),
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                expected_behavior_occurred=False,
                status=LiveTestStatus.TIMED_OUT,
                limitations=f"Timed out after {timeout_seconds}s.",
                duration_ms=duration,
            )

        ok = res.success and exit_code == 0
        if ok and expected_stdout is not None:
            ok = expected_stdout in stdout

        return LiveRunTestRecord(
            test_id=new_id("ltest"),
            task_id=task_id,
            executed_operation=str(command),
            target_component="cmd.run",
            environment=self._base_environment("subprocess"),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            expected_behavior_occurred=ok,
            status=LiveTestStatus.PASSED if ok else LiveTestStatus.FAILED,
            duration_ms=duration,
        )

    def test_external_model_live(self, task_id: str) -> LiveRunTestRecord:
        """
        Exercise the configured external AI model connection.
        Accurately reports UNAVAILABLE when credentials or network are absent, and sets
        is_mocked=True if using a deterministic test double so it is never confused with
        live external service connectivity.
        """
        report = self.context.model_interface.check_connectivity()
        is_mocked = report.status.value == "DETERMINISTIC_TEST_DOUBLE"
        if not report.is_live_provider:
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation=f"model_interface.check_connectivity({report.provider})",
                target_component="ModelInterface",
                environment=self._base_environment("network"),
                exit_code=None,
                stdout="",
                stderr=report.details,
                expected_behavior_occurred=False,
                status=LiveTestStatus.UNAVAILABLE,
                limitations=report.details,
                reproduction_command="python3 -m agent911 model-test",
                is_mocked=is_mocked,
                duration_ms=report.latency_ms,
            )
        return LiveRunTestRecord(
            test_id=new_id("ltest"),
            task_id=task_id,
            executed_operation=f"model_interface.check_connectivity({report.provider})",
            target_component="ModelInterface",
            environment=self._base_environment("network"),
            exit_code=0,
            stdout=report.details,
            stderr="",
            expected_behavior_occurred=True,
            status=LiveTestStatus.PASSED,
            reproduction_command="python3 -m agent911 model-test",
            is_mocked=False,
            duration_ms=report.latency_ms,
        )

    def _evaluate_executed_capability_live(
        self, task_id: str, call: Any
    ) -> LiveRunTestRecord:
        """Convert or re-verify the primary capability execution as a live runtime record."""
        if call.capability_id in ("fs.list", "fs.read", "fs.stat", "project.inspect", "memory.retrieve", "experience.retrieve", "capability.discover"):
            # Re-invoke read-only/inspection capability to confirm live reproducibility
            return self.test_capability_invocation(task_id, call.capability_id, call.parameters)

        # For command/test/mkdir/model operations, inspect the actual execution result
        data = call.output_data if isinstance(call.output_data, dict) else {}
        if call.error_type in ("CapabilityUnavailableError", "missing_credentials"):
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation=f"{call.capability_id}({call.parameters})",
                target_component=call.capability_id,
                environment=self._base_environment(),
                exit_code=None,
                stdout="",
                stderr=call.error or "",
                expected_behavior_occurred=False,
                status=LiveTestStatus.UNAVAILABLE,
                limitations=call.error or "Capability unavailable due to missing configuration.",
                duration_ms=call.duration_ms,
            )

        timed_out = bool(data.get("timed_out", False)) or call.error_type == "TimeoutError"
        if timed_out:
            return LiveRunTestRecord(
                test_id=new_id("ltest"),
                task_id=task_id,
                executed_operation=f"{call.capability_id}({call.parameters})",
                target_component=call.capability_id,
                environment=self._base_environment(),
                exit_code=None,
                stdout=str(data.get("stdout", "")),
                stderr=call.error or str(data.get("stderr", "")),
                expected_behavior_occurred=False,
                status=LiveTestStatus.TIMED_OUT,
                limitations="Execution timed out.",
                duration_ms=call.duration_ms,
            )

        exit_code = data.get("exit_code", 0 if call.success else 1)
        return LiveRunTestRecord(
            test_id=new_id("ltest"),
            task_id=task_id,
            executed_operation=f"{call.capability_id}({call.parameters})",
            target_component=call.capability_id,
            environment=self._base_environment(),
            exit_code=exit_code,
            stdout=str(data.get("stdout", json.dumps(data, default=str)[:400])),
            stderr=call.error or str(data.get("stderr", "")),
            expected_behavior_occurred=call.success,
            status=LiveTestStatus.PASSED if call.success else LiveTestStatus.FAILED,
            duration_ms=call.duration_ms,
        )
