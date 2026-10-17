"""Test execution capability running real unittest or pytest suites and capturing results."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from typing import Any, Dict, List

from agent911.capabilities.base import Capability, CapabilityContext
from agent911.capabilities.filesystem import WorkspaceBoundaryError, resolve_in_workspace
from agent911.models import CapabilityResult


class RunTestsCapability(Capability):
    identifier = "test.run"
    name = "Run Automated Tests"
    purpose = (
        "Execute automated tests using unittest or pytest in the workspace and capture "
        "actual test counts, exit status, stdout, and stderr."
    )
    input_schema = {
        "required": [],
        "properties": {
            "target": {
                "type": "string",
                "description": "Test directory or test file path relative to workspace (default 'tests')",
            },
            "runner": {
                "type": "string",
                "description": "Test runner to use: 'unittest' or 'pytest' (default 'unittest')",
            },
            "pattern": {
                "type": "string",
                "description": "Test file discovery pattern for unittest (default 'test*.py')",
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Timeout for test run in seconds",
            },
        },
    }
    output_format = {
        "runner": "str",
        "target": "str",
        "exit_code": "int | None",
        "tests_ran": "int",
        "passed": "bool",
        "stdout": "str",
        "stderr": "str",
        "timed_out": "bool",
        "duration_ms": "float",
    }

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        target_rel = params.get("target", "tests")
        runner = str(params.get("runner", "unittest")).lower()
        pattern = str(params.get("pattern", "test*.py"))
        timeout = float(params.get("timeout_seconds", context.config.command_timeout_seconds))

        try:
            target_path = resolve_in_workspace(context.workspace_path, target_rel)
        except (WorkspaceBoundaryError, ValueError) as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

        if not target_path.exists():
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Test target path does not exist: '{target_rel}'",
                error_type="FileNotFoundError",
            )

        ws = context.workspace_path.resolve()
        rel_target_str = str(target_path.relative_to(ws)) if target_path != ws else "."

        if runner == "pytest":
            cmd: List[str] = [sys.executable, "-m", "pytest", rel_target_str, "-q"]
        elif runner == "unittest":
            if target_path.is_file():
                cmd = [sys.executable, "-m", "unittest", rel_target_str, "-v"]
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    rel_target_str,
                    "-p",
                    pattern,
                    "-v",
                ]
        else:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Unsupported test runner '{runner}'. Choose 'unittest' or 'pytest'.",
                error_type="InputValidationError",
            )

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ws),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            combined = f"{stdout}\n{stderr}"
            tests_ran = self._parse_tests_ran(combined, runner)
            passed = proc.returncode == 0 and tests_ran > 0

            data = {
                "runner": runner,
                "target": rel_target_str,
                "command": cmd,
                "exit_code": proc.returncode,
                "tests_ran": tests_ran,
                "passed": passed,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": False,
                "duration_ms": elapsed_ms,
            }
            if passed:
                return CapabilityResult(
                    capability_id=self.identifier,
                    success=True,
                    data=data,
                    metadata={"duration_ms": elapsed_ms, "tests_ran": tests_ran},
                )
            reason = (
                "No tests were discovered or executed."
                if proc.returncode == 0 and tests_ran == 0
                else f"Test suite failed with exit code {proc.returncode}."
            )
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                data=data,
                error=f"{reason} Output: {combined.strip()[-400:]}",
                error_type="TestFailureError",
                metadata={"duration_ms": elapsed_ms, "tests_ran": tests_ran},
            )
        except subprocess.TimeoutExpired as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                data={
                    "runner": runner,
                    "target": rel_target_str,
                    "command": cmd,
                    "exit_code": None,
                    "tests_ran": 0,
                    "passed": False,
                    "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
                    "stderr": (exc.stderr or "") if isinstance(exc.stderr, str) else "",
                    "timed_out": True,
                    "duration_ms": elapsed_ms,
                },
                error=f"Test execution timed out after {timeout}s.",
                error_type="TimeoutError",
                metadata={"duration_ms": elapsed_ms, "timed_out": True},
            )

    @staticmethod
    def _parse_tests_ran(output: str, runner: str) -> int:
        if runner == "unittest":
            m = re.search(r"Ran\s+(\d+)\s+test", output)
            if m:
                return int(m.group(1))
        # Also check pytest summary patterns like "5 passed" or "2 failed, 3 passed"
        counts = re.findall(r"(\d+)\s+(?:passed|failed|error|errors)", output)
        if counts:
            return sum(int(c) for c in counts)
        return 0
