"""Project inspection capability for discovering structure, manifests, entry points, and tests."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from agent911.capabilities.base import Capability, CapabilityContext
from agent911.capabilities.filesystem import WorkspaceBoundaryError, resolve_in_workspace
from agent911.models import CapabilityResult


_IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".agent911",
}

_CONFIG_FILENAMES = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "package.json",
    "tsconfig.json",
    "Makefile",
    "Dockerfile",
    "agent911.config.json",
    "agent911.config.example.json",
    "agent911.config.local.json",
}

_MANIFEST_FILENAMES = {
    "pyproject.toml",
    "setup.py",
    "requirements.txt",
    "package.json",
    "Cargo.toml",
    "go.mod",
}


class InspectProjectCapability(Capability):
    identifier = "project.inspect"
    name = "Inspect Software Project"
    purpose = (
        "Inspect a project directory to discover project files, source directories, "
        "configuration files, documentation, tests, package manifests, entry points, "
        "and dependencies."
    )
    input_schema = {
        "required": [],
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative directory inside workspace to inspect (default '.')",
            },
            "include_ast_symbols": {
                "type": "boolean",
                "description": "Whether to parse Python AST classes and functions",
            },
        },
    }
    output_format = {
        "root": "str",
        "project_files": "list[str]",
        "source_directories": "list[str]",
        "configuration_files": "list[str]",
        "documentation_files": "list[str]",
        "test_files": "list[str]",
        "package_manifests": "list[str]",
        "entry_points": "list[str]",
        "dependencies": "list[str]",
        "python_symbols": "dict[str, dict]",
    }

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        rel_path = params.get("path", ".")
        include_ast = bool(params.get("include_ast_symbols", True))

        try:
            root = resolve_in_workspace(context.workspace_path, rel_path)
        except (WorkspaceBoundaryError, ValueError) as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

        if not root.exists() or not root.is_dir():
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Project directory does not exist: '{rel_path}'",
                error_type="FileNotFoundError",
            )

        project_files: List[str] = []
        source_dirs_set: set[str] = set()
        config_files: List[str] = []
        doc_files: List[str] = []
        test_files: List[str] = []
        manifests: List[str] = []
        entry_points: List[str] = []
        dependencies: List[str] = []
        python_symbols: Dict[str, Dict[str, List[str]]] = {}

        for item in sorted(root.rglob("*"), key=lambda p: str(p)):
            rel = item.relative_to(root)
            if any(part in _IGNORED_DIRS or part.endswith(".egg-info") for part in rel.parts):
                continue

            rel_str = str(rel)
            if item.is_dir():
                if (item / "__init__.py").exists() or item.name in ("src", "lib", "app"):
                    source_dirs_set.add(rel_str)
                continue

            project_files.append(rel_str)
            fname = item.name

            # Source directories inferred from code files
            if item.suffix in (".py", ".ts", ".js", ".go", ".rs") and len(rel.parts) > 1:
                top_dir = rel.parts[0]
                if top_dir not in ("tests", "test", "docs", "doc"):
                    source_dirs_set.add(top_dir)

            # Configuration files
            if fname in _CONFIG_FILENAMES or item.suffix in (".toml", ".yaml", ".yml", ".ini", ".cfg"):
                config_files.append(rel_str)

            # Package manifests
            if fname in _MANIFEST_FILENAMES:
                manifests.append(rel_str)
                dependencies.extend(self._extract_dependencies(item))

            # Documentation
            if (
                fname.lower().startswith(("readme", "changelog", "contributing", "license"))
                or item.suffix.lower() in (".md", ".rst")
                or (len(rel.parts) > 1 and rel.parts[0].lower() in ("docs", "doc"))
            ):
                doc_files.append(rel_str)

            # Tests
            if (
                fname.startswith("test_")
                or fname.endswith("_test.py")
                or fname.endswith(".test.js")
                or fname.endswith(".spec.js")
                or (len(rel.parts) > 1 and rel.parts[0].lower() in ("tests", "test"))
            ):
                if item.suffix in (".py", ".js", ".ts"):
                    test_files.append(rel_str)

            # Entry points
            if fname in ("__main__.py", "main.py", "cli.py", "app.py", "manage.py", "index.js"):
                entry_points.append(rel_str)

            if item.suffix == ".py" and item.stat().st_size <= 250_000:
                try:
                    src = item.read_text(encoding="utf-8", errors="replace")
                    if 'if __name__ == "__main__"' in src or "if __name__ == '__main__'" in src:
                        if rel_str not in entry_points:
                            entry_points.append(rel_str)
                    if include_ast and len(python_symbols) < 50:
                        tree = ast.parse(src, filename=rel_str)
                        classes = [
                            node.name for node in tree.body if isinstance(node, ast.ClassDef)
                        ]
                        functions = [
                            node.name
                            for node in tree.body
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        ]
                        if classes or functions:
                            python_symbols[rel_str] = {
                                "classes": classes,
                                "functions": functions,
                            }
                except Exception:
                    pass

        ws = context.workspace_path.resolve()
        return CapabilityResult(
            capability_id=self.identifier,
            success=True,
            data={
                "root": str(root.relative_to(ws)) if root != ws else ".",
                "project_files": project_files,
                "source_directories": sorted(source_dirs_set),
                "configuration_files": config_files,
                "documentation_files": doc_files,
                "test_files": test_files,
                "package_manifests": manifests,
                "entry_points": entry_points,
                "dependencies": sorted(set(dependencies)),
                "python_symbols": python_symbols,
            },
        )

    @staticmethod
    def _extract_dependencies(manifest_path: Path) -> List[str]:
        deps: List[str] = []
        try:
            text = manifest_path.read_text(encoding="utf-8", errors="replace")
            if manifest_path.name == "requirements.txt":
                for line in text.splitlines():
                    s = line.strip()
                    if s and not s.startswith("#"):
                        deps.append(s)
            elif manifest_path.name == "package.json":
                pkg = json.loads(text)
                for section in ("dependencies", "devDependencies"):
                    if isinstance(pkg.get(section), dict):
                        deps.extend(pkg[section].keys())
            elif manifest_path.name == "pyproject.toml":
                # Extract dependencies list items from pyproject.toml
                for match in re.findall(r'"([a-zA-Z0-9_\-]+(?:[><=~!][^\"]+)?)"', text):
                    if not match.startswith("setuptools") and match not in ("MIT", "agent911"):
                        if any(op in match for op in (">=", "==", "<=", "~=", ">", "<")):
                            deps.append(match)
        except Exception:
            pass
        return deps
