"""Filesystem capabilities enforcing workspace boundary and real I/O validation."""

from __future__ import annotations

import hashlib
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agent911.capabilities.base import Capability, CapabilityContext
from agent911.models import CapabilityResult


class WorkspaceBoundaryError(PermissionError):
    """Raised when a filesystem operation attempts to access a path outside workspace_root."""


def resolve_in_workspace(workspace_root: Path, requested_path: str) -> Path:
    """
    Resolve a relative or absolute path and verify that it remains inside workspace_root.
    Raises WorkspaceBoundaryError if path escapes the configured workspace.
    """
    if not requested_path or not isinstance(requested_path, str):
        raise ValueError("Path must be a non-empty string.")
    ws = workspace_root.resolve()
    raw = Path(requested_path)
    candidate = raw.resolve() if raw.is_absolute() else (ws / raw).resolve()
    if not candidate.is_relative_to(ws):
        raise WorkspaceBoundaryError(
            f"Path '{requested_path}' resolves to '{candidate}', which is outside the "
            f"configured workspace boundary '{ws}'."
        )
    return candidate


class ListDirectoryCapability(Capability):
    identifier = "fs.list"
    name = "List Directory"
    purpose = "List files and directories inside a workspace directory."
    input_schema = {
        "required": [],
        "properties": {
            "path": {"type": "string", "description": "Relative path within workspace (default '.')"},
            "recursive": {"type": "boolean", "description": "Whether to list recursively"},
            "include_hidden": {"type": "boolean", "description": "Whether to include hidden files"},
        },
    }
    output_format = {"entries": "list of file/directory metadata objects"}

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        rel_path = params.get("path", ".")
        recursive = bool(params.get("recursive", False))
        include_hidden = bool(params.get("include_hidden", False))

        try:
            target = resolve_in_workspace(context.workspace_path, rel_path)
        except (WorkspaceBoundaryError, ValueError) as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

        if not target.exists():
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Directory not found: '{rel_path}'",
                error_type="FileNotFoundError",
            )
        if not target.is_dir():
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Path is not a directory: '{rel_path}'",
                error_type="NotADirectoryError",
            )

        entries: List[Dict[str, Any]] = []
        iterator = target.rglob("*") if recursive else target.iterdir()
        ws = context.workspace_path.resolve()
        for item in sorted(iterator, key=lambda p: str(p)):
            rel_parts = item.relative_to(ws).parts
            if not include_hidden and any(p.startswith(".") for p in rel_parts):
                continue
            try:
                st = item.stat()
                entries.append(
                    {
                        "name": item.name,
                        "path": str(item.relative_to(ws)),
                        "type": "directory" if item.is_dir() else "file",
                        "size_bytes": st.st_size if item.is_file() else 0,
                    }
                )
            except OSError:
                continue

        return CapabilityResult(
            capability_id=self.identifier,
            success=True,
            data={"path": str(target.relative_to(ws)), "count": len(entries), "entries": entries},
        )


class ReadFileCapability(Capability):
    identifier = "fs.read"
    name = "Read File"
    purpose = "Read text content of a file inside the configured workspace."
    input_schema = {
        "required": ["path"],
        "properties": {
            "path": {"type": "string", "description": "Relative or workspace path to file"},
            "max_bytes": {"type": "integer", "description": "Maximum bytes to read"},
        },
    }
    output_format = {"path": "str", "content": "str", "size_bytes": "int"}

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        req_path = params["path"]
        max_bytes = int(params.get("max_bytes", 500_000))
        try:
            target = resolve_in_workspace(context.workspace_path, req_path)
        except (WorkspaceBoundaryError, ValueError) as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

        if not target.exists():
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"File does not exist: '{req_path}'",
                error_type="FileNotFoundError",
            )
        if not target.is_file():
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Path is not a regular file: '{req_path}'",
                error_type="IsADirectoryError",
            )

        try:
            raw = target.read_bytes()
            truncated = len(raw) > max_bytes
            content = raw[:max_bytes].decode("utf-8", errors="replace")
            ws = context.workspace_path.resolve()
            return CapabilityResult(
                capability_id=self.identifier,
                success=True,
                data={
                    "path": str(target.relative_to(ws)),
                    "content": content,
                    "size_bytes": len(raw),
                    "truncated": truncated,
                },
            )
        except OSError as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Failed reading file '{req_path}': {exc}",
                error_type=type(exc).__name__,
            )


class WriteFileCapability(Capability):
    identifier = "fs.write"
    name = "Write File"
    purpose = "Create or overwrite a file within the configured workspace."
    input_schema = {
        "required": ["path", "content"],
        "properties": {
            "path": {"type": "string", "description": "Path inside workspace"},
            "content": {"type": "string", "description": "Text content to write"},
            "create_parents": {
                "type": "boolean",
                "description": "Create missing parent directories automatically (default False)",
            },
        },
    }
    output_format = {"path": "str", "bytes_written": "int", "sha256": "str"}

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        req_path = params["path"]
        content = params["content"]
        create_parents = bool(params.get("create_parents", False))

        try:
            target = resolve_in_workspace(context.workspace_path, req_path)
        except (WorkspaceBoundaryError, ValueError) as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

        ws = context.workspace_path.resolve()
        if not target.parent.exists():
            if create_parents:
                target.parent.mkdir(parents=True, exist_ok=True)
            else:
                rel_parent = str(target.parent.relative_to(ws))
                return CapabilityResult(
                    capability_id=self.identifier,
                    success=False,
                    error=f"Parent directory '{rel_parent}' does not exist for file '{req_path}'.",
                    error_type="ParentDirectoryNotFoundError",
                    metadata={"missing_parent": rel_parent, "target_path": req_path},
                )

        if target.exists() and target.is_dir():
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Cannot write file because '{req_path}' is an existing directory.",
                error_type="IsADirectoryError",
            )

        try:
            encoded = content.encode("utf-8")
            target.write_bytes(encoded)
            digest = hashlib.sha256(encoded).hexdigest()
            return CapabilityResult(
                capability_id=self.identifier,
                success=True,
                data={
                    "path": str(target.relative_to(ws)),
                    "bytes_written": len(encoded),
                    "sha256": digest,
                },
            )
        except OSError as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Failed writing file '{req_path}': {exc}",
                error_type=type(exc).__name__,
            )


class UpdateFileCapability(Capability):
    identifier = "fs.update"
    name = "Update File"
    purpose = "Update an existing file in the workspace by replacing text or appending content."
    input_schema = {
        "required": ["path"],
        "properties": {
            "path": {"type": "string", "description": "File path inside workspace"},
            "old_text": {"type": "string", "description": "Substring to find and replace"},
            "new_text": {"type": "string", "description": "Replacement text"},
            "append_text": {"type": "string", "description": "Text to append to end of file"},
        },
    }
    output_format = {"path": "str", "bytes_written": "int", "sha256": "str"}

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        req_path = params["path"]
        old_text = params.get("old_text")
        new_text = params.get("new_text")
        append_text = params.get("append_text")

        try:
            target = resolve_in_workspace(context.workspace_path, req_path)
        except (WorkspaceBoundaryError, ValueError) as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

        if not target.exists() or not target.is_file():
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Cannot update non-existent file: '{req_path}'",
                error_type="FileNotFoundError",
            )

        try:
            current = target.read_text(encoding="utf-8")
            if old_text is not None:
                if new_text is None:
                    return CapabilityResult(
                        capability_id=self.identifier,
                        success=False,
                        error="Parameter 'new_text' is required when 'old_text' is provided.",
                        error_type="InputValidationError",
                    )
                if old_text not in current:
                    return CapabilityResult(
                        capability_id=self.identifier,
                        success=False,
                        error=f"Target substring 'old_text' not found in '{req_path}'.",
                        error_type="ValueError",
                    )
                updated = current.replace(old_text, new_text, 1)
            elif append_text is not None:
                updated = current + append_text
            else:
                return CapabilityResult(
                    capability_id=self.identifier,
                    success=False,
                    error="Either ('old_text', 'new_text') or 'append_text' must be provided.",
                    error_type="InputValidationError",
                )

            encoded = updated.encode("utf-8")
            target.write_bytes(encoded)
            ws = context.workspace_path.resolve()
            return CapabilityResult(
                capability_id=self.identifier,
                success=True,
                data={
                    "path": str(target.relative_to(ws)),
                    "bytes_written": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                },
            )
        except OSError as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Failed updating file '{req_path}': {exc}",
                error_type=type(exc).__name__,
            )


class MakeDirectoryCapability(Capability):
    identifier = "fs.mkdir"
    name = "Create Directory"
    purpose = "Create a directory (and parent directories if needed) inside the workspace."
    input_schema = {
        "required": ["path"],
        "properties": {
            "path": {"type": "string", "description": "Directory path to create inside workspace"},
            "parents": {"type": "boolean", "description": "Whether to create parent directories"},
        },
    }
    output_format = {"path": "str", "created": "bool"}

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        req_path = params["path"]
        parents = bool(params.get("parents", True))
        try:
            target = resolve_in_workspace(context.workspace_path, req_path)
        except (WorkspaceBoundaryError, ValueError) as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

        if target.exists() and not target.is_dir():
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Path '{req_path}' already exists and is a file, not a directory.",
                error_type="FileExistsError",
            )

        try:
            target.mkdir(parents=parents, exist_ok=True)
            ws = context.workspace_path.resolve()
            return CapabilityResult(
                capability_id=self.identifier,
                success=True,
                data={"path": str(target.relative_to(ws)), "created": True},
            )
        except OSError as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Failed creating directory '{req_path}': {exc}",
                error_type=type(exc).__name__,
            )


class StatFileCapability(Capability):
    identifier = "fs.stat"
    name = "Inspect File Metadata"
    purpose = "Inspect metadata (size, permissions, timestamps, sha256) of a workspace path."
    input_schema = {
        "required": ["path"],
        "properties": {
            "path": {"type": "string", "description": "File or directory path inside workspace"},
        },
    }
    output_format = {
        "path": "str",
        "exists": "bool",
        "is_file": "bool",
        "is_dir": "bool",
        "size_bytes": "int",
        "mode_octal": "str",
        "modified_at": "str",
        "sha256": "optional str",
    }

    def execute(self, params: Dict[str, Any], context: CapabilityContext) -> CapabilityResult:
        req_path = params["path"]
        try:
            target = resolve_in_workspace(context.workspace_path, req_path)
        except (WorkspaceBoundaryError, ValueError) as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

        if not target.exists():
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Path does not exist: '{req_path}'",
                error_type="FileNotFoundError",
                data={"path": req_path, "exists": False},
            )

        try:
            st = target.stat()
            ws = context.workspace_path.resolve()
            sha = None
            if target.is_file() and st.st_size <= 5_000_000:
                sha = hashlib.sha256(target.read_bytes()).hexdigest()
            return CapabilityResult(
                capability_id=self.identifier,
                success=True,
                data={
                    "path": str(target.relative_to(ws)),
                    "exists": True,
                    "is_file": target.is_file(),
                    "is_dir": target.is_dir(),
                    "size_bytes": st.st_size,
                    "mode_octal": oct(stat.S_IMODE(st.st_mode)),
                    "modified_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                    "sha256": sha,
                },
            )
        except OSError as exc:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error=f"Failed inspecting metadata for '{req_path}': {exc}",
                error_type=type(exc).__name__,
            )
