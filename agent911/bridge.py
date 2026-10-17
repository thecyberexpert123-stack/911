"""JSON bridge connecting the Node.js Chat UI server to the Python Agent911Orchestrator."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from agent911.config import AgentConfig, ConfigurationError
from agent911.models import MemoryCategory, MemoryConfidence
from agent911.orchestrator import Agent911Orchestrator


def handle_request(req: Dict[str, Any]) -> Dict[str, Any]:
    action = req.get("action")
    workspace = req.get("workspace")
    config_path = req.get("config_path")

    try:
        config = AgentConfig.load(
            config_path=config_path,
            workspace_override=workspace,
        )
        orchestrator = Agent911Orchestrator(config=config)
    except ConfigurationError as exc:
        return {"ok": False, "error": str(exc), "error_type": "ConfigurationError"}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}

    try:
        if action == "status":
            conn = orchestrator.model_interface.check_connectivity()
            caps = orchestrator.registry.list_capabilities()
            tasks = orchestrator.memory_store.list_tasks(limit=10)
            memories = orchestrator.memory_store.retrieve_memories(limit=100)
            experiences = orchestrator.memory_store.retrieve_experiences(limit=100)
            proposals = orchestrator.list_improvement_proposals()
            improvement_runs = orchestrator.list_self_improvement_runs(limit=20)
            return {
                "ok": True,
                "data": {
                    "configuration": config.to_safe_dict(),
                    "model_connectivity": conn.to_dict(),
                    "capabilities_registered": len(caps),
                    "capabilities_available": sum(
                        1 for c in caps if c["availability"] == "AVAILABLE"
                    ),
                    "persisted_memories_count": len(memories),
                    "persisted_experiences_count": len(experiences),
                    "proposals_count": len(proposals),
                    "self_improvement_runs_count": len(improvement_runs),
                    "recent_tasks": tasks,
                },
            }

        if action == "update_config":
            updates = req.get("updates") or {}
            if "model" in updates and isinstance(updates["model"], dict):
                m_up = updates["model"]
                active_prof = config.models.get(config.active_model)
                active_prov = (
                    config.providers.get(active_prof.provider_id)
                    if active_prof
                    else None
                )
                if active_prov:
                    if m_up.get("provider"):
                        active_prov.adapter_type = str(m_up["provider"])
                    if m_up.get("endpoint"):
                        active_prov.endpoint = str(m_up["endpoint"])
                    if m_up.get("api_key_env"):
                        active_prov.api_key_env = str(m_up["api_key_env"])
                if active_prof:
                    if m_up.get("model_id"):
                        active_prof.model_id = str(m_up["model_id"])
                    if m_up.get("temperature") is not None:
                        active_prof.temperature = float(m_up["temperature"])
                    if m_up.get("max_tokens") is not None:
                        active_prof.max_tokens = int(m_up["max_tokens"])
                    if m_up.get("timeout_seconds") is not None:
                        active_prof.timeout_seconds = float(m_up["timeout_seconds"])
                    if m_up.get("max_retries") is not None:
                        active_prof.max_retries = int(m_up["max_retries"])

            for k in (
                "max_recovery_attempts_per_step",
                "command_timeout_seconds",
                "max_context_chars",
            ):
                if k in updates and updates[k] is not None:
                    setattr(config, k, updates[k])

            config._sync_effective_model()
            config.validate()
            config.save()
            updated_orch = Agent911Orchestrator(config=config)
            conn = updated_orch.model_interface.check_connectivity()
            return {
                "ok": True,
                "data": {
                    "configuration": config.to_safe_dict(),
                    "model_connectivity": conn.to_dict(),
                },
            }

        if action == "list_providers":
            return {
                "ok": True,
                "data": {"providers": orchestrator.list_providers()},
            }

        if action == "upsert_provider":
            pdata = req.get("provider") or {}
            api_key = req.get("api_key")
            saved = orchestrator.upsert_provider(pdata, api_key=api_key, save=True)
            return {
                "ok": True,
                "data": {
                    "provider": saved,
                    "providers": orchestrator.list_providers(),
                    "configuration": config.to_safe_dict(),
                },
            }

        if action == "remove_provider":
            pid = str(req.get("provider_id", "")).strip()
            orchestrator.remove_provider(pid, save=True)
            return {
                "ok": True,
                "data": {
                    "removed": pid,
                    "providers": orchestrator.list_providers(),
                    "models": orchestrator.list_models(),
                },
            }

        if action == "list_models":
            return {
                "ok": True,
                "data": {
                    "active_model": config.active_model,
                    "default_model": config.default_model,
                    "fallback_model": config.fallback_model,
                    "models": orchestrator.list_models(),
                },
            }

        if action == "upsert_model":
            mdata = req.get("model") or {}
            saved = orchestrator.upsert_model(mdata, save=True)
            return {
                "ok": True,
                "data": {
                    "model": saved,
                    "models": orchestrator.list_models(),
                    "configuration": config.to_safe_dict(),
                },
            }

        if action == "remove_model":
            mid = str(req.get("profile_id", "")).strip()
            orchestrator.remove_model(mid, save=True)
            return {
                "ok": True,
                "data": {
                    "removed": mid,
                    "models": orchestrator.list_models(),
                },
            }

        if action == "select_model":
            mid = req.get("profile_id")
            set_default = bool(req.get("set_as_default", False))
            fallback_id = req.get("fallback_model")
            res: Dict[str, Any] = {}
            if mid:
                res = orchestrator.select_active_model(
                    str(mid), set_as_default=set_default, save=True
                )
            if "fallback_model" in req:
                fb_val = (
                    None
                    if fallback_id is None or str(fallback_id).lower() in ("", "none")
                    else str(fallback_id)
                )
                res = orchestrator.set_fallback_model(fb_val, save=True)
            return {
                "ok": True,
                "data": {
                    "selection": res,
                    "models": orchestrator.list_models(),
                    "configuration": config.to_safe_dict(),
                },
            }

        if action == "test_model_connection":
            prof_id = req.get("profile_id")
            report = orchestrator.test_model_connection(
                profile_id=str(prof_id) if prof_id else None
            )
            return {"ok": True, "data": report.to_dict()}

        if action == "distill":
            raw_text = str(req.get("instruction") or req.get("text") or "").strip()
            if not raw_text:
                return {
                    "ok": False,
                    "error": "Instruction text cannot be empty.",
                    "error_type": "ValueError",
                }
            spec_dict = orchestrator.distill_instruction(raw_text)
            return {"ok": True, "data": spec_dict}

        if action == "chat":
            instruction = str(req.get("instruction", "")).strip()
            if not instruction:
                return {
                    "ok": False,
                    "error": "Instruction cannot be empty.",
                    "error_type": "ValueError",
                }
            explicit_plan = req.get("explicit_plan")
            explicit_requirements = req.get("explicit_requirements")
            task = orchestrator.execute_task(
                instruction=instruction,
                explicit_plan=explicit_plan,
                explicit_requirements=explicit_requirements,
            )
            task_dict = task.to_dict()
            task_dict["execution_logs"] = orchestrator.memory_store.get_execution_logs(
                task.task_id
            )
            return {"ok": True, "data": task_dict}

        if action == "list_tasks":
            limit = int(req.get("limit", 30))
            tasks = orchestrator.memory_store.list_tasks(limit=limit)
            return {"ok": True, "data": {"count": len(tasks), "tasks": tasks}}

        if action == "inspect_task":
            task_id = str(req.get("task_id", "")).strip()
            task = orchestrator.memory_store.load_task(task_id)
            if task is None:
                return {
                    "ok": False,
                    "error": f"Task '{task_id}' not found.",
                    "error_type": "KeyError",
                }
            task_dict = task.to_dict()
            task_dict["execution_logs"] = orchestrator.memory_store.get_execution_logs(
                task_id
            )
            return {"ok": True, "data": task_dict}

        if action == "resume_task":
            task_id = str(req.get("task_id", "")).strip()
            task = orchestrator.resume_task(task_id)
            task_dict = task.to_dict()
            task_dict["execution_logs"] = orchestrator.memory_store.get_execution_logs(
                task.task_id
            )
            return {"ok": True, "data": task_dict}

        if action == "list_capabilities":
            only_available = bool(req.get("only_available", False))
            caps = orchestrator.registry.list_capabilities(only_available=only_available)
            return {"ok": True, "data": {"count": len(caps), "capabilities": caps}}

        if action == "develop_capability":
            cap_id = str(req.get("capability_id", "")).strip()
            name = str(req.get("name", "")).strip()
            purpose = str(req.get("purpose", "")).strip()
            implementation_code = req.get("implementation_code")
            if not cap_id or not name or not purpose:
                return {
                    "ok": False,
                    "error": "capability_id, name, and purpose are required.",
                    "error_type": "InputValidationError",
                }
            dev_params: Dict[str, Any] = {
                "capability_id": cap_id,
                "name": name,
                "purpose": purpose,
                "behavior_spec": purpose,
                "overwrite": bool(req.get("overwrite", True)),
            }
            if implementation_code and str(implementation_code).strip():
                dev_params["implementation_code"] = str(implementation_code)

            task = orchestrator.execute_task(
                f"Develop capability '{cap_id}': {purpose}",
                explicit_plan=[
                    {
                        "step_id": "step-1",
                        "description": f"Develop, verify, and live-test capability '{cap_id}'",
                        "capability_id": "capability.develop",
                        "parameters": dev_params,
                    }
                ],
                explicit_requirements=[
                    {
                        "requirement_id": f"req-cap-{cap_id}",
                        "description": f"Capability '{cap_id}' is operational",
                        "category": "capability_operational",
                        "target": cap_id,
                    }
                ],
            )
            task_dict = task.to_dict()
            task_dict["execution_logs"] = orchestrator.memory_store.get_execution_logs(
                task.task_id
            )
            return {"ok": True, "data": task_dict}

        if action == "list_memories":
            query = req.get("query")
            category = req.get("category")
            limit = int(req.get("limit", 30))
            records = orchestrator.memory_store.retrieve_memories(
                query=query,
                category=category if category else None,
                limit=limit,
            )
            return {
                "ok": True,
                "data": {
                    "count": len(records),
                    "records": [r.to_dict() for r in records],
                },
            }

        if action == "store_memory":
            key = str(req.get("key", "")).strip()
            content = str(req.get("content", "")).strip()
            category = str(req.get("category", "FACTUAL")).upper()
            confidence = str(req.get("confidence", "OBSERVED")).upper()
            if not key or not content:
                return {
                    "ok": False,
                    "error": "Both 'key' and 'content' are required to store memory.",
                    "error_type": "InputValidationError",
                }
            rec = orchestrator.memory_store.store_memory(
                category=MemoryCategory(category),
                key=key,
                content=content,
                source="chat_ui",
                confidence=MemoryConfidence(confidence),
            )
            return {"ok": True, "data": rec.to_dict()}

        if action == "list_experiences":
            query = req.get("query")
            limit = int(req.get("limit", 20))
            exps = orchestrator.memory_store.retrieve_experiences(
                query=query, limit=limit
            )
            return {
                "ok": True,
                "data": {
                    "count": len(exps),
                    "experiences": [e.to_dict() for e in exps],
                },
            }

        if action == "list_proposals":
            status = req.get("status")
            proposals = orchestrator.list_improvement_proposals(status=status)
            runs = orchestrator.list_self_improvement_runs(limit=20)
            return {
                "ok": True,
                "data": {
                    "count": len(proposals),
                    "proposals": proposals,
                    "self_improvement_runs": runs,
                },
            }

        if action == "detect_proposals":
            detected = orchestrator.detect_and_propose_improvements()
            proposals = orchestrator.list_improvement_proposals()
            return {
                "ok": True,
                "data": {
                    "newly_detected_count": len(detected),
                    "newly_detected": detected,
                    "proposals": proposals,
                },
            }

        if action == "self_improve":
            prop_id = str(req.get("proposal_id", "")).strip()
            if not prop_id:
                return {
                    "ok": False,
                    "error": "Field 'proposal_id' is required.",
                    "error_type": "InputValidationError",
                }
            run_rec = orchestrator.run_controlled_self_improvement(prop_id)
            return {
                "ok": True,
                "data": {
                    "run": run_rec,
                    "proposals": orchestrator.list_improvement_proposals(),
                },
            }

        if action == "diagnose_task":
            task_id = str(req.get("task_id", "")).strip()
            if not task_id:
                return {
                    "ok": False,
                    "error": "Field 'task_id' is required.",
                    "error_type": "InputValidationError",
                }
            diag = orchestrator.diagnose_task(task_id)
            return {"ok": True, "data": diag}

        return {
            "ok": False,
            "error": f"Unknown bridge action '{action}'.",
            "error_type": "ValueError",
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"ok": False, "error": "Empty request payload"}))
        return 1
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "error": f"Invalid JSON: {exc}"}))
        return 1
    resp = handle_request(req)
    print(json.dumps(resp))
    return 0 if resp.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
