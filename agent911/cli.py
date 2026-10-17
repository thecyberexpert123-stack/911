"""Command-line interface for configuring, operating, and inspecting Agent 911."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from agent911.config import AgentConfig, ConfigurationError
from agent911.models import TaskOutcome, TaskState
from agent911.orchestrator import Agent911Orchestrator


def _format_task_report(task: TaskState) -> str:
    lines: List[str] = []
    lines.append(f"=== AGENT 911 TASK EXECUTION REPORT [{task.task_id}] ===")
    lines.append(f"Original Instruction : {task.original_instruction}")
    lines.append(f"Interpreted Objective: {task.interpreted_objective}")
    lines.append(f"Lifecycle Stage      : {task.current_execution_state.value}")
    lines.append(f"Interaction Mode     : {task.interaction_mode}")
    lines.append(f"Final Outcome        : {task.final_outcome.value}")
    lines.append(f"Outcome Summary      : {task.outcome_summary}")
    if task.distilled_spec and task.distilled_spec.is_build_intent:
        lines.append("\n--- Distilled Build Specification (From User Writing) ---")
        lines.append(f"  What User Wants to Build : {task.distilled_spec.what_user_wants_to_build}")
        lines.append(f"  Messy Input Distilled    : {task.distilled_spec.is_messy_input}")
        lines.append(f"  Concrete Instruction     : {task.distilled_spec.concrete_instruction}")
        if task.distilled_spec.extracted_functions:
            lines.append(f"  Identified Functions     : {', '.join(task.distilled_spec.extracted_functions)}")
        if task.distilled_spec.ambiguities_resolved:
            lines.append(f"  Ambiguities Resolved     : {'; '.join(task.distilled_spec.ambiguities_resolved)}")
    if task.assistant_response:
        lines.append(f"\n--- Agent 911 Conversational Response ---\n{task.assistant_response}")

    if task.decisions:
        lines.append("\n--- Key Decisions & Experience Applied ---")
        for dec in task.decisions:
            lines.append(f"  * {dec}")

    lines.append(f"\n--- Planned Actions ({len(task.planned_actions)}) ---")
    for act in task.planned_actions:
        lines.append(
            f"  [{act.status.value}] {act.step_id}: {act.capability_id} - {act.description}"
        )
        if act.output_summary:
            lines.append(f"      Output: {act.output_summary}")
        if act.error:
            lines.append(f"      Error : {act.error}")

    if task.errors:
        lines.append(f"\n--- Errors & Recovery ({len(task.errors)} error(s)) ---")
        for err in task.errors:
            rec_tag = "RECOVERED" if err.recovered else "UNRECOVERED"
            lines.append(
                f"  [{rec_tag}] step={err.step_id} ({err.error_type}): {err.message}"
            )
        for att in task.recovery_attempts:
            status_tag = "SUCCEEDED" if att.succeeded else "FAILED"
            lines.append(
                f"  Recovery Attempt [{status_tag}] step={att.step_id}: "
                f"cause='{att.probable_cause}' -> correction='{att.selected_correction}'"
            )

    lines.append(
        f"\n--- Formal Evaluation Mechanism 1: Verification ({len(task.verification_results)}) ---"
    )
    for v in task.verification_results:
        lines.append(
            f"  [{v.status.value}] {v.requirement} (method={v.method}): {v.observed_result}"
        )
        if v.failure_explanation:
            lines.append(f"      Explanation: {v.failure_explanation}")

    lines.append(
        f"\n--- Formal Evaluation Mechanism 2: Live Run Testing ({len(task.live_run_results)}) ---"
    )
    for l in task.live_run_results:
        lines.append(
            f"  [{l.status.value}] target={l.target_component} | op={l.executed_operation} "
            f"| exit_code={l.exit_code} | expected_occurred={l.expected_behavior_occurred}"
        )
        if l.stdout:
            lines.append(f"      stdout: {l.stdout.strip()[:200]}")
        if l.stderr:
            lines.append(f"      stderr: {l.stderr.strip()[:200]}")
        if l.limitations:
            lines.append(f"      limitations: {l.limitations}")

    lines.append(f"\nAssociated Experience IDs: {task.associated_experience_ids}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent911",
        description="Agent 911: Instruction-Driven, Self-Expanding AI Agent",
    )
    parser.add_argument("--config", help="Path to configuration JSON file", default=None)
    parser.add_argument("--workspace", help="Override workspace root directory", default=None)
    parser.add_argument("--provider", help="Override AI model provider", default=None)
    parser.add_argument("--model-id", help="Override AI model identifier", default=None)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output")

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # status
    subparsers.add_parser(
        "status",
        help="Show real application configuration, model connectivity, and memory status",
    )

    # capabilities
    cap_parser = subparsers.add_parser(
        "capabilities",
        help="List all registered capabilities and their operational availability",
    )
    cap_parser.add_argument(
        "--only-available",
        action="store_true",
        help="Filter to only currently AVAILABLE capabilities",
    )

    # run
    run_parser = subparsers.add_parser(
        "run",
        help="Execute a natural-language instruction through the full Agent 911 lifecycle",
    )
    run_parser.add_argument("instruction", help="Natural-language instruction to execute")

    # chat
    chat_parser = subparsers.add_parser(
        "chat",
        help="Talk conversationally with Agent 911 or execute instructions in single-turn or interactive REPL mode",
    )
    chat_parser.add_argument("message", nargs="?", help="Message or instruction (omit for interactive REPL)", default=None)

    # distill
    distill_parser = subparsers.add_parser(
        "distill",
        help="Distill messy, informal user writing into a concrete build specification (and optionally execute it)",
    )
    distill_parser.add_argument("instruction", help="Messy or informal user writing to distill")
    distill_parser.add_argument(
        "--execute",
        action="store_true",
        help="Immediately build and evaluate the distilled concrete specification",
    )

    # resume
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume an interrupted or incomplete task by task_id",
    )
    resume_parser.add_argument("task_id", help="Task identifier to resume")

    # tasks
    tasks_parser = subparsers.add_parser(
        "tasks",
        help="List recent tasks recorded in persistent storage",
    )
    tasks_parser.add_argument("--limit", type=int, default=20, help="Maximum tasks to list")

    # inspect
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect full execution state, verification, live tests, and logs of a task",
    )
    inspect_parser.add_argument("task_id", help="Task identifier to inspect")

    # memory
    mem_parser = subparsers.add_parser(
        "memory",
        help="Query or store persistent memory records",
    )
    mem_parser.add_argument("--query", help="Search query", default=None)
    mem_parser.add_argument("--category", help="Filter by category", default=None)
    mem_parser.add_argument("--store-key", help="Store a memory with this key", default=None)
    mem_parser.add_argument("--store-content", help="Content to store", default=None)

    # experiences
    exp_parser = subparsers.add_parser(
        "experiences",
        help="Retrieve recorded task experiences and lessons",
    )
    exp_parser.add_argument("--query", help="Search query", default=None)
    exp_parser.add_argument("--limit", type=int, default=10, help="Maximum records")

    # develop-capability
    dev_parser = subparsers.add_parser(
        "develop-capability",
        help="Develop, verify, live-test, and register a new capability",
    )
    dev_parser.add_argument("capability_id", help="Identifier (e.g. 'text.metrics')")
    dev_parser.add_argument("--name", required=True, help="Human-readable capability name")
    dev_parser.add_argument("--purpose", required=True, help="Purpose / behavior specification")

    # providers
    prov_parser = subparsers.add_parser(
        "providers",
        help="List, add, update, enable/disable, or remove AI model providers",
    )
    prov_parser.add_argument("--upsert-id", help="Provider ID to add or update", default=None)
    prov_parser.add_argument("--name", help="Human-readable provider name", default=None)
    prov_parser.add_argument("--adapter", help="Adapter type (openai_compatible, anthropic, ollama, local_only, deterministic)", default=None)
    prov_parser.add_argument("--endpoint", help="Base/chat endpoint URL", default=None)
    prov_parser.add_argument("--api-key-env", help="Environment variable for API key", default="AGENT911_API_KEY")
    prov_parser.add_argument("--api-key", help="API key to store in mode-0600 CredentialStore", default=None)
    prov_parser.add_argument("--enabled", choices=["true", "false"], help="Enable or disable provider", default=None)
    prov_parser.add_argument("--remove", help="Remove provider by provider_id", default=None)

    # models
    mod_parser = subparsers.add_parser(
        "models",
        help="List, add, update, enable/disable, or remove configured model profiles",
    )
    mod_parser.add_argument("--upsert-id", help="Model profile ID to add or update", default=None)
    mod_parser.add_argument("--name", help="Human-readable model profile name", default=None)
    mod_parser.add_argument("--provider-id", help="Provider ID for this model", default=None)
    mod_parser.add_argument("--model-name", help="Provider-native model identifier (e.g. gpt-4o-mini)", default=None)
    mod_parser.add_argument("--endpoint-override", help="Optional endpoint override", default=None)
    mod_parser.add_argument("--context-window", type=int, help="Context window tokens (optional)", default=None)
    mod_parser.add_argument("--temperature", type=float, help="Temperature (0.0-2.0)", default=None)
    mod_parser.add_argument("--max-tokens", type=int, help="Max generation tokens", default=None)
    mod_parser.add_argument("--timeout", type=float, help="Request timeout seconds", default=None)
    mod_parser.add_argument("--retries", type=int, help="Max retries", default=None)
    mod_parser.add_argument("--enabled", choices=["true", "false"], help="Enable or disable model profile", default=None)
    mod_parser.add_argument("--remove", help="Remove model profile by profile_id", default=None)

    # model-select
    sel_parser = subparsers.add_parser(
        "model-select",
        help="Switch active model at runtime and optionally persist as default or set fallback model",
    )
    sel_parser.add_argument("profile_id", nargs="?", help="Model profile ID to activate", default=None)
    sel_parser.add_argument("--set-default", action="store_true", help="Also set as persistent default model")
    sel_parser.add_argument("--fallback", help="Set fallback model profile ID", default=None)

    # model-test
    mtest_parser = subparsers.add_parser(
        "model-test",
        help="Run a genuine connection test against the active or specified model profile",
    )
    mtest_parser.add_argument("--profile", help="Target model profile ID (defaults to active model)", default=None)

    # proposals
    prop_parser = subparsers.add_parser(
        "proposals",
        help="Detect operational weaknesses from execution history and list prioritized ImprovementProposals",
    )
    prop_parser.add_argument("--detect", action="store_true", help="Scan task/experience history and generate new proposals")
    prop_parser.add_argument("--status", help="Filter by status (PROPOSED, VALIDATED_SUCCESS, ROLLED_BACK_FAILED)", default=None)

    # self-improve
    impr_parser = subparsers.add_parser(
        "self-improve",
        help="Execute controlled, bounded self-improvement for a proposal with Verification, Live Run Testing, and rollback",
    )
    impr_parser.add_argument("proposal_id", help="Target ImprovementProposal ID to apply and evaluate")

    # diagnose
    diag_parser = subparsers.add_parser(
        "diagnose",
        help="Inspect structured self-diagnosis, strategy history, capability gap analysis, and resource metrics for a task",
    )
    diag_parser.add_argument("task_id", help="Task ID to diagnose")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    model_overrides: Dict[str, Any] = {}
    if args.provider:
        model_overrides["provider"] = args.provider
    if args.model_id:
        model_overrides["model_id"] = args.model_id
    if model_overrides:
        overrides["model"] = model_overrides

    try:
        config = AgentConfig.load(
            config_path=args.config,
            workspace_override=args.workspace,
            overrides=overrides if overrides else None,
        )
        orchestrator = Agent911Orchestrator(config=config)
    except ConfigurationError as exc:
        if args.json:
            print(json.dumps({"error": str(exc), "error_type": "ConfigurationError"}))
        else:
            print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    cmd = args.subcommand

    if cmd == "status":
        conn_report = orchestrator.model_interface.check_connectivity()
        caps = orchestrator.registry.list_capabilities()
        recent_tasks = orchestrator.memory_store.list_tasks(limit=5)
        memories = orchestrator.memory_store.retrieve_memories(limit=100)
        experiences = orchestrator.memory_store.retrieve_experiences(limit=100)
        payload = {
            "configuration": config.to_safe_dict(),
            "model_connectivity": conn_report.to_dict(),
            "capabilities_registered": len(caps),
            "capabilities_available": sum(
                1 for c in caps if c["availability"] == "AVAILABLE"
            ),
            "persisted_memories_count": len(memories),
            "persisted_experiences_count": len(experiences),
            "recent_tasks_count": len(recent_tasks),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print("=== AGENT 911 STATUS ===")
            print(f"Workspace Root         : {config.workspace_path}")
            print(f"Database Path          : {config.db_path}")
            print(f"Model Provider         : {conn_report.provider} ({conn_report.model_id})")
            print(f"Model Connectivity     : {conn_report.status.value} (live_reached={conn_report.provider_reached})")
            print(f"Connectivity Details   : {conn_report.details}")
            print(
                f"Registered Capabilities: {payload['capabilities_available']}/{payload['capabilities_registered']} available"
            )
            print(f"Persisted Memories     : {payload['persisted_memories_count']}")
            print(f"Persisted Experiences  : {payload['persisted_experiences_count']}")
        return 0

    if cmd == "capabilities":
        caps = orchestrator.registry.list_capabilities(only_available=args.only_available)
        if args.json:
            print(json.dumps({"count": len(caps), "capabilities": caps}, indent=2))
        else:
            print(f"=== REGISTERED CAPABILITIES ({len(caps)}) ===")
            for c in caps:
                grown_tag = " [CUSTOM GROWN]" if c.get("is_custom_grown") else ""
                print(
                    f"  - {c['identifier']} [{c['availability']}]{grown_tag}: {c['purpose']}"
                )
        return 0

    if cmd == "run":
        task = orchestrator.execute_task(args.instruction)
        if args.json:
            print(json.dumps(task.to_dict(), indent=2))
        else:
            print(_format_task_report(task))
        return 0 if task.final_outcome == TaskOutcome.SUCCESS else 1

    if cmd == "chat":
        if args.message:
            task = orchestrator.execute_task(args.message)
            if args.json:
                print(json.dumps(task.to_dict(), indent=2))
            else:
                print(f"Agent 911 ({task.interaction_mode}):\n{task.assistant_response}")
            return 0 if task.final_outcome == TaskOutcome.SUCCESS else 1

        print("=== Agent 911 Interactive Conversational & Execution REPL ===")
        print("Type any message to chat or any instruction to execute (or 'exit' to quit).")
        while True:
            try:
                user_msg = input("\nYou > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not user_msg:
                continue
            if user_msg.lower() in ("exit", "quit", ":q"):
                print("Goodbye!")
                break
            task = orchestrator.execute_task(user_msg)
            print(f"\nAgent 911 [{task.interaction_mode} | {task.final_outcome.value}] >\n{task.assistant_response}")
        return 0

    if cmd == "distill":
        if args.execute:
            task = orchestrator.execute_task(args.instruction)
            if args.json:
                print(json.dumps(task.to_dict(), indent=2))
            else:
                print(_format_task_report(task))
            return 0 if task.final_outcome == TaskOutcome.SUCCESS else 1

        spec_dict = orchestrator.distill_instruction(args.instruction)
        if args.json:
            print(json.dumps(spec_dict, indent=2))
        else:
            print("=== AGENT 911 DISTILLED CONCRETE SPECIFICATION ===")
            print(f"Raw User Writing         : {spec_dict['raw_input']}")
            print(f"Messy Input Detected     : {spec_dict['is_messy_input']}")
            print(f"Build Intent Detected    : {spec_dict['is_build_intent']}")
            print(f"What User Wants to Build : {spec_dict['what_user_wants_to_build']}")
            print(f"Concrete Instruction     : {spec_dict['concrete_instruction']}")
            if spec_dict.get("target_artifacts"):
                print("\nTarget Artifacts:")
                for art in spec_dict["target_artifacts"]:
                    print(
                        f"  - {art['path']} ({art['artifact_type']}) | functions={art.get('functions', [])}"
                    )
            if spec_dict.get("ambiguities_resolved"):
                print("\nAmbiguities Resolved:")
                for amb in spec_dict["ambiguities_resolved"]:
                    print(f"  * {amb}")
        return 0

    if cmd == "resume":
        try:
            task = orchestrator.resume_task(args.task_id)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(task.to_dict(), indent=2))
        else:
            print(_format_task_report(task))
        return 0 if task.final_outcome == TaskOutcome.SUCCESS else 1

    if cmd == "tasks":
        tasks = orchestrator.memory_store.list_tasks(limit=args.limit)
        if args.json:
            print(json.dumps({"count": len(tasks), "tasks": tasks}, indent=2))
        else:
            print(f"=== PERSISTED TASKS ({len(tasks)}) ===")
            for t in tasks:
                print(
                    f"  {t['task_id']} | {t['outcome']} | {t['stage']} | {t['original_instruction']}"
                )
        return 0

    if cmd == "inspect":
        task = orchestrator.memory_store.load_task(args.task_id)
        if task is None:
            print(f"Task '{args.task_id}' not found.", file=sys.stderr)
            return 1
        logs = orchestrator.memory_store.get_execution_logs(args.task_id)
        if args.json:
            out = task.to_dict()
            out["execution_logs"] = logs
            print(json.dumps(out, indent=2))
        else:
            print(_format_task_report(task))
            print(f"\n--- Execution Logs ({len(logs)}) ---")
            for lg in logs:
                print(
                    f"  [{lg['timestamp']}] step={lg['step_id']} cap={lg['capability_id']} "
                    f"success={lg['success']} ({lg['duration_ms']:.1f}ms)"
                )
        return 0

    if cmd == "memory":
        if args.store_key and args.store_content is not None:
            rec = orchestrator.memory_store.store_memory(
                category=args.category or "FACTUAL",
                key=args.store_key,
                content=args.store_content,
            )
            if args.json:
                print(json.dumps(rec.to_dict(), indent=2))
            else:
                print(f"Stored memory [{rec.memory_id}] {rec.category.value}:{rec.key} = {rec.content}")
            return 0

        records = orchestrator.memory_store.retrieve_memories(
            query=args.query, category=args.category, limit=20
        )
        if args.json:
            print(json.dumps([r.to_dict() for r in records], indent=2))
        else:
            print(f"=== PERSISTENT MEMORIES ({len(records)}) ===")
            for r in records:
                print(
                    f"  [{r.memory_id}] {r.category.value} | {r.key} ({r.confidence.value}): {r.content}"
                )
        return 0

    if cmd == "experiences":
        exps = orchestrator.memory_store.retrieve_experiences(
            query=args.query, limit=args.limit
        )
        if args.json:
            print(json.dumps([e.to_dict() for e in exps], indent=2))
        else:
            print(f"=== RECORDED EXPERIENCES ({len(exps)}) ===")
            for e in exps:
                print(
                    f"  [{e.experience_id}] task={e.task_id} outcome={e.outcome.value} "
                    f"caps={e.capabilities_used} | objective='{e.objective}'"
                )
                if e.successful_remedies:
                    print(f"      remedies: {e.successful_remedies}")
        return 0

    if cmd == "develop-capability":
        task = orchestrator.execute_task(
            f"Develop capability '{args.capability_id}': {args.purpose}",
            explicit_plan=[
                {
                    "step_id": "step-1",
                    "description": f"Develop and register capability '{args.capability_id}'",
                    "capability_id": "capability.develop",
                    "parameters": {
                        "capability_id": args.capability_id,
                        "name": args.name,
                        "purpose": args.purpose,
                        "behavior_spec": args.purpose,
                    },
                }
            ],
            explicit_requirements=[
                {
                    "requirement_id": "req-cap-op",
                    "description": f"Capability '{args.capability_id}' is operational",
                    "category": "capability_operational",
                    "target": args.capability_id,
                }
            ],
        )
        if args.json:
            print(json.dumps(task.to_dict(), indent=2))
        else:
            print(_format_task_report(task))
        return 0 if task.final_outcome == TaskOutcome.SUCCESS else 1

    if cmd == "providers":
        try:
            if args.remove:
                orchestrator.remove_provider(args.remove, save=True)
                msg = {"removed_provider": args.remove}
                print(json.dumps(msg, indent=2) if args.json else f"Removed provider '{args.remove}'.")
                return 0
            if args.upsert_id:
                existing = config.providers.get(args.upsert_id)
                pdata: Dict[str, Any] = existing.to_dict() if existing else {
                    "provider_id": args.upsert_id,
                    "name": args.name or args.upsert_id,
                    "adapter_type": args.adapter or "openai_compatible",
                    "endpoint": args.endpoint or "https://api.openai.com/v1/chat/completions",
                    "api_key_env": args.api_key_env or "AGENT911_API_KEY",
                    "enabled": True,
                }
                if args.name:
                    pdata["name"] = args.name
                if args.adapter:
                    pdata["adapter_type"] = args.adapter
                if args.endpoint:
                    pdata["endpoint"] = args.endpoint
                if args.api_key_env:
                    pdata["api_key_env"] = args.api_key_env
                if args.enabled is not None:
                    pdata["enabled"] = args.enabled == "true"
                saved = orchestrator.upsert_provider(pdata, api_key=args.api_key, save=True)
                print(json.dumps(saved, indent=2) if args.json else f"Saved provider '{args.upsert_id}'.")
                return 0
            provs = orchestrator.list_providers()
            if args.json:
                print(json.dumps({"providers": provs}, indent=2))
            else:
                print(f"=== CONFIGURED PROVIDERS ({len(provs)}) ===")
                for p in provs:
                    state = "ENABLED" if p["enabled"] else "DISABLED"
                    cred = p.get("credential_preview") or "no-key"
                    print(f"  - {p['provider_id']} [{state}] ({p['adapter_type']}): {p['endpoint']} | key={cred}")
            return 0
        except ConfigurationError as exc:
            print(json.dumps(exc.to_dict(), indent=2) if args.json else f"Error: {exc}", file=sys.stderr)
            return 2

    if cmd == "models":
        try:
            if args.remove:
                orchestrator.remove_model(args.remove, save=True)
                msg = {"removed_model": args.remove}
                print(json.dumps(msg, indent=2) if args.json else f"Removed model '{args.remove}'.")
                return 0
            if args.upsert_id:
                existing_m = config.models.get(args.upsert_id)
                mdata: Dict[str, Any] = existing_m.to_dict() if existing_m else {
                    "profile_id": args.upsert_id,
                    "name": args.name or args.upsert_id,
                    "provider_id": args.provider_id or "default",
                    "model_id": args.model_name or args.upsert_id,
                }
                if args.name:
                    mdata["name"] = args.name
                if args.provider_id:
                    mdata["provider_id"] = args.provider_id
                if args.model_name:
                    mdata["model_id"] = args.model_name
                if args.endpoint_override is not None:
                    mdata["endpoint_override"] = args.endpoint_override or None
                if args.context_window is not None:
                    mdata["context_window_tokens"] = args.context_window
                if args.temperature is not None:
                    mdata["temperature"] = args.temperature
                if args.max_tokens is not None:
                    mdata["max_tokens"] = args.max_tokens
                if args.timeout is not None:
                    mdata["timeout_seconds"] = args.timeout
                if args.retries is not None:
                    mdata["max_retries"] = args.retries
                if args.enabled is not None:
                    mdata["enabled"] = args.enabled == "true"
                saved = orchestrator.upsert_model(mdata, save=True)
                print(json.dumps(saved, indent=2) if args.json else f"Saved model profile '{args.upsert_id}'.")
                return 0
            mods = orchestrator.list_models()
            if args.json:
                print(
                    json.dumps(
                        {
                            "active_model": config.active_model,
                            "default_model": config.default_model,
                            "fallback_model": config.fallback_model,
                            "models": mods,
                        },
                        indent=2,
                    )
                )
            else:
                print(f"=== CONFIGURED MODEL PROFILES ({len(mods)}) ===")
                for m in mods:
                    flags = []
                    if m.get("is_active"):
                        flags.append("ACTIVE")
                    if m.get("is_default"):
                        flags.append("DEFAULT")
                    if m.get("is_fallback"):
                        flags.append("FALLBACK")
                    if not m.get("enabled", True):
                        flags.append("DISABLED")
                    badge = f" [{', '.join(flags)}]" if flags else ""
                    print(
                        f"  - {m['profile_id']}{badge}: provider={m['provider_id']} model_id={m['model_id']} "
                        f"ctx={m.get('context_window_tokens') or 'unknown'}"
                    )
            return 0
        except ConfigurationError as exc:
            print(json.dumps(exc.to_dict(), indent=2) if args.json else f"Error: {exc}", file=sys.stderr)
            return 2

    if cmd == "model-select":
        try:
            res: Dict[str, Any] = {}
            if args.profile_id:
                res = orchestrator.select_active_model(
                    args.profile_id, set_as_default=args.set_default, save=True
                )
            if args.fallback is not None:
                fb_val = None if args.fallback.lower() in ("none", "") else args.fallback
                res = orchestrator.set_fallback_model(fb_val, save=True)
            if not res:
                res = {
                    "active_model": config.active_model,
                    "default_model": config.default_model,
                    "fallback_model": config.fallback_model,
                }
            print(json.dumps(res, indent=2) if args.json else f"Active model selection: {res}")
            return 0
        except ConfigurationError as exc:
            print(json.dumps(exc.to_dict(), indent=2) if args.json else f"Error: {exc}", file=sys.stderr)
            return 2

    if cmd == "model-test":
        report = orchestrator.test_model_connection(profile_id=args.profile)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print("=== MODEL CONNECTION TEST ===")
            print(f"Status             : {report.status.value}")
            print(f"Provider / Model   : {report.provider} / {report.model_id}")
            print(f"Endpoint           : {report.endpoint}")
            print(f"Config Valid       : {report.config_valid}")
            print(f"Endpoint Reachable : {report.endpoint_reachable}")
            print(f"Auth Valid         : {report.auth_valid}")
            print(f"Model Accepted     : {report.model_accepted}")
            print(f"Inference Succeeded: {report.inference_succeeded}")
            print(f"Conclusive         : {report.conclusive}")
            print(f"Details            : {report.details}")
        return 0 if report.inference_succeeded or report.status.value in ("DETERMINISTIC_TEST_DOUBLE", "LOCAL_ONLY") else 1

    if cmd == "proposals":
        if args.detect:
            orchestrator.detect_and_propose_improvements()
        proposals = orchestrator.list_improvement_proposals(status=args.status)
        if args.json:
            print(json.dumps(proposals, indent=2))
        else:
            print(f"=== IMPROVEMENT PROPOSALS ({len(proposals)}) ===")
            for p in proposals:
                print(
                    f"  [{p['proposal_id']}] {p['priority']} | {p['status']} | {p['title']}\n"
                    f"      Problem : {p['observed_problem']}\n"
                    f"      Change  : {p['proposed_change']}"
                )
        return 0

    if cmd == "self-improve":
        try:
            run_rec = orchestrator.run_controlled_self_improvement(args.proposal_id)
            if args.json:
                print(json.dumps(run_rec, indent=2))
            else:
                print("=== CONTROLLED SELF-IMPROVEMENT RUN ===")
                print(f"Run ID           : {run_rec['run_id']}")
                print(f"Proposal ID      : {run_rec['proposal_id']}")
                print(f"Disposition      : {run_rec['final_disposition']}")
                print(f"Verification     : {run_rec['verification_status']}")
                print(f"Live Run Testing : {run_rec['live_test_status']}")
                print(f"Rolled Back      : {run_rec['rolled_back']}")
                print(f"Notes            : {run_rec['notes']}")
            return 0 if run_rec["final_disposition"] == "VALIDATED_SUCCESS" else 1
        except Exception as exc:
            print(json.dumps({"error": str(exc)}, indent=2) if args.json else f"Error: {exc}", file=sys.stderr)
            return 1

    if cmd == "diagnose":
        try:
            diag = orchestrator.diagnose_task(args.task_id)
            if args.json:
                print(json.dumps(diag, indent=2))
            else:
                print("=== TASK SELF-DIAGNOSIS & STRATEGY REPORT ===")
                print(f"Task ID              : {diag['task_id']}")
                print(f"Outcome              : {diag['outcome']}")
                print(f"Selected Strategy    : {diag['selected_strategy']}")
                print(f"Loop Detected        : {diag['unproductive_loop_detected']} ({diag['unproductive_loop_description']})")
                print(f"Diagnostic Reports   : {len(diag['diagnostic_reports'])}")
                for dr in diag["diagnostic_reports"]:
                    print(
                        f"  - [{dr['report_id']}] step={dr['step_id']} root_cause={dr['root_cause_category']} "
                        f"confirmed={dr['confirmed_facts']}"
                    )
            return 0
        except KeyError as exc:
            print(json.dumps({"error": str(exc)}, indent=2) if args.json else f"Error: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
