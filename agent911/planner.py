"""Instruction understanding, task decomposition, capability selection, and experience-informed planning."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent911.capabilities.registry import CapabilityRegistry
from agent911.config import AgentConfig
from agent911.model_provider import ModelInterface, ModelProviderError
from agent911.models import (
    ActionStatus,
    CapabilityAvailability,
    InstructionIntent,
    PlannedAction,
    ProblemSolvingStrategy,
    TaskRequirement,
    TaskState,
    new_id,
)
from agent911.strategy import StrategySelector


class TaskPlanner:
    """
    Translates user instructions and retrieved context/experience into structured
    InstructionIntent, TaskRequirements, and actionable PlannedActions.
    """

    def __init__(
        self,
        config: AgentConfig,
        registry: CapabilityRegistry,
        model_interface: ModelInterface,
    ) -> None:
        self.config = config
        self.registry = registry
        self.model_interface = model_interface
        self.strategy_selector = StrategySelector()

    def understand_and_plan(
        self,
        task: TaskState,
        retrieved_context: Dict[str, Any],
        explicit_plan: Optional[List[Dict[str, Any]]] = None,
        explicit_requirements: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Populate task.intent, task.interpreted_objective, task.requirements,
        task.assumptions, and task.planned_actions.
        """
        instruction = task.original_instruction.strip()
        experiences = retrieved_context.get("relevant_experiences", [])
        memories = retrieved_context.get("factual_memories", []) + retrieved_context.get(
            "instruction_memories", []
        )

        # If caller supplied an explicit structured plan/requirements, honor it while still
        # extracting intent and applying experience lessons.
        if explicit_plan is not None:
            intent = self._derive_heuristic_intent(instruction, experiences, memories)
            task.intent = intent
            task.interpreted_objective = intent.main_goal
            task.assumptions = list(intent.assumptions)
            task.planned_actions = [
                PlannedAction(
                    step_id=p.get("step_id") or f"step-{idx + 1}",
                    description=p.get("description", f"Execute {p['capability_id']}"),
                    capability_id=p["capability_id"],
                    parameters=dict(p.get("parameters", {})),
                    depends_on=list(p.get("depends_on", [])),
                    required=bool(p.get("required", True)),
                    is_capability_growth=(p["capability_id"] == "capability.develop"),
                )
                for idx, p in enumerate(explicit_plan)
            ]
            if explicit_requirements is not None:
                task.requirements = [
                    TaskRequirement(
                        requirement_id=r.get("requirement_id") or f"req-{idx + 1}",
                        description=r["description"],
                        category=r["category"],
                        target=r.get("target"),
                        expected=r.get("expected"),
                        required=bool(r.get("required", True)),
                    )
                    for idx, r in enumerate(explicit_requirements)
                ]
            else:
                task.requirements = self._infer_requirements_from_actions(task.planned_actions)
            self._apply_experience_to_plan(task, experiences, memories)
            self._finalize_plan_and_strategy(task, experiences)
            task.refresh_action_lists()
            return

        # Try AI model-based understanding and planning when available and configured
        used_model = False
        if self._should_try_model(instruction):
            try:
                used_model = self._plan_via_model(task, retrieved_context)
            except ModelProviderError as exc:
                if not self.config.model.allow_local_fallback:
                    raise
                task.decisions.append(
                    f"Model provider unavailable ({exc.error_type}); using local instruction planner."
                )

        if not used_model:
            intent = self._derive_heuristic_intent(instruction, experiences, memories)
            task.intent = intent
            task.interpreted_objective = intent.main_goal
            task.assumptions = list(intent.assumptions)
            actions, reqs = self._build_heuristic_plan(instruction, intent, experiences)
            if explicit_requirements is not None:
                reqs = [
                    TaskRequirement(
                        requirement_id=r.get("requirement_id") or f"req-{idx + 1}",
                        description=r["description"],
                        category=r["category"],
                        target=r.get("target"),
                        expected=r.get("expected"),
                        required=bool(r.get("required", True)),
                    )
                    for idx, r in enumerate(explicit_requirements)
                ]
            task.planned_actions = actions
            task.requirements = reqs

        self._apply_experience_to_plan(task, experiences, memories)
        self._finalize_plan_and_strategy(task, experiences)
        task.refresh_action_lists()

    def _should_try_model(self, instruction: str) -> bool:
        if self.config.model.provider == "deterministic" or self.model_interface.deterministic_responder:
            return True
        if self.config.model.provider == "local_only":
            return False
        if self.config.model.resolve_api_key() or self.config.model.provider == "ollama":
            return True
        # If instruction explicitly demands live AI model reasoning without fallback
        if "query the ai model" in instruction.lower() or "use model.query" in instruction.lower():
            return False  # Handled as a model.query step so availability is properly checked
        return False

    def _plan_via_model(
        self, task: TaskState, retrieved_context: Dict[str, Any]
    ) -> bool:
        caps_summary = [
            {
                "identifier": c["identifier"],
                "purpose": c["purpose"],
                "availability": c["availability"],
                "required_params": c.get("input_schema", {}).get("required", []),
            }
            for c in self.registry.list_capabilities()
        ]
        sys_prompt = (
            "You are Agent 911's task planner. Analyze the user's instruction and return a "
            "JSON object with keys: 'main_goal' (str), 'explicit_requirements' (list[str]), "
            "'expected_outputs' (list[str]), 'assumptions' (list[str]), 'planned_actions' "
            "(list of {'step_id', 'description', 'capability_id', 'parameters'}), and "
            "'requirements' (list of {'requirement_id', 'description', 'category', 'target', 'expected'})."
        )
        resp = self.model_interface.generate(
            system_prompt=sys_prompt,
            user_prompt=f"Instruction: {task.original_instruction}",
            context_payload={
                "available_capabilities": caps_summary,
                "retrieved_context": retrieved_context,
            },
            expect_json=True,
        )
        data = resp.structured_data
        if not isinstance(data, dict) or not data.get("planned_actions"):
            return False

        intent = InstructionIntent(
            original_instruction=task.original_instruction,
            main_goal=str(data.get("main_goal") or task.original_instruction),
            explicit_requirements=list(data.get("explicit_requirements", [])),
            expected_outputs=list(data.get("expected_outputs", [])),
            constraints=list(data.get("constraints", [])),
            required_capabilities=[
                a.get("capability_id", "") for a in data.get("planned_actions", [])
            ],
            assumptions=list(data.get("assumptions", [])),
            completion_conditions=list(data.get("completion_conditions", [])),
            target_artifacts=list(data.get("target_artifacts", [])),
            interpreted_by="model",
        )
        task.intent = intent
        task.interpreted_objective = intent.main_goal
        task.assumptions = list(intent.assumptions)

        actions: List[PlannedAction] = []
        for idx, raw_act in enumerate(data["planned_actions"]):
            cap_id = str(raw_act.get("capability_id", ""))
            actions.append(
                PlannedAction(
                    step_id=str(raw_act.get("step_id") or f"step-{idx + 1}"),
                    description=str(raw_act.get("description") or f"Invoke {cap_id}"),
                    capability_id=cap_id,
                    parameters=dict(raw_act.get("parameters") or {}),
                    depends_on=list(raw_act.get("depends_on") or []),
                    required=bool(raw_act.get("required", True)),
                    is_capability_growth=(cap_id == "capability.develop"),
                )
            )
        task.planned_actions = actions

        raw_reqs = data.get("requirements")
        if isinstance(raw_reqs, list) and raw_reqs:
            task.requirements = [
                TaskRequirement(
                    requirement_id=str(r.get("requirement_id") or f"req-{idx + 1}"),
                    description=str(r.get("description") or "Verify requirement"),
                    category=str(r.get("category") or "output_present"),
                    target=r.get("target"),
                    expected=r.get("expected"),
                    required=bool(r.get("required", True)),
                )
                for idx, r in enumerate(raw_reqs)
            ]
        else:
            task.requirements = self._infer_requirements_from_actions(actions)

        task.decisions.append(
            f"Derived task plan using model interface ({resp.provider}:{resp.model_id})."
        )
        return True

    def _derive_heuristic_intent(
        self,
        instruction: str,
        experiences: List[Dict[str, Any]],
        memories: List[Dict[str, Any]],
    ) -> InstructionIntent:
        lower = instruction.lower()
        reqs: List[str] = []
        outputs: List[str] = []
        caps: List[str] = []
        assumptions: List[str] = []
        artifacts: List[str] = []

        # Extract file paths mentioned in instruction
        file_matches = re.findall(
            r"(?:['\"`])([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+)(?:['\"`])|\b([a-zA-Z0-9_\-\./]+\.(?:py|txt|md|json|csv|toml|yaml|yml|sh|log))\b",
            instruction,
        )
        for m1, m2 in file_matches:
            cand = m1 or m2
            if cand and cand not in artifacts:
                artifacts.append(cand)

        if any(w in lower for w in ("write", "create file", "save", "generate file", "create a python", "create script")):
            caps.append("fs.write")
            reqs.append("Create requested file artifact(s) with expected content.")
            outputs.extend(artifacts or ["created_file"])
        if any(w in lower for w in ("update file", "modify file", "replace in", "append to")):
            caps.append("fs.update")
            reqs.append("Update existing file with requested changes.")
        if any(w in lower for w in ("mkdir", "create directory", "make directory", "create folder")):
            caps.append("fs.mkdir")
            reqs.append("Create target directory inside workspace.")
        if any(w in lower for w in ("read file", "read the file", "show content of", "cat ")):
            caps.append("fs.read")
            reqs.append("Read and return contents of target file.")
        if any(w in lower for w in ("list files", "list directory", "ls ")):
            caps.append("fs.list")
            reqs.append("List entries in target workspace directory.")
        if any(w in lower for w in ("inspect project", "analyze project", "project structure", "discover project")):
            caps.append("project.inspect")
            reqs.append("Inspect software project structure, manifests, entry points, and tests.")
            outputs.append("project_inspection_report")
        if any(w in lower for w in ("run test", "execute test", "pytest", "unittest")):
            caps.append("test.run")
            reqs.append("Run automated test suite and verify tests pass.")
            outputs.append("test_execution_report")
        if any(w in lower for w in ("run command", "execute command", "shell:", "cmd:")):
            caps.append("cmd.run")
            reqs.append("Execute specified command and verify exit code 0.")
            outputs.append("command_output")
        if any(w in lower for w in ("remember", "store memory", "save to memory", "memorize")):
            caps.append("memory.store")
            reqs.append("Persist requested information into SQLite memory store.")
        if any(w in lower for w in ("recall", "retrieve memory", "search memory")):
            caps.append("memory.retrieve")
            reqs.append("Retrieve relevant records from persistent memory.")
        if any(w in lower for w in ("develop capability", "create capability", "build capability", "new capability")):
            caps.append("capability.develop")
            reqs.append("Develop, verify, live-test, and register new capability.")
        if any(w in lower for w in ("word count", "text metrics", "text.metrics")):
            caps.append("text.metrics")
            reqs.append("Compute text metrics using capability 'text.metrics'.")
        if any(w in lower for w in ("csv to json", "data.csv_to_json")):
            caps.append("data.csv_to_json")
            reqs.append("Convert CSV data to JSON using capability 'data.csv_to_json'.")
        if any(w in lower for w in ("query the ai model", "ask the model", "use model.query")):
            caps.append("model.query")
            reqs.append("Query configured AI model provider.")

        if not caps:
            if any(
                w in lower
                for w in (
                    "inspect",
                    "analyze",
                    "discover",
                    "build",
                    "audit",
                    "scan",
                    "workspace structure",
                )
            ):
                caps.append("project.inspect")
                assumptions.append(
                    "No explicit tool named in instruction; defaulting to workspace inspection."
                )
            else:
                caps.append("chat.respond")
                reqs.append("Respond naturally and helpfully to the user's conversational message.")
                outputs.append("conversational_reply")

        inferred_reqs: List[str] = []
        for art in artifacts:
            if art.endswith(".py"):
                inferred_reqs.append(f"Python file '{art}' must compile without syntax errors.")
            elif art.endswith(".json"):
                inferred_reqs.append(f"JSON file '{art}' must parse as valid JSON.")

        avail_resources = [
            c["identifier"]
            for c in self.registry.list_capabilities()
            if c["availability"] == CapabilityAvailability.AVAILABLE.value
        ]

        unknowns: List[str] = []
        clarification_needed = False
        clarification_question: Optional[str] = None
        if any(w in lower for w in ("delete everything", "wipe production", "drop database")):
            clarification_needed = True
            clarification_question = (
                "Instruction requests an irreversible destructive operation; explicit confirmation of target scope is required."
            )
            unknowns.append("Destructive target scope confirmation")

        if len(caps) == 1 and caps[0] in ("fs.read", "fs.list", "project.inspect", "memory.store", "memory.retrieve", "model.query", "chat.respond"):
            complexity = "simple_direct"
        elif len(caps) <= 2 and "capability.develop" not in caps:
            complexity = "moderate_inspect_act"
        else:
            complexity = "complex_multi_stage"

        return InstructionIntent(
            original_instruction=instruction,
            main_goal=instruction,
            explicit_requirements=reqs or [f"Accomplish: {instruction}"],
            inferred_requirements=inferred_reqs,
            expected_outputs=outputs or ["task_execution_result"],
            constraints=["Operate within configured workspace boundary."],
            available_resources=avail_resources,
            relevant_context={
                "matched_experiences": len(experiences),
                "matched_memories": len(memories),
            },
            required_capabilities=caps,
            assumptions=assumptions,
            completion_conditions=["All required Verification checks and Live Run Tests pass."],
            unknowns=unknowns,
            clarification_needed=clarification_needed,
            clarification_question=clarification_question,
            complexity_level=complexity,
            target_artifacts=artifacts,
            interpreted_by="heuristic",
        )

    def _build_heuristic_plan(
        self,
        instruction: str,
        intent: InstructionIntent,
        experiences: List[Dict[str, Any]],
    ) -> Tuple[List[PlannedAction], List[TaskRequirement]]:
        lower = instruction.lower()
        actions: List[PlannedAction] = []
        requirements: List[TaskRequirement] = []
        step_counter = 1

        def next_step_id() -> str:
            nonlocal step_counter
            sid = f"step-{step_counter}"
            step_counter += 1
            return sid

        # 1. Check for capability development or missing capability gaps
        cap_dev_match = re.search(
            r"(?:develop|create|build)\s+capability\s+['\"`]?([a-zA-Z0-9_\.]+)['\"`]?",
            instruction,
            re.IGNORECASE,
        )
        if cap_dev_match:
            cap_id = cap_dev_match.group(1)
            actions.append(
                PlannedAction(
                    step_id=next_step_id(),
                    description=f"Develop, verify, and live-test capability '{cap_id}'",
                    capability_id="capability.develop",
                    parameters={
                        "capability_id": cap_id,
                        "name": f"Custom {cap_id}",
                        "purpose": instruction,
                        "behavior_spec": instruction,
                    },
                    is_capability_growth=True,
                )
            )
            requirements.append(
                TaskRequirement(
                    requirement_id=f"req-cap-{cap_id}",
                    description=f"Capability '{cap_id}' is registered and operational",
                    category="capability_operational",
                    target=cap_id,
                )
            )

        # Check if any required capability in intent does not yet exist in registry
        for req_cap in intent.required_capabilities:
            if req_cap == "capability.develop":
                continue
            avail = self.registry.get_availability(req_cap)
            if avail == CapabilityAvailability.NOT_IMPLEMENTED:
                if not any(
                    a.capability_id == "capability.develop"
                    and a.parameters.get("capability_id") == req_cap
                    for a in actions
                ):
                    dev_sid = next_step_id()
                    actions.append(
                        PlannedAction(
                            step_id=dev_sid,
                            description=f"Develop missing capability '{req_cap}' required by instruction",
                            capability_id="capability.develop",
                            parameters={
                                "capability_id": req_cap,
                                "name": f"Grown {req_cap}",
                                "purpose": f"Auto-developed capability for: {instruction}",
                                "behavior_spec": instruction,
                            },
                            is_capability_growth=True,
                        )
                    )
                    requirements.append(
                        TaskRequirement(
                            requirement_id=f"req-cap-{req_cap}",
                            description=f"Newly grown capability '{req_cap}' is operational",
                            category="capability_operational",
                            target=req_cap,
                        )
                    )
                # Also invoke the newly developed capability
                invoke_params: Dict[str, Any] = {}
                if intent.target_artifacts:
                    invoke_params["path"] = intent.target_artifacts[0]
                else:
                    invoke_params["text"] = instruction
                actions.append(
                    PlannedAction(
                        step_id=next_step_id(),
                        description=f"Invoke capability '{req_cap}' on target input",
                        capability_id=req_cap,
                        parameters=invoke_params,
                    )
                )

        # 2. Check for file creation / writing instructions
        if "fs.write" in intent.required_capabilities and intent.target_artifacts:
            target_file = intent.target_artifacts[0]
            content, func_name = self._extract_file_content_from_instruction(
                instruction, target_file
            )
            actions.append(
                PlannedAction(
                    step_id=next_step_id(),
                    description=f"Write file '{target_file}'",
                    capability_id="fs.write",
                    parameters={"path": target_file, "content": content},
                )
            )
            requirements.append(
                TaskRequirement(
                    requirement_id="req-file-exists",
                    description=f"File '{target_file}' exists in workspace",
                    category="file_exists",
                    target=target_file,
                )
            )
            if target_file.endswith(".py"):
                requirements.append(
                    TaskRequirement(
                        requirement_id="req-code-compiles",
                        description=f"Python file '{target_file}' compiles without syntax errors",
                        category="code_compiles",
                        target=target_file,
                    )
                )
                if func_name:
                    requirements.append(
                        TaskRequirement(
                            requirement_id="req-func-exists",
                            description=f"Function '{func_name}' is defined in '{target_file}'",
                            category="function_exists",
                            target=target_file,
                            expected=func_name,
                        )
                    )
            elif target_file.endswith(".json"):
                requirements.append(
                    TaskRequirement(
                        requirement_id="req-json-valid",
                        description=f"JSON file '{target_file}' is valid JSON",
                        category="json_valid",
                        target=target_file,
                    )
                )
            else:
                requirements.append(
                    TaskRequirement(
                        requirement_id="req-file-content",
                        description=f"File '{target_file}' contains expected content",
                        category="file_content",
                        target=target_file,
                        expected=content.strip()[:40] if content.strip() else None,
                    )
                )

        # 3. Check for directory creation
        if "fs.mkdir" in intent.required_capabilities:
            dir_match = re.search(
                r"(?:directory|folder|mkdir)\s+['\"`]?([a-zA-Z0-9_\-\./]+)['\"`]?",
                instruction,
                re.IGNORECASE,
            )
            target_dir = dir_match.group(1) if dir_match else "new_directory"
            actions.append(
                PlannedAction(
                    step_id=next_step_id(),
                    description=f"Create directory '{target_dir}'",
                    capability_id="fs.mkdir",
                    parameters={"path": target_dir, "parents": True},
                )
            )
            requirements.append(
                TaskRequirement(
                    requirement_id="req-dir-exists",
                    description=f"Directory '{target_dir}' exists",
                    category="directory_exists",
                    target=target_dir,
                )
            )

        # 4. Check for file reading
        if "fs.read" in intent.required_capabilities and intent.target_artifacts:
            target_file = intent.target_artifacts[0]
            actions.append(
                PlannedAction(
                    step_id=next_step_id(),
                    description=f"Read file '{target_file}'",
                    capability_id="fs.read",
                    parameters={"path": target_file},
                )
            )
            requirements.append(
                TaskRequirement(
                    requirement_id="req-read-output",
                    description=f"Read content from '{target_file}'",
                    category="file_exists",
                    target=target_file,
                )
            )

        # 5. Check for project inspection
        if "project.inspect" in intent.required_capabilities and not actions:
            actions.append(
                PlannedAction(
                    step_id=next_step_id(),
                    description="Inspect workspace project structure, manifests, and tests",
                    capability_id="project.inspect",
                    parameters={"path": ".", "include_ast_symbols": True},
                )
            )
            requirements.append(
                TaskRequirement(
                    requirement_id="req-project-inspect",
                    description="Project inspection output produced",
                    category="output_present",
                    target="step-1",
                )
            )

        # 6. Check for test execution
        if "test.run" in intent.required_capabilities:
            target_test = intent.target_artifacts[0] if intent.target_artifacts else "tests"
            runner = "pytest" if "pytest" in lower else "unittest"
            actions.append(
                PlannedAction(
                    step_id=next_step_id(),
                    description=f"Run automated tests on '{target_test}' with {runner}",
                    capability_id="test.run",
                    parameters={"target": target_test, "runner": runner},
                )
            )
            requirements.append(
                TaskRequirement(
                    requirement_id="req-tests-pass",
                    description="Automated test suite passes",
                    category="tests_pass",
                    target=target_test,
                )
            )

        # 7. Check for command execution
        if "cmd.run" in intent.required_capabilities:
            cmd_match = re.search(
                r"(?:command:?|cmd:|shell:)\s*(.+)$", instruction, re.IGNORECASE
            )
            cmd_str = cmd_match.group(1).strip() if cmd_match else "python3 --version"
            if (cmd_str.startswith("`") and cmd_str.endswith("`")) or (
                cmd_str.startswith('"') and cmd_str.endswith('"')
            ):
                cmd_str = cmd_str[1:-1].strip()
            actions.append(
                PlannedAction(
                    step_id=next_step_id(),
                    description=f"Execute command: {cmd_str}",
                    capability_id="cmd.run",
                    parameters={"command": cmd_str},
                )
            )
            requirements.append(
                TaskRequirement(
                    requirement_id="req-cmd-succeeds",
                    description=f"Command '{cmd_str}' succeeds with exit code 0",
                    category="command_succeeds",
                )
            )

        # 8. Check for memory operations
        if "memory.store" in intent.required_capabilities:
            kv_match = re.search(
                r"(?:remember|store memory|memorize)\s+(?:that\s+)?([a-zA-Z0-9_\-\.]+)\s*(?:=|is|:)\s*(.+)",
                instruction,
                re.IGNORECASE,
            )
            mem_key = kv_match.group(1).strip() if kv_match else "user_note"
            mem_val = kv_match.group(2).strip() if kv_match else instruction
            actions.append(
                PlannedAction(
                    step_id=next_step_id(),
                    description=f"Store memory key '{mem_key}'",
                    capability_id="memory.store",
                    parameters={
                        "category": "FACTUAL",
                        "key": mem_key,
                        "content": mem_val,
                        "confidence": "VERIFIED",
                    },
                )
            )
            requirements.append(
                TaskRequirement(
                    requirement_id="req-mem-stored",
                    description=f"Memory record '{mem_key}' persisted in SQLite",
                    category="memory_stored",
                    target=mem_key,
                    expected=mem_val,
                )
            )

        if "memory.retrieve" in intent.required_capabilities:
            actions.append(
                PlannedAction(
                    step_id=next_step_id(),
                    description="Retrieve matching memories from SQLite",
                    capability_id="memory.retrieve",
                    parameters={"query": instruction},
                )
            )
            requirements.append(
                TaskRequirement(
                    requirement_id="req-mem-retrieved",
                    description="Memory retrieval output present",
                    category="output_present",
                )
            )

        # 9. Check for explicit AI model query
        if "model.query" in intent.required_capabilities:
            actions.append(
                PlannedAction(
                    step_id=next_step_id(),
                    description="Query configured AI model provider",
                    capability_id="model.query",
                    parameters={"prompt": instruction},
                )
            )
            requirements.append(
                TaskRequirement(
                    requirement_id="req-model-output",
                    description="AI model query returned response",
                    category="output_present",
                )
            )

        # 10. Conversational response via chat.respond
        if "chat.respond" in intent.required_capabilities and not any(
            a.capability_id == "chat.respond" for a in actions
        ):
            sid = next_step_id()
            actions.append(
                PlannedAction(
                    step_id=sid,
                    description="Generate contextual conversational response",
                    capability_id="chat.respond",
                    parameters={"message": instruction},
                )
            )
            requirements.append(
                TaskRequirement(
                    requirement_id=f"req-chat-{sid}",
                    description="Conversational agent response generated",
                    category="output_present",
                    target=sid,
                )
            )

        # 11. Check if already-registered custom capability was requested
        for req_cap in intent.required_capabilities:
            if (
                req_cap not in ("fs.write", "fs.read", "fs.mkdir", "fs.list", "fs.update", "project.inspect", "test.run", "cmd.run", "memory.store", "memory.retrieve", "capability.develop", "model.query", "chat.respond")
                and self.registry.has(req_cap)
                and not any(a.capability_id == req_cap for a in actions)
            ):
                params: Dict[str, Any] = {}
                if intent.target_artifacts:
                    params["path"] = intent.target_artifacts[0]
                else:
                    params["text"] = instruction
                sid = next_step_id()
                actions.append(
                    PlannedAction(
                        step_id=sid,
                        description=f"Invoke capability '{req_cap}'",
                        capability_id=req_cap,
                        parameters=params,
                    )
                )
                requirements.append(
                    TaskRequirement(
                        requirement_id=f"req-out-{sid}",
                        description=f"Capability '{req_cap}' produced output",
                        category="output_present",
                        target=sid,
                    )
                )

        if not requirements and actions:
            requirements = self._infer_requirements_from_actions(actions)

        return actions, requirements

    def _infer_requirements_from_actions(
        self, actions: List[PlannedAction]
    ) -> List[TaskRequirement]:
        reqs: List[TaskRequirement] = []
        for idx, act in enumerate(actions):
            cid = act.capability_id
            p = act.parameters
            if cid == "fs.write" and p.get("path"):
                path = str(p["path"])
                reqs.append(
                    TaskRequirement(
                        requirement_id=f"req-{idx + 1}-exists",
                        description=f"File '{path}' exists",
                        category="file_exists",
                        target=path,
                    )
                )
                if path.endswith(".py"):
                    reqs.append(
                        TaskRequirement(
                            requirement_id=f"req-{idx + 1}-compiles",
                            description=f"Python file '{path}' compiles cleanly",
                            category="code_compiles",
                            target=path,
                        )
                    )
            elif cid == "fs.mkdir" and p.get("path"):
                reqs.append(
                    TaskRequirement(
                        requirement_id=f"req-{idx + 1}-dir",
                        description=f"Directory '{p['path']}' exists",
                        category="directory_exists",
                        target=str(p["path"]),
                    )
                )
            elif cid == "test.run":
                reqs.append(
                    TaskRequirement(
                        requirement_id=f"req-{idx + 1}-tests",
                        description="Automated tests pass",
                        category="tests_pass",
                        target=p.get("target", "tests"),
                    )
                )
            elif cid == "cmd.run":
                reqs.append(
                    TaskRequirement(
                        requirement_id=f"req-{idx + 1}-cmd",
                        description="Command executes with exit code 0",
                        category="command_succeeds",
                    )
                )
            elif cid == "capability.develop" and p.get("capability_id"):
                reqs.append(
                    TaskRequirement(
                        requirement_id=f"req-{idx + 1}-cap",
                        description=f"Capability '{p['capability_id']}' is operational",
                        category="capability_operational",
                        target=str(p["capability_id"]),
                    )
                )
            elif cid == "memory.store" and p.get("key"):
                reqs.append(
                    TaskRequirement(
                        requirement_id=f"req-{idx + 1}-mem",
                        description=f"Memory '{p['key']}' stored in SQLite",
                        category="memory_stored",
                        target=str(p["key"]),
                        expected=p.get("content"),
                    )
                )
            else:
                reqs.append(
                    TaskRequirement(
                        requirement_id=f"req-{idx + 1}-out",
                        description=f"Step '{act.step_id}' ({cid}) produced output",
                        category="output_present",
                        target=act.step_id,
                    )
                )
        return reqs

    @staticmethod
    def _extract_file_content_from_instruction(
        instruction: str, target_file: str
    ) -> Tuple[str, Optional[str]]:
        # 1. Check for fenced code block
        if "```" in instruction:
            parts = instruction.split("```")
            if len(parts) >= 3:
                block = parts[1]
                first_nl = block.find("\n")
                if first_nl != -1 and block[:first_nl].strip().isalpha():
                    block = block[first_nl + 1 :]
                func_m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", block)
                return block, (func_m.group(1) if func_m else None)

        # 2. Check for explicit content in quotes after 'with content' / 'containing'
        content_m = re.search(
            r"(?:with content|containing)\s+['\"]([^'\"]+)['\"]",
            instruction,
            re.IGNORECASE,
        )
        if content_m:
            text = content_m.group(1)
            if not text.endswith("\n"):
                text += "\n"
            return text, None

        # 3. If target is a Python file and instruction mentions a function name
        if target_file.endswith(".py"):
            fn_match = re.search(
                r"function\s+['\"`]?([a-zA-Z_][a-zA-Z0-9_]*)['\"`]?",
                instruction,
                re.IGNORECASE,
            )
            func_name = fn_match.group(1) if fn_match else "run_task"
            code = (
                f'"""Generated module {target_file}."""\n\n\n'
                f"def {func_name}(value=1):\n"
                f'    """Execute {func_name}."""\n'
                f"    return value\n\n\n"
                f'if __name__ == "__main__":\n'
                f"    print({func_name}())\n"
            )
            return code, func_name

        if target_file.endswith(".json"):
            return json.dumps({"status": "ok", "artifact": target_file}, indent=2) + "\n", None

        return f"Generated by Agent 911 for instruction: {instruction}\n", None

    def _apply_experience_to_plan(
        self,
        task: TaskState,
        experiences: List[Dict[str, Any]],
        memories: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Use relevant historical experience and learned operational policies to improve
        the current task plan and avoid repeating known mistakes (Section 5 & Section 7).
        """
        # 1. Check learned operational policy memories (e.g. from ImprovementEngine)
        policy_create_parents = False
        for mem in memories or []:
            if (
                mem.get("key") == "policy.fs_write.create_parents"
                and str(mem.get("content", "")).lower() in ("true", "1", "yes")
            ):
                policy_create_parents = True
                break

        # 2. Check if past experiences recorded a remedy for ParentDirectoryNotFoundError
        has_parent_dir_remedy = False
        remedy_exp_ids: List[str] = []
        for exp in experiences or []:
            failures_str = " ".join(exp.get("unexpected_failures", [])).lower()
            remedies_str = " ".join(exp.get("successful_remedies", [])).lower()
            if "parentdirectorynotfounderror" in failures_str or "parent directory" in remedies_str:
                has_parent_dir_remedy = True
                if exp.get("experience_id"):
                    remedy_exp_ids.append(str(exp["experience_id"]))

        if has_parent_dir_remedy or policy_create_parents:
            for action in task.planned_actions:
                if action.capability_id == "fs.write":
                    rel_path = action.parameters.get("path", "")
                    if "/" in str(rel_path) and "create_parents" not in action.parameters:
                        action.parameters["create_parents"] = True
                        source_label = (
                            "learned policy 'policy.fs_write.create_parents'"
                            if policy_create_parents and not has_parent_dir_remedy
                            else "prior experience remedy"
                        )
                        task.decisions.append(
                            f"Applied {source_label}: enabled create_parents=True on "
                            f"fs.write for nested path '{rel_path}' to avoid ParentDirectoryNotFoundError."
                        )

    def _finalize_plan_and_strategy(
        self, task: TaskState, experiences: List[Dict[str, Any]]
    ) -> None:
        """
        Enforce subtask bounds, populate expected_outputs and dependencies, and select
        the initial ProblemSolvingStrategy with evidence-based rationale (Sections 3 & 4).
        """
        max_subtasks = getattr(self.config, "max_subtasks_per_task", 20)
        if len(task.planned_actions) > max_subtasks:
            task.planned_actions = task.planned_actions[:max_subtasks]
            task.decisions.append(
                f"Bounded planned subtasks to configured max_subtasks_per_task={max_subtasks}."
            )

        # Populate expected_outputs and infer explicit step dependencies where missing
        mkdir_steps: Dict[str, str] = {}
        write_step_ids: List[str] = []
        for act in task.planned_actions:
            if not act.expected_outputs:
                if act.capability_id in ("fs.write", "fs.update") and act.parameters.get("path"):
                    act.expected_outputs = [str(act.parameters["path"])]
                elif act.capability_id == "fs.mkdir" and act.parameters.get("path"):
                    act.expected_outputs = [str(act.parameters["path"])]
                else:
                    act.expected_outputs = [f"{act.step_id}_output"]

            if act.capability_id == "fs.mkdir" and act.parameters.get("path"):
                mkdir_steps[str(act.parameters["path"]).rstrip("/")] = act.step_id
            elif act.capability_id in ("fs.write", "fs.update"):
                write_step_ids.append(act.step_id)
                rel_p = str(act.parameters.get("path", ""))
                parent_p = str(Path(rel_p).parent)
                if parent_p in mkdir_steps:
                    dep_id = mkdir_steps[parent_p]
                    if dep_id not in act.depends_on and dep_id != act.step_id:
                        act.depends_on.append(dep_id)
            elif act.capability_id == "test.run" and write_step_ids:
                for wid in write_step_ids:
                    if wid not in act.depends_on and wid != act.step_id:
                        act.depends_on.append(wid)

        avail_cap_ids = {
            c["identifier"]
            for c in self.registry.list_capabilities()
            if c["availability"] == CapabilityAvailability.AVAILABLE.value
        }
        strategy, rationale = self.strategy_selector.select_initial_strategy(
            task=task,
            intent=task.intent,
            available_capability_ids=avail_cap_ids,
            relevant_experiences=experiences,
            is_resumption=False,
        )
        task.decisions.append(f"Selected strategy {strategy.value}: {rationale}")

    def revise_plan_with_new_evidence(
        self,
        task: TaskState,
        new_pending_actions: List[PlannedAction],
        reason: str,
    ) -> None:
        """
        Dynamically revise remaining PENDING steps in a task plan when new information
        or failure diagnosis changes the approach, while strictly preserving the original
        user instruction, interpreted objective, and constraints (Section 3.2).
        """
        completed_or_running = [
            a for a in task.planned_actions if a.status != ActionStatus.PENDING
        ]
        max_subtasks = getattr(self.config, "max_subtasks_per_task", 20)
        combined = completed_or_running + list(new_pending_actions)
        task.planned_actions = combined[:max_subtasks]
        task.decisions.append(f"Revised task plan based on new evidence: {reason}")
        task.refresh_action_lists()
