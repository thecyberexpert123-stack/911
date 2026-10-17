"""Conversational agent capability allowing Agent 911 to chat naturally while sharing memory and workspace context."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any, Dict, List, Optional

from agent911.capabilities.base import Capability, CapabilityContext
from agent911.model_provider import ModelProviderError
from agent911.models import (
    CapabilityAvailability,
    CapabilityResult,
    MemoryCategory,
    MemoryConfidence,
)


_SAFE_MATH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _try_eval_math(expr_text: str) -> Optional[float | int]:
    """Safely evaluate a pure arithmetic expression using Python AST."""
    cleaned = expr_text.strip().rstrip("?.!")
    cleaned = re.sub(
        r"^(?:what\s+is|calculate|compute|solve|evaluate)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = cleaned.replace("^", "**")
    if not re.match(r"^[0-9\.\+\-\*\/\%\(\)\s]+$", cleaned):
        return None
    if not any(op in cleaned for op in ("+", "-", "*", "/", "%")):
        return None
    try:
        tree = ast.parse(cleaned, mode="eval")

        def _eval_node(node: ast.AST) -> float | int:
            if isinstance(node, ast.Expression):
                return _eval_node(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_MATH_OPS:
                left = _eval_node(node.left)
                right = _eval_node(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                    raise ValueError("Exponent too large")
                return _SAFE_MATH_OPS[type(node.op)](left, right)
            if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_MATH_OPS:
                return _SAFE_MATH_OPS[type(node.op)](_eval_node(node.operand))
            raise ValueError("Unsupported AST node")

        val = _eval_node(tree)
        if isinstance(val, float) and val.is_integer():
            return int(val)
        return round(val, 6) if isinstance(val, float) else val
    except Exception:
        return None


class ChatRespondCapability(Capability):
    """
    Conversational capability ('chat.respond') that enables Agent 911 to talk naturally
    with users like a normal chatbot while retaining full access to persistent SQLite
    memories, recent task history, workspace state, and configured AI models.
    """

    identifier = "chat.respond"
    name = "Conversational Agent Responder"
    purpose = (
        "Hold natural multi-turn conversations, answer questions, remember user facts, "
        "and explain workspace/task status."
    )
    input_schema = {
        "required": ["message"],
        "optional": ["conversation_context"],
        "types": {"message": "string"},
    }
    output_schema = {"fields": ["reply", "source", "stored_facts"]}
    operational_boundaries = "workspace_confined"

    def check_availability(
        self, config: Optional[Any] = None
    ) -> CapabilityAvailability:
        return CapabilityAvailability.AVAILABLE

    def execute(
        self, params: Dict[str, Any], context: CapabilityContext
    ) -> CapabilityResult:
        message = str(params.get("message", "")).strip()
        if not message:
            return CapabilityResult(
                capability_id=self.identifier,
                success=False,
                error="Parameter 'message' must be a non-empty string.",
                error_type="InputValidationError",
            )

        # 1. Extract and persist any conversational facts the user shared (e.g., "My name is ...")
        stored_facts = self._extract_and_store_user_facts(message, context)

        # 2. Gather memory and recent task context from SQLite
        factual_memories = context.memory_store.retrieve_memories(
            category=MemoryCategory.FACTUAL, limit=15
        )
        recent_tasks = context.memory_store.list_tasks(limit=5)
        caps = context.registry.list_capabilities()

        # 3. Try live AI model generation if a model provider is configured and reachable
        if self._should_try_live_model(context):
            try:
                mem_summary = {m.key: m.content for m in factual_memories}
                sys_prompt = (
                    "You are Agent 911, an intelligent, friendly conversational AI agent and "
                    "software engineering assistant. You can both chat naturally with the user "
                    "and execute real workspace instructions (creating/editing files, running "
                    "commands and tests, managing SQLite memory, and growing new capabilities). "
                    "Reply naturally, clearly, and concisely."
                )
                resp = context.model_interface.generate(
                    system_prompt=sys_prompt,
                    user_prompt=message,
                    context_payload={
                        "known_user_and_project_facts": mem_summary,
                        "recent_tasks": recent_tasks[:3],
                        "stored_facts_this_turn": stored_facts,
                    },
                    expect_json=False,
                )
                reply_text = resp.text.strip()
                if reply_text:
                    return CapabilityResult(
                        capability_id=self.identifier,
                        success=True,
                        data={
                            "reply": reply_text,
                            "source": f"{resp.provider}:{resp.model_id}",
                            "stored_facts": stored_facts,
                        },
                    )
            except ModelProviderError:
                # Fall back seamlessly to context-aware local conversational responder
                pass

        # 4. Context-aware local conversational response
        reply_text = self._build_contextual_reply(
            message=message,
            stored_facts=stored_facts,
            factual_memories=factual_memories,
            recent_tasks=recent_tasks,
            capabilities=caps,
            context=context,
        )
        return CapabilityResult(
            capability_id=self.identifier,
            success=True,
            data={
                "reply": reply_text,
                "source": "agent911_conversational_engine",
                "stored_facts": stored_facts,
            },
        )

    @staticmethod
    def _should_try_live_model(context: CapabilityContext) -> bool:
        if context.model_interface.deterministic_responder is not None:
            return False
        if context.config.model.provider == "local_only":
            return False
        if context.config.model.resolve_api_key(context.config.credential_store) or context.config.model.provider == "ollama":
            return True
        return False

    @staticmethod
    def _extract_and_store_user_facts(
        message: str, context: CapabilityContext
    ) -> List[Dict[str, str]]:
        stored: List[Dict[str, str]] = []

        # Pattern 1: "My name is X" / "Call me X"
        name_m = re.search(
            r"\b(?:my\s+name\s+is|call\s+me)\s+([A-Za-z][A-Za-z0-9_\- ]{0,30}?)(?:\s+and\b|[\.,!;\?]|$)",
            message,
            re.IGNORECASE,
        )
        if name_m:
            user_name = name_m.group(1).strip()
            if user_name.lower() not in ("what", "who", "agent", "agent 911"):
                context.memory_store.store_memory(
                    category=MemoryCategory.FACTUAL,
                    key="user.name",
                    content=user_name,
                    task_id=context.task_id,
                    source="chat_conversation",
                    confidence=MemoryConfidence.VERIFIED,
                )
                stored.append({"key": "user.name", "value": user_name})

        # Pattern 2: "My favorite <thing> is <val>" or "My <attr> is <val>"
        pref_m = re.search(
            r"\bmy\s+(favorite\s+[a-zA-Z0-9_ ]+|role|profession|job|city|location|project)\s+is\s+([^.\n!?]+)",
            message,
            re.IGNORECASE,
        )
        if pref_m:
            attr = re.sub(r"\s+", "_", pref_m.group(1).strip().lower())
            val = pref_m.group(2).strip()
            key = f"user.{attr}"
            context.memory_store.store_memory(
                category=MemoryCategory.FACTUAL,
                key=key,
                content=val,
                task_id=context.task_id,
                source="chat_conversation",
                confidence=MemoryConfidence.VERIFIED,
            )
            stored.append({"key": key, "value": val})

        return stored

    def _build_contextual_reply(
        self,
        message: str,
        stored_facts: List[Dict[str, str]],
        factual_memories: List[Any],
        recent_tasks: List[Dict[str, Any]],
        capabilities: List[Dict[str, Any]],
        context: CapabilityContext,
    ) -> str:
        lower = message.lower().strip()
        mem_map = {m.key: m.content for m in factual_memories}
        user_name = mem_map.get("user.name")
        greeting_name = f", {user_name}" if user_name else ""

        # If the user just introduced themselves or shared a personal fact
        if stored_facts:
            fact_descriptions = []
            for f in stored_facts:
                if f["key"] == "user.name":
                    fact_descriptions.append(f"your name is **{f['value']}**")
                else:
                    label = f["key"].replace("user.", "").replace("_", " ")
                    fact_descriptions.append(f"your {label} is **{f['value']}**")
            joined = " and ".join(fact_descriptions)
            return (
                f"Nice to meet you{greeting_name}! I've noted in my persistent memory that {joined}. "
                f"How can I help you today? We can chat, answer questions, or execute tasks in your workspace."
            )

        # 1. Check for arithmetic/math questions
        math_val = _try_eval_math(message)
        if math_val is not None:
            return f"The result is **{math_val}**."

        # 2. Questions asking about the user's name or stored facts
        if any(
            p in lower
            for p in (
                "what is my name",
                "what's my name",
                "who am i",
                "do you know my name",
                "remember my name",
            )
        ):
            if user_name:
                return f"Your name is **{user_name}**! I have it saved in my persistent SQLite memory."
            return (
                "You haven't told me your name yet! Say something like *\"My name is Alex\"* "
                "and I'll remember it across sessions."
            )

        if any(
            p in lower
            for p in (
                "what do you remember",
                "what do you know about me",
                "what is in your memory",
                "show memories",
            )
        ):
            if not mem_map:
                return "I don't have any factual memories stored yet. You can tell me facts in chat or ask me to remember anything!"
            items = [f"- **{k}**: {v}" for k, v in list(mem_map.items())[:10]]
            return "Here is what I currently have stored in my factual memory:\n" + "\n".join(items)

        # 3. Greetings & small talk
        if re.match(
            r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening)|howdy|yo)\b",
            lower,
        ) or lower in ("how are you", "how are you?", "how are you doing", "how's it going", "whats up", "what's up"):
            return (
                f"Hello{greeting_name}! I'm **Agent 911**, your conversational AI agent. "
                f"I'm running smoothly with **{len(capabilities)} operational capabilities** and persistent SQLite memory.\n\n"
                "You can talk to me like a normal chatbot (ask questions, share context, or discuss code) "
                "or give me real instructions to execute in the workspace—such as:\n"
                "- *\"Create a Python file `app.py` with a function `main`\"*\n"
                "- *\"Inspect the project structure\"*\n"
                "- *\"Run the test suite\"*\n"
                "- *\"Develop capability `text.metrics` to count words and lines\"*"
            )

        # 4. Thanks / appreciation / goodbye
        if any(w in lower for w in ("thank you", "thanks", "thx", "appreciate it")):
            return f"You're very welcome{greeting_name}! Let me know whenever you want to chat or run another task."

        if lower in ("bye", "goodbye", "see you", "cya"):
            return f"Goodbye{greeting_name}! All your memories and task history are safely persisted in SQLite whenever you return."

        # 5. Questions about Agent 911's identity, capabilities, or how it works
        if any(
            p in lower
            for p in (
                "who are you",
                "what are you",
                "what can you do",
                "what are your capabilities",
                "help",
                "how do you work",
                "what tools do you have",
            )
        ):
            cap_ids = [c["identifier"] for c in capabilities if c["availability"] == "AVAILABLE"]
            return (
                f"I am **Agent 911**, a conversational and autonomous AI agent. I work in two seamlessly integrated modes:\n\n"
                f"1. **Conversational Mode**: We can chat naturally, brainstorm, do math, and I automatically remember facts you share in SQLite.\n"
                f"2. **Autonomous Execution Mode**: When you ask me to create/edit files, inspect code, run commands/tests, or build new tools, "
                f"I plan the steps, execute them using my **{len(cap_ids)} capabilities** (`{', '.join(cap_ids[:10])}`, ...), and validate every result using **Verification** and **Live Run Testing**."
            )

        # 6. Questions about active model / Groq / configuration status
        if any(
            p in lower
            for p in (
                "what model",
                "which model",
                "groq",
                "active model",
                "provider status",
            )
        ):
            return (
                f"Currently, my active model profile is **`{context.config.active_model}`** "
                f"(`{context.config.model.provider}` -> `{context.config.model.model_id}`) "
                f"with fallback model **`{context.config.fallback_model or 'none'}`**. "
                f"When external cloud endpoints are unreachable from the sandbox network, I automatically use my built-in conversational and planning engine so you never lose functionality."
            )

        # 7. Questions about recent tasks / history
        if any(
            p in lower
            for p in (
                "recent task",
                "last task",
                "what did we do",
                "task history",
                "what have you done",
            )
        ):
            other_tasks = [t for t in recent_tasks if t["task_id"] != context.task_id][:4]
            if not other_tasks:
                return "We haven't executed any other tasks in this workspace yet. Try asking me to inspect the project or create a file!"
            lines = [
                f"- **`{t['task_id']}`** ({t['outcome']}): *{t['original_instruction']}*"
                for t in other_tasks
            ]
            return "Here are the most recent tasks I've executed:\n" + "\n".join(lines)

        # 8. Programming / technical explanations or general questions
        if "python" in lower or "function" in lower or "code" in lower:
            return (
                f"I'd be happy to help with Python or coding{greeting_name}! "
                f"If you'd like me to write and live-test a real Python script in the workspace, just tell me the filename and what it should do—for example:\n"
                f"*\"Write a Python file `calculator.py` with a function `add`\"*."
            )

        # 9. General friendly conversational fallback that checks if any memory matches keywords
        matched_mems = context.memory_store.retrieve_memories(query=message, limit=3)
        if matched_mems:
            mem_lines = [f"- **{m.key}**: {m.content}" for m in matched_mems]
            return (
                f"Here's what I found in my memory related to your message:\n"
                + "\n".join(mem_lines)
                + "\n\nWould you like me to run an action in the workspace or update this information?"
            )

        return (
            f"I hear you{greeting_name}! (You said: *\"{message}\"*)\n\n"
            "I can chat with you, remember facts you tell me (like *\"My name is ...\"* or *\"Remember that key = value\"*), "
            "or execute concrete actions in your workspace (like reading/writing files, inspecting the project, running tests, or developing new capabilities). "
            "What would you like to do next?"
        )
