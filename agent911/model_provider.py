"""AI model interface, provider adapters, genuine connection testing, and runtime model switching."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from agent911.config import (
    AgentConfig,
    ConfigurationError,
    CredentialStore,
    ModelConfig,
    redact_secrets,
)


class ProviderConnectivityStatus(str, Enum):
    """Honest connectivity states for the configured AI model provider."""

    CONNECTED = "CONNECTED"
    MISSING_CREDENTIALS = "MISSING_CREDENTIALS"
    AUTH_ERROR = "AUTH_ERROR"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    UNREACHABLE = "UNREACHABLE"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    DETERMINISTIC_TEST_DOUBLE = "DETERMINISTIC_TEST_DOUBLE"
    LOCAL_ONLY = "LOCAL_ONLY"


class ModelProviderError(RuntimeError):
    """Raised when an AI model provider fails to respond or returns an error."""

    def __init__(
        self,
        message: str,
        error_type: str = "provider_error",
        status_code: Optional[int] = None,
        retryable: bool = False,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        self.retryable = retryable
        self.original_cause = cause


class MalformedModelOutputError(ModelProviderError):
    """Raised when the model returns output that cannot be parsed as required."""

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message, error_type="malformed_output", retryable=False)
        self.raw_output = raw_output


class UnsupportedModelFeatureError(ModelProviderError):
    """Raised when a model or provider does not support a requested feature."""

    def __init__(self, message: str, feature: str) -> None:
        super().__init__(message, error_type="unsupported_feature", retryable=False)
        self.feature = feature


class RequestCancelledError(ModelProviderError):
    """Raised when a model request is cancelled before or during execution."""

    def __init__(self, message: str = "Model request was cancelled.") -> None:
        super().__init__(message, error_type="cancelled", retryable=False)


@dataclass
class ModelResponse:
    """Structured response returned by the AI model interface."""

    content: str
    structured_data: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    provider: str = ""
    model_id: str = ""
    profile_id: str = ""
    provider_reached: bool = False
    is_simulated: bool = False
    used_fallback_model: bool = False
    retry_attempts: int = 0
    latency_ms: float = 0.0
    usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectivityReport:
    """
    Granular result of testing an AI provider/model connection.
    Distinguishes configuration validation, network reachability, authentication,
    model acceptance, and successful inference.
    """

    status: ProviderConnectivityStatus
    provider: str
    model_id: str
    endpoint: str
    provider_reached: bool
    is_live_provider: bool
    details: str
    latency_ms: float = 0.0
    config_valid: bool = True
    endpoint_reachable: bool = False
    auth_valid: Optional[bool] = None
    model_accepted: Optional[bool] = None
    inference_succeeded: bool = False
    conclusive: bool = False
    error_category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "provider": self.provider,
            "model_id": self.model_id,
            "endpoint": self.endpoint,
            "provider_reached": self.provider_reached,
            "is_live_provider": self.is_live_provider,
            "details": self.details,
            "latency_ms": self.latency_ms,
            "config_valid": self.config_valid,
            "endpoint_reachable": self.endpoint_reachable,
            "auth_valid": self.auth_valid,
            "model_accepted": self.model_accepted,
            "inference_succeeded": self.inference_succeeded,
            "conclusive": self.conclusive,
            "error_category": self.error_category,
        }


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from raw model text or fenced markdown blocks."""
    cleaned = text.strip()
    if not cleaned:
        raise MalformedModelOutputError("Model returned empty output.", raw_output=text)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        raise MalformedModelOutputError(
            "Expected a JSON object at top level, got another JSON type.", raw_output=text
        )
    except json.JSONDecodeError:
        pass

    if "```" in cleaned:
        parts = cleaned.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i].strip()
            if block.lower().startswith("json"):
                block = block[4:].strip()
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            raise MalformedModelOutputError(
                f"Failed to parse JSON object from model response: {exc}", raw_output=text
            ) from exc

    raise MalformedModelOutputError(
        "No valid JSON object found in model output.", raw_output=text
    )


# ---------------------------------------------------------------------------
# Provider Adapters
# ---------------------------------------------------------------------------


class ProviderAdapter(ABC):
    """Documented adapter interface for provider-specific request formatting and parsing."""

    adapter_type: str = ""
    requires_api_key: bool = True

    @abstractmethod
    def build_http_request(
        self,
        model_cfg: ModelConfig,
        system_prompt: str,
        user_prompt: str,
        api_key: Optional[str],
        expect_json: bool,
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """Return (endpoint_url, headers_dict, json_body_dict)."""
        raise NotImplementedError

    @abstractmethod
    def parse_http_response(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Return (content_text, tool_calls, usage_dict)."""
        raise NotImplementedError


class OpenAICompatibleAdapter(ProviderAdapter):
    adapter_type = "openai_compatible"
    requires_api_key = True

    def build_http_request(
        self,
        model_cfg: ModelConfig,
        system_prompt: str,
        user_prompt: str,
        api_key: Optional[str],
        expect_json: bool,
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if model_cfg.custom_headers:
            headers.update(model_cfg.custom_headers)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body: Dict[str, Any] = {
            "model": model_cfg.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": model_cfg.temperature,
            "max_tokens": model_cfg.max_tokens,
        }
        if expect_json and model_cfg.supports_structured_output is not False:
            body["response_format"] = {"type": "json_object"}
        return model_cfg.endpoint, headers, body

    def parse_http_response(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not choices or not isinstance(choices, list):
            raise ModelProviderError(
                "OpenAI-compatible response missing 'choices' array.",
                error_type="invalid_response",
            )
        msg = choices[0].get("message", {})
        content_text = msg.get("content") or ""
        tool_calls = msg["tool_calls"] if isinstance(msg.get("tool_calls"), list) else []
        usage = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}
        return content_text, tool_calls, usage


class AnthropicAdapter(ProviderAdapter):
    adapter_type = "anthropic"
    requires_api_key = True

    def build_http_request(
        self,
        model_cfg: ModelConfig,
        system_prompt: str,
        user_prompt: str,
        api_key: Optional[str],
        expect_json: bool,
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if model_cfg.custom_headers:
            headers.update(model_cfg.custom_headers)
        if api_key:
            headers["x-api-key"] = api_key
        body: Dict[str, Any] = {
            "model": model_cfg.model_id,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": model_cfg.temperature,
            "max_tokens": model_cfg.max_tokens,
        }
        return model_cfg.endpoint, headers, body

    def parse_http_response(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        blocks = payload.get("content") if isinstance(payload, dict) else None
        if not blocks or not isinstance(blocks, list):
            raise ModelProviderError(
                "Anthropic response missing 'content' blocks.",
                error_type="invalid_response",
            )
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for blk in blocks:
            if isinstance(blk, dict):
                if blk.get("type") == "text":
                    text_parts.append(blk.get("text", ""))
                elif blk.get("type") == "tool_use":
                    tool_calls.append(blk)
        usage = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}
        return "\n".join(text_parts), tool_calls, usage


class OllamaAdapter(ProviderAdapter):
    adapter_type = "ollama"
    requires_api_key = False

    def build_http_request(
        self,
        model_cfg: ModelConfig,
        system_prompt: str,
        user_prompt: str,
        api_key: Optional[str],
        expect_json: bool,
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if model_cfg.custom_headers:
            headers.update(model_cfg.custom_headers)
        body: Dict[str, Any] = {
            "model": model_cfg.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if expect_json and model_cfg.supports_structured_output is not False:
            body["format"] = "json"
        return model_cfg.endpoint, headers, body

    def parse_http_response(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        msg = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(msg, dict) or "content" not in msg:
            raise ModelProviderError(
                "Ollama response missing 'message.content'.",
                error_type="invalid_response",
            )
        content_text = str(msg.get("content") or "")
        tool_calls = msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else []
        usage = {
            k: payload[k]
            for k in ("prompt_eval_count", "eval_count", "total_duration")
            if k in payload
        }
        return content_text, tool_calls, usage


_ADAPTER_REGISTRY: Dict[str, Type[ProviderAdapter]] = {
    "openai_compatible": OpenAICompatibleAdapter,
    "anthropic": AnthropicAdapter,
    "ollama": OllamaAdapter,
}


def register_provider_adapter(
    adapter_type: str, adapter_cls: Type[ProviderAdapter]
) -> None:
    """Register a custom ProviderAdapter class for extending provider support."""
    if not issubclass(adapter_cls, ProviderAdapter):
        raise TypeError("adapter_cls must inherit from ProviderAdapter.")
    _ADAPTER_REGISTRY[adapter_type] = adapter_cls


def get_provider_adapter(adapter_type: str) -> Optional[ProviderAdapter]:
    cls = _ADAPTER_REGISTRY.get(adapter_type)
    return cls() if cls is not None else None


# ---------------------------------------------------------------------------
# Unified Model Interface
# ---------------------------------------------------------------------------


class ModelInterface:
    """
    Unified AI model interface.
    Communicates with real HTTP model endpoints via provider adapters, supports runtime
    model switching and fallback routing, validates feature support, and performs
    genuine connection tests.
    """

    def __init__(
        self,
        config: ModelConfig,
        max_context_chars: int = 16000,
        deterministic_responder: Optional[
            Callable[[str, Dict[str, Any]], str | Dict[str, Any]]
        ] = None,
        credential_store: Optional[CredentialStore] = None,
        agent_config: Optional[AgentConfig] = None,
    ) -> None:
        self.config = config
        self.max_context_chars = max_context_chars
        self.deterministic_responder = deterministic_responder
        self.credential_store = credential_store
        self.agent_config = agent_config
        self._cache: Dict[str, ModelResponse] = {}
        self.last_error: Optional[str] = None

    def switch_active_model(
        self, profile_id: str, set_as_default: bool = False
    ) -> ModelConfig:
        """
        Switch the active model at runtime without restarting the application.
        Updates both AgentConfig and this ModelInterface's active ModelConfig.
        """
        if self.agent_config is None:
            raise ConfigurationError(
                "Cannot switch model profile because no AgentConfig registry is attached."
            )
        self.agent_config.select_active_model(profile_id, set_as_default=set_as_default)
        self.config = self.agent_config.model
        return self.config

    def check_connectivity(
        self, target_profile_id: Optional[str] = None
    ) -> ConnectivityReport:
        """
        Perform a genuine connection test against the active or specified model profile.
        Clearly distinguishes configuration validation, network reachability,
        authentication validation, model availability, and inference completion.
        """
        target_cfg = self.config
        if target_profile_id and self.agent_config is not None:
            try:
                target_cfg = self.agent_config.resolve_model_Config_for_profile(
                    target_profile_id
                )
            except ConfigurationError as exc:
                return ConnectivityReport(
                    status=ProviderConnectivityStatus.UNREACHABLE,
                    provider="unknown",
                    model_id=target_profile_id,
                    endpoint="",
                    provider_reached=False,
                    is_live_provider=False,
                    details=str(exc),
                    config_valid=False,
                    conclusive=True,
                    error_category="configuration_error",
                )

        try:
            target_cfg.validate()
        except ConfigurationError as exc:
            return ConnectivityReport(
                status=ProviderConnectivityStatus.UNREACHABLE,
                provider=target_cfg.provider,
                model_id=target_cfg.model_id,
                endpoint=target_cfg.endpoint,
                provider_reached=False,
                is_live_provider=False,
                details=str(exc),
                config_valid=False,
                conclusive=True,
                error_category="configuration_error",
            )

        provider = target_cfg.provider
        if provider == "deterministic" or self.deterministic_responder is not None:
            return ConnectivityReport(
                status=ProviderConnectivityStatus.DETERMINISTIC_TEST_DOUBLE,
                provider=provider,
                model_id=target_cfg.model_id,
                endpoint="in-memory://deterministic",
                provider_reached=False,
                is_live_provider=False,
                details=(
                    "Using deterministic test double for isolated testing. "
                    "No external AI provider was contacted."
                ),
                config_valid=True,
                endpoint_reachable=False,
                auth_valid=None,
                model_accepted=None,
                inference_succeeded=False,
                conclusive=False,
            )

        if provider == "local_only":
            return ConnectivityReport(
                status=ProviderConnectivityStatus.LOCAL_ONLY,
                provider=provider,
                model_id=target_cfg.model_id,
                endpoint="local://heuristic",
                provider_reached=False,
                is_live_provider=False,
                details="Configured for local structured execution without external AI provider.",
                config_valid=True,
                endpoint_reachable=False,
                auth_valid=None,
                model_accepted=None,
                inference_succeeded=False,
                conclusive=False,
            )

        adapter = get_provider_adapter(provider)
        api_key = target_cfg.resolve_api_key(self.credential_store)
        if adapter and adapter.requires_api_key and not api_key:
            return ConnectivityReport(
                status=ProviderConnectivityStatus.MISSING_CREDENTIALS,
                provider=provider,
                model_id=target_cfg.model_id,
                endpoint=target_cfg.endpoint,
                provider_reached=False,
                is_live_provider=False,
                details=(
                    f"Missing API credentials (checked CredentialStore for '{target_cfg.provider_id}' "
                    f"and env var '{target_cfg.api_key_env}')."
                ),
                config_valid=True,
                endpoint_reachable=False,
                auth_valid=None,
                model_accepted=None,
                inference_succeeded=False,
                conclusive=False,
                error_category="missing_credentials",
            )

        t0 = time.monotonic()
        try:
            resp, _ = self._execute_http_request_with_config(
                model_cfg=target_cfg,
                system_prompt="Respond with OK.",
                user_prompt="ping",
                api_key=api_key,
                expect_json=False,
            )
            elapsed = (time.monotonic() - t0) * 1000.0
            return ConnectivityReport(
                status=ProviderConnectivityStatus.CONNECTED,
                provider=provider,
                model_id=target_cfg.model_id,
                endpoint=target_cfg.endpoint,
                provider_reached=resp.provider_reached,
                is_live_provider=resp.provider_reached and not resp.is_simulated,
                details=f"Successfully reached provider '{provider}' and verified model '{target_cfg.model_id}' ({elapsed:.1f} ms).",
                latency_ms=elapsed,
                config_valid=True,
                endpoint_reachable=True,
                auth_valid=True,
                model_accepted=True,
                inference_succeeded=True,
                conclusive=True,
            )
        except ModelProviderError as exc:
            elapsed = (time.monotonic() - t0) * 1000.0
            safe_msg = str(redact_secrets(str(exc), [api_key] if api_key else []))
            self.last_error = safe_msg

            if exc.error_type == "auth_error":
                return ConnectivityReport(
                    status=ProviderConnectivityStatus.AUTH_ERROR,
                    provider=provider,
                    model_id=target_cfg.model_id,
                    endpoint=target_cfg.endpoint,
                    provider_reached=True,
                    is_live_provider=False,
                    details=safe_msg,
                    latency_ms=elapsed,
                    config_valid=True,
                    endpoint_reachable=True,
                    auth_valid=False,
                    model_accepted=None,
                    inference_succeeded=False,
                    conclusive=True,
                    error_category="auth_error",
                )
            if exc.error_type == "model_not_found":
                return ConnectivityReport(
                    status=ProviderConnectivityStatus.MODEL_NOT_FOUND,
                    provider=provider,
                    model_id=target_cfg.model_id,
                    endpoint=target_cfg.endpoint,
                    provider_reached=True,
                    is_live_provider=False,
                    details=safe_msg,
                    latency_ms=elapsed,
                    config_valid=True,
                    endpoint_reachable=True,
                    auth_valid=True,
                    model_accepted=False,
                    inference_succeeded=False,
                    conclusive=True,
                    error_category="model_not_found",
                )
            if exc.error_type in ("invalid_response", "malformed_output"):
                return ConnectivityReport(
                    status=ProviderConnectivityStatus.INVALID_RESPONSE,
                    provider=provider,
                    model_id=target_cfg.model_id,
                    endpoint=target_cfg.endpoint,
                    provider_reached=True,
                    is_live_provider=False,
                    details=safe_msg,
                    latency_ms=elapsed,
                    config_valid=True,
                    endpoint_reachable=True,
                    auth_valid=True,
                    model_accepted=True,
                    inference_succeeded=False,
                    conclusive=True,
                    error_category=exc.error_type,
                )
            return ConnectivityReport(
                status=ProviderConnectivityStatus.UNREACHABLE,
                provider=provider,
                model_id=target_cfg.model_id,
                endpoint=target_cfg.endpoint,
                provider_reached=False,
                is_live_provider=False,
                details=safe_msg,
                latency_ms=elapsed,
                config_valid=True,
                endpoint_reachable=False,
                auth_valid=None,
                model_accepted=None,
                inference_succeeded=False,
                conclusive=False,
                error_category=exc.error_type,
            )

    def _validate_feature_requirements(
        self,
        model_cfg: ModelConfig,
        require_tool_calling: bool = False,
        require_streaming: bool = False,
        input_type: str = "text",
    ) -> None:
        if require_tool_calling and model_cfg.supports_tool_calling is False:
            raise UnsupportedModelFeatureError(
                f"Model '{model_cfg.model_id}' does not support tool calling.",
                feature="tool_calling",
            )
        if require_streaming and model_cfg.supports_streaming is False:
            raise UnsupportedModelFeatureError(
                f"Model '{model_cfg.model_id}' does not support streaming responses.",
                feature="streaming",
            )
        if (
            input_type != "text"
            and "unknown" not in model_cfg.supported_inputs
            and input_type not in model_cfg.supported_inputs
        ):
            raise UnsupportedModelFeatureError(
                f"Model '{model_cfg.model_id}' does not support input type '{input_type}' "
                f"(supported: {model_cfg.supported_inputs}).",
                feature=f"input:{input_type}",
            )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        context_payload: Optional[Dict[str, Any]] = None,
        expect_json: bool = False,
        use_cache: bool = True,
        model_profile_override: Optional[str] = None,
        require_tool_calling: bool = False,
        require_streaming: bool = False,
        input_type: str = "text",
        cancel_event: Optional[threading.Event] = None,
    ) -> ModelResponse:
        """
        Send a request to the configured AI model (or deterministic test double).
        Supports task-specific model_profile_override, fallback model switching,
        bounded retries, feature checks, and request cancellation.
        """
        if cancel_event is not None and cancel_event.is_set():
            raise RequestCancelledError("Model request cancelled prior to execution.")

        active_cfg = self.config
        if model_profile_override and self.agent_config is not None:
            active_cfg = self.agent_config.resolve_model_Config_for_profile(
                model_profile_override
            )

        self._validate_feature_requirements(
            active_cfg,
            require_tool_calling=require_tool_calling,
            require_streaming=require_streaming,
            input_type=input_type,
        )

        # Respect active model's context_window_tokens if known
        effective_max_chars = self.max_context_chars
        if active_cfg.context_window_tokens:
            effective_max_chars = min(
                effective_max_chars, active_cfg.context_window_tokens * 4
            )

        combined_user = user_prompt
        if context_payload:
            ctx_json = json.dumps(context_payload, default=str)
            if len(ctx_json) > effective_max_chars:
                ctx_json = ctx_json[:effective_max_chars] + "...[truncated]"
            combined_user = f"{user_prompt}\n\nCONTEXT:\n{ctx_json}"

        if len(system_prompt) + len(combined_user) > effective_max_chars * 2:
            combined_user = combined_user[:effective_max_chars]

        cache_key = hashlib.sha256(
            f"{active_cfg.provider}:{active_cfg.model_id}:{system_prompt}:{combined_user}:{expect_json}".encode(
                "utf-8"
            )
        ).hexdigest()

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            resp = self._generate_with_config(
                model_cfg=active_cfg,
                system_prompt=system_prompt,
                combined_user=combined_user,
                context_payload=context_payload,
                user_prompt=user_prompt,
                expect_json=expect_json,
                cancel_event=cancel_event,
            )
            if use_cache:
                self._cache[cache_key] = resp
            return resp
        except ModelProviderError as primary_err:
            self.last_error = str(primary_err)
            # If a compatible fallback model is configured on AgentConfig, attempt fallback
            if (
                self.agent_config is not None
                and self.agent_config.fallback_model
                and self.agent_config.fallback_model != active_cfg.profile_id
                and primary_err.error_type != "cancelled"
            ):
                try:
                    fallback_cfg = self.agent_config.resolve_model_Config_for_profile(
                        self.agent_config.fallback_model
                    )
                    fallback_resp = self._generate_with_config(
                        model_cfg=fallback_cfg,
                        system_prompt=system_prompt,
                        combined_user=combined_user,
                        context_payload=context_payload,
                        user_prompt=user_prompt,
                        expect_json=expect_json,
                        cancel_event=cancel_event,
                    )
                    fallback_resp.used_fallback_model = True
                    return fallback_resp
                except Exception:
                    pass
            raise

    def _generate_with_config(
        self,
        model_cfg: ModelConfig,
        system_prompt: str,
        combined_user: str,
        context_payload: Optional[Dict[str, Any]],
        user_prompt: str,
        expect_json: bool,
        cancel_event: Optional[threading.Event],
    ) -> ModelResponse:
        if model_cfg.provider == "deterministic" or self.deterministic_responder is not None:
            if self.deterministic_responder is None:
                raise ModelProviderError(
                    "Provider is set to 'deterministic' but no deterministic_responder was supplied.",
                    error_type="configuration_error",
                )
            t0 = time.monotonic()
            raw_out = self.deterministic_responder(user_prompt, context_payload or {})
            elapsed = (time.monotonic() - t0) * 1000.0
            if isinstance(raw_out, dict):
                content_str = json.dumps(raw_out)
                structured = raw_out
            else:
                content_str = str(raw_out)
                structured = extract_json_object(content_str) if expect_json else None

            return ModelResponse(
                content=content_str,
                structured_data=structured,
                provider="deterministic",
                model_id=model_cfg.model_id,
                profile_id=model_cfg.profile_id,
                provider_reached=False,
                is_simulated=True,
                latency_ms=elapsed,
            )

        if model_cfg.provider == "local_only":
            raise ModelProviderError(
                "Model provider is configured as 'local_only'; external AI calls are disabled.",
                error_type="local_only_mode",
                retryable=False,
            )

        adapter = get_provider_adapter(model_cfg.provider)
        if adapter is None:
            raise ModelProviderError(
                f"No provider adapter registered for '{model_cfg.provider}'.",
                error_type="configuration_error",
            )

        api_key = model_cfg.resolve_api_key(self.credential_store)
        if adapter.requires_api_key and not api_key:
            raise ModelProviderError(
                f"No API key available for provider '{model_cfg.provider}' "
                f"(expected CredentialStore entry or env var '{model_cfg.api_key_env}').",
                error_type="missing_credentials",
                retryable=False,
            )

        last_error: Optional[ModelProviderError] = None
        attempts = model_cfg.max_retries + 1
        for attempt in range(attempts):
            if cancel_event is not None and cancel_event.is_set():
                raise RequestCancelledError("Model request cancelled during retry cycle.")
            try:
                response, _ = self._execute_http_request_with_config(
                    model_cfg=model_cfg,
                    system_prompt=system_prompt,
                    user_prompt=combined_user,
                    api_key=api_key,
                    expect_json=expect_json,
                )
                response.retry_attempts = attempt
                return response
            except ModelProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt == attempts - 1:
                    raise
                time.sleep(min(0.2 * (attempt + 1), 1.0))

        assert last_error is not None
        raise last_error

    def _execute_http_request_with_config(
        self,
        model_cfg: ModelConfig,
        system_prompt: str,
        user_prompt: str,
        api_key: Optional[str],
        expect_json: bool,
    ) -> Tuple[ModelResponse, int]:
        adapter = get_provider_adapter(model_cfg.provider)
        if adapter is None:
            raise ModelProviderError(
                f"Unsupported provider '{model_cfg.provider}'.",
                error_type="configuration_error",
            )

        endpoint_url, headers, body = adapter.build_http_request(
            model_cfg=model_cfg,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=api_key,
            expect_json=expect_json,
        )

        req = urllib.request.Request(
            endpoint_url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=model_cfg.timeout_seconds) as resp:
                raw_bytes = resp.read()
                status_code = getattr(resp, "status", 200)
        except urllib.error.HTTPError as exc:
            code = exc.code
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            safe_body = str(redact_secrets(err_body, [api_key] if api_key else []))
            if code in (401, 403):
                raise ModelProviderError(
                    f"Authentication failed with HTTP {code}: {safe_body}",
                    error_type="auth_error",
                    status_code=code,
                    retryable=False,
                    cause=exc,
                ) from exc
            if code == 404 or "model_not_found" in safe_body.lower():
                raise ModelProviderError(
                    f"Model '{model_cfg.model_id}' not found or endpoint invalid (HTTP {code}): {safe_body}",
                    error_type="model_not_found",
                    status_code=code,
                    retryable=False,
                    cause=exc,
                ) from exc
            retryable = code in (408, 429, 500, 502, 503, 504)
            raise ModelProviderError(
                f"Provider returned HTTP {code}: {safe_body}",
                error_type="http_error",
                status_code=code,
                retryable=retryable,
                cause=exc,
            ) from exc
        except urllib.error.URLError as exc:
            reason = str(exc.reason)
            is_timeout = "timed out" in reason.lower() or isinstance(
                exc.reason, TimeoutError
            )
            raise ModelProviderError(
                f"Network error contacting '{endpoint_url}': {reason}",
                error_type="timeout" if is_timeout else "network_error",
                retryable=True,
                cause=exc,
            ) from exc
        except TimeoutError as exc:
            raise ModelProviderError(
                f"Request to '{endpoint_url}' timed out after {model_cfg.timeout_seconds}s.",
                error_type="timeout",
                retryable=True,
                cause=exc,
            ) from exc

        elapsed = (time.monotonic() - t0) * 1000.0
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            raise ModelProviderError(
                f"Invalid non-JSON HTTP response (status {status_code}) from provider.",
                error_type="invalid_response",
                status_code=status_code,
                retryable=False,
                cause=exc,
            ) from exc

        content_text, tool_calls, usage = adapter.parse_http_response(payload)

        structured: Optional[Dict[str, Any]] = None
        if expect_json:
            structured = extract_json_object(content_text)

        return (
            ModelResponse(
                content=content_text,
                structured_data=structured,
                tool_calls=tool_calls,
                provider=model_cfg.provider,
                model_id=model_cfg.model_id,
                profile_id=model_cfg.profile_id,
                provider_reached=True,
                is_simulated=False,
                latency_ms=elapsed,
                usage=usage,
            ),
            status_code,
        )
