"""Configuration management, provider/model registries, credential store, and secret redaction."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


class ConfigurationError(ValueError):
    """Raised when Agent 911 configuration is missing, malformed, or invalid."""

    def __init__(
        self,
        message: str,
        cause: Optional[str] = None,
        changes_applied: bool = False,
        next_step: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause or message
        self.changes_applied = changes_applied
        self.next_step = next_step or "Inspect configuration values and retry."

    def to_actionable_dict(self) -> Dict[str, Any]:
        return {
            "what_failed": self.message,
            "known_cause": self.cause,
            "changes_applied": self.changes_applied,
            "next_step": self.next_step,
        }


_SECRET_PATTERNS = [
    re.compile(r"\b(sk-[A-Za-z0-9_\-]{12,})\b"),
    re.compile(r"\b(ant-[A-Za-z0-9_\-]{12,})\b"),
    re.compile(r"\b(gsk_[A-Za-z0-9_\-]{12,})\b"),
    re.compile(r"(Bearer\s+[A-Za-z0-9_\-\.]{8,})", re.IGNORECASE),
    re.compile(r'("api_key"\s*:\s*")([^"]+)(")', re.IGNORECASE),
    re.compile(r"(https?://[^/\s:@]+:)([^/\s@]+)(@)", re.IGNORECASE),
]

SUPPORTED_ADAPTER_TYPES = {
    "openai_compatible",
    "anthropic",
    "ollama",
    "deterministic",
    "local_only",
}


def atomic_write_text(target_path: Path, content: str, mode: Optional[int] = None) -> None:
    """Atomically write text to target_path using a temporary file in the same directory."""
    target_path = target_path.resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target_path.parent),
        prefix=f".{target_path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def validate_endpoint_url(endpoint: str) -> None:
    """Validate an endpoint URL and ensure no credentials are embedded in the URL."""
    if not endpoint or not isinstance(endpoint, str):
        raise ConfigurationError(
            "Endpoint URL must be a non-empty string.",
            next_step="Provide a valid http://, https://, local://, or in-memory:// endpoint.",
        )
    if endpoint.startswith(("local://", "in-memory://")):
        return
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ConfigurationError(
            f"Invalid endpoint URL '{endpoint}'. Must be a valid http:// or https:// URL.",
            next_step="Check the provider or model endpoint URL syntax.",
        )
    if parsed.username or parsed.password or "@" in parsed.netloc:
        raise ConfigurationError(
            "Embedding credentials in endpoint URLs is prohibited for security.",
            cause="Endpoint URL contains userinfo (username/password).",
            next_step="Remove credentials from the URL and configure them via CredentialStore or environment variables.",
        )


def mask_secret_preview(secret: Optional[str]) -> Optional[str]:
    """Return a heavily masked preview of a secret without exposing its value."""
    if not secret:
        return None
    if len(secret) <= 8:
        return "****"
    return f"{secret[:3]}...{secret[-2:]}"


def redact_secrets(value: Any, explicit_secrets: Optional[List[str]] = None) -> Any:
    """Recursively redact sensitive tokens and API keys from strings, dicts, and lists."""
    if value is None:
        return None
    secrets = [s for s in (explicit_secrets or []) if s and len(s) >= 4]
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        redacted = _SECRET_PATTERNS[0].sub("[REDACTED_KEY]", redacted)
        redacted = _SECRET_PATTERNS[1].sub("[REDACTED_KEY]", redacted)
        redacted = _SECRET_PATTERNS[2].sub("[REDACTED_KEY]", redacted)
        redacted = _SECRET_PATTERNS[3].sub("Bearer [REDACTED]", redacted)
        redacted = _SECRET_PATTERNS[4].sub(r"\1[REDACTED]\3", redacted)
        redacted = _SECRET_PATTERNS[5].sub(r"\1[REDACTED]\3", redacted)
        return redacted
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for k, v in value.items():
            k_lower = str(k).lower()
            if any(
                token in k_lower
                for token in (
                    "api_key",
                    "secret",
                    "password",
                    "auth_token",
                    "access_token",
                    "authorization",
                    "x-api-key",
                )
            ) and not k_lower.endswith("_env"):
                result[k] = "[REDACTED]" if v else None
            else:
                result[k] = redact_secrets(v, secrets)
        return result
    if isinstance(value, list):
        return [redact_secrets(item, secrets) for item in value]
    return value


class CredentialStore:
    """
    Manages provider credentials separately from ordinary configuration files.
    Supports OS environment variables and a mode-0600 local credentials vault.
    Never exposes full credentials in diagnostic output.
    """

    def __init__(self, vault_path: Path) -> None:
        self.vault_path = vault_path.resolve()

    def _load_vault(self) -> Dict[str, str]:
        if not self.vault_path.exists():
            return {}
        try:
            data = json.loads(self.vault_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items() if v}
        except Exception:
            return {}
        return {}

    def set_credential(self, provider_id: str, api_key: str) -> None:
        """Persist a provider credential in the mode-0600 credential vault."""
        if not provider_id or not isinstance(provider_id, str):
            raise ConfigurationError("provider_id must be a non-empty string.")
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise ConfigurationError("Credential value cannot be empty.")
        vault = self._load_vault()
        vault[provider_id] = api_key.strip()
        atomic_write_text(
            self.vault_path,
            json.dumps(vault, indent=2),
            mode=stat.S_IRUSR | stat.S_IWUSR,
        )

    def remove_credential(self, provider_id: str) -> bool:
        """Remove a stored credential for a provider."""
        vault = self._load_vault()
        if provider_id not in vault:
            return False
        del vault[provider_id]
        atomic_write_text(
            self.vault_path,
            json.dumps(vault, indent=2),
            mode=stat.S_IRUSR | stat.S_IWUSR,
        )
        return True

    def resolve_credential(
        self,
        provider_id: str,
        api_key_env: Optional[str] = None,
        inline_api_key: Optional[str] = None,
    ) -> Tuple[Optional[str], str]:
        """
        Resolve credential for a provider and return (secret, source).
        Priority: inline override -> provider vault -> specific env var -> standard fallback env vars.
        """
        if inline_api_key:
            return inline_api_key, "inline_config"
        vault = self._load_vault()
        if provider_id in vault and vault[provider_id]:
            return vault[provider_id], "credential_store"
        if api_key_env and os.environ.get(api_key_env):
            return os.environ[api_key_env], f"env:{api_key_env}"
        for fallback_env in ("AGENT911_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
            if os.environ.get(fallback_env):
                return os.environ[fallback_env], f"env:{fallback_env}"
        return None, "none"

    def has_credential(self, provider_id: str) -> bool:
        vault = self._load_vault()
        return bool(vault.get(provider_id))

    def get_masked_credential(self, provider_id: str) -> Optional[str]:
        vault = self._load_vault()
        val = vault.get(provider_id)
        return mask_secret_preview(val) if val else None

    def get_status(
        self,
        provider_id: str,
        api_key_env: Optional[str] = None,
        inline_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        secret, source = self.resolve_credential(provider_id, api_key_env, inline_api_key)
        return {
            "provider_id": provider_id,
            "configured": bool(secret),
            "source": source,
            "api_key_env": api_key_env,
            "masked_preview": mask_secret_preview(secret),
        }

    def all_known_secrets(self, inline_keys: Optional[List[str]] = None) -> List[str]:
        secrets = list(self._load_vault().values())
        for env_name in ("AGENT911_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            val = os.environ.get(env_name)
            if val:
                secrets.append(val)
        if inline_keys:
            secrets.extend([k for k in inline_keys if k])
        return secrets


@dataclass
class ProviderConfig:
    """Provider-level configuration separated from individual model configurations."""

    provider_id: str
    name: str
    adapter_type: str = "openai_compatible"
    endpoint: str = "https://api.openai.com/v1/chat/completions"
    api_key_env: Optional[str] = "AGENT911_API_KEY"
    enabled: bool = True
    custom_headers: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    def validate(self) -> None:
        if not self.provider_id or not isinstance(self.provider_id, str):
            raise ConfigurationError("Provider 'provider_id' must be a non-empty string.")
        if not self.name or not isinstance(self.name, str):
            raise ConfigurationError(f"Provider '{self.provider_id}' must have a non-empty 'name'.")
        if self.adapter_type not in SUPPORTED_ADAPTER_TYPES:
            raise ConfigurationError(
                f"Unsupported adapter_type '{self.adapter_type}' for provider '{self.provider_id}'. "
                f"Supported: {sorted(SUPPORTED_ADAPTER_TYPES)}"
            )
        validate_endpoint_url(self.endpoint)
        for hk, hv in (self.custom_headers or {}).items():
            if not isinstance(hk, str) or not isinstance(hv, str):
                raise ConfigurationError("Provider custom_headers must map strings to strings.")

    def to_dict(self) -> Dict[str, Any]:
        return self.to_safe_dict()

    def to_safe_dict(self, cred_store: Optional[CredentialStore] = None) -> Dict[str, Any]:
        raw = asdict(self)
        raw["custom_headers"] = redact_secrets(self.custom_headers)
        if cred_store is not None:
            raw["credential_status"] = cred_store.get_status(self.provider_id, self.api_key_env)
        return raw


def _normalize_capability_flag(val: Any) -> bool | str:
    if isinstance(val, bool):
        return val
    if isinstance(val, str) and val.lower() in ("true", "false", "unknown"):
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False
        return "unknown"
    return "unknown"


@dataclass
class ModelProfileConfig:
    """
    Model-level configuration supporting multiple models per provider and explicit
    representation of known vs unknown model capabilities.
    """

    profile_id: str
    name: str
    provider_id: str
    model_id: str
    endpoint_override: Optional[str] = None
    context_window_tokens: Optional[int] = None  # None means unknown
    supported_inputs: List[str] = field(default_factory=lambda: ["text"])
    supports_tool_calling: bool | str = "unknown"
    supports_structured_output: bool | str = "unknown"
    supports_streaming: bool | str = "unknown"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_seconds: float = 30.0
    max_retries: int = 2
    enabled: bool = True

    def __post_init__(self) -> None:
        self.supports_tool_calling = _normalize_capability_flag(self.supports_tool_calling)
        self.supports_structured_output = _normalize_capability_flag(
            self.supports_structured_output
        )
        self.supports_streaming = _normalize_capability_flag(self.supports_streaming)
        if not self.supported_inputs:
            self.supported_inputs = ["unknown"]

    def validate(self) -> None:
        if not self.profile_id or not isinstance(self.profile_id, str):
            raise ConfigurationError("Model 'profile_id' must be a non-empty string.")
        if not self.name or not isinstance(self.name, str):
            raise ConfigurationError(f"Model profile '{self.profile_id}' must have a non-empty name.")
        if not self.provider_id or not isinstance(self.provider_id, str):
            raise ConfigurationError(
                f"Model profile '{self.profile_id}' must reference a non-empty 'provider_id'."
            )
        if not self.model_id or not isinstance(self.model_id, str):
            raise ConfigurationError(
                f"Model profile '{self.profile_id}' must have a non-empty 'model_id'."
            )
        if self.endpoint_override:
            validate_endpoint_url(self.endpoint_override)
        if self.context_window_tokens is not None and self.context_window_tokens <= 0:
            raise ConfigurationError(
                f"Model '{self.profile_id}' context_window_tokens must be positive when specified."
            )
        if self.timeout_seconds <= 0:
            raise ConfigurationError(
                f"Model '{self.profile_id}' timeout_seconds must be greater than 0."
            )
        if self.max_retries < 0:
            raise ConfigurationError(
                f"Model '{self.profile_id}' max_retries cannot be negative."
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigurationError(
                f"Model '{self.profile_id}' temperature must be between 0.0 and 2.0."
            )
        if self.max_tokens <= 0:
            raise ConfigurationError(
                f"Model '{self.profile_id}' max_tokens must be greater than 0."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """
    Legacy/effective single-model view maintained for seamless compatibility with
    orchestrator components and existing tests.
    """

    provider: str = "openai_compatible"
    model_id: str = "gpt-4o-mini"
    endpoint: str = "https://api.openai.com/v1/chat/completions"
    api_key_env: str = "AGENT911_API_KEY"
    api_key: Optional[str] = None
    timeout_seconds: float = 30.0
    max_retries: int = 2
    temperature: float = 0.1
    max_tokens: int = 2048
    allow_local_fallback: bool = True
    provider_id: str = "default"
    profile_id: str = "default"
    context_window_tokens: Optional[int] = None
    supported_inputs: List[str] = field(default_factory=lambda: ["text"])
    supports_tool_calling: bool | str = "unknown"
    supports_structured_output: bool | str = "unknown"
    supports_streaming: bool | str = "unknown"
    custom_headers: Dict[str, str] = field(default_factory=dict)

    def resolve_api_key(
        self, cred_store: Optional[CredentialStore] = None
    ) -> Optional[str]:
        """Resolve API key from inline config, CredentialStore, or environment variables."""
        if self.api_key:
            return self.api_key
        if cred_store is not None:
            secret, _ = cred_store.resolve_credential(
                self.provider_id, self.api_key_env, self.api_key
            )
            if secret:
                return secret
        if self.api_key_env and os.environ.get(self.api_key_env):
            return os.environ.get(self.api_key_env)
        for fallback_env in ("AGENT911_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
            val = os.environ.get(fallback_env)
            if val:
                return val
        return None

    def validate(self) -> None:
        if not self.provider or not isinstance(self.provider, str):
            raise ConfigurationError("Model provider must be a non-empty string.")
        if self.provider not in SUPPORTED_ADAPTER_TYPES:
            raise ConfigurationError(
                f"Unsupported model provider '{self.provider}'. "
                f"Supported providers: {sorted(SUPPORTED_ADAPTER_TYPES)}"
            )
        if not self.model_id or not isinstance(self.model_id, str):
            raise ConfigurationError("Model identifier (model_id) must be a non-empty string.")
        validate_endpoint_url(self.endpoint)
        if self.timeout_seconds <= 0:
            raise ConfigurationError("Model timeout_seconds must be greater than 0.")
        if self.max_retries < 0:
            raise ConfigurationError("Model max_retries cannot be negative.")
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigurationError("Model temperature must be between 0.0 and 2.0.")
        if self.max_tokens <= 0:
            raise ConfigurationError("Model max_tokens must be greater than 0.")

    def to_safe_dict(self, cred_store: Optional[CredentialStore] = None) -> Dict[str, Any]:
        raw = asdict(self)
        resolved = self.resolve_api_key(cred_store)
        raw["has_credentials"] = bool(resolved)
        raw["masked_api_key"] = mask_secret_preview(resolved)
        raw["api_key"] = "[REDACTED]" if self.api_key else None
        raw["custom_headers"] = redact_secrets(self.custom_headers)
        return raw


@dataclass
class AgentConfig:
    """
    Top-level configuration for Agent 911 supporting multi-provider and multi-model
    registries, active/default/fallback model selection, schema versioning, and atomic persistence.
    """

    workspace_root: str = "."
    data_dir: str = ".agent911"
    database_path: str = ".agent911/agent911.db"
    custom_capabilities_dir: str = ".agent911/capabilities"
    max_recovery_attempts_per_step: int = 2
    command_timeout_seconds: float = 30.0
    max_context_chars: int = 16000
    max_model_calls_per_task: int = 10
    max_subtasks_per_task: int = 20
    max_task_duration_seconds: float = 120.0
    schema_version: int = 2
    model: ModelConfig = field(default_factory=ModelConfig)
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    models: Dict[str, ModelProfileConfig] = field(default_factory=dict)
    default_model: str = "default"
    active_model: str = "default"
    fallback_model: Optional[str] = None
    model_routing_rules: Dict[str, str] = field(default_factory=dict)
    allow_local_fallback: bool = True
    validation_warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)

        # Convert raw provider dicts if passed
        norm_providers: Dict[str, ProviderConfig] = {}
        for pid, pval in (self.providers or {}).items():
            if isinstance(pval, ProviderConfig):
                norm_providers[pid] = pval
            elif isinstance(pval, dict):
                copied = dict(pval)
                copied.setdefault("provider_id", pid)
                norm_providers[pid] = ProviderConfig(**copied)
        self.providers = norm_providers

        # Convert raw model dicts if passed
        norm_models: Dict[str, ModelProfileConfig] = {}
        for mid, mval in (self.models or {}).items():
            if isinstance(mval, ModelProfileConfig):
                norm_models[mid] = mval
            elif isinstance(mval, dict):
                copied = dict(mval)
                copied.setdefault("profile_id", mid)
                norm_models[mid] = ModelProfileConfig(**copied)
        self.models = norm_models

        self._bootstrap_registries_from_model_if_empty()
        self._sync_effective_model()

    def _bootstrap_registries_from_model_if_empty(self) -> None:
        """Ensure default provider and model registries are populated from `self.model`."""
        if not self.providers:
            pid = self.model.provider_id or "default"
            self.providers[pid] = ProviderConfig(
                provider_id=pid,
                name=f"Default ({self.model.provider})",
                adapter_type=self.model.provider,
                endpoint=self.model.endpoint,
                api_key_env=self.model.api_key_env,
                enabled=True,
                custom_headers=dict(self.model.custom_headers),
            )
            # Also register built-in local_only provider for easy switching/fallback
            if "local_heuristic" not in self.providers:
                self.providers["local_heuristic"] = ProviderConfig(
                    provider_id="local_heuristic",
                    name="Local Heuristic Engine",
                    adapter_type="local_only",
                    endpoint="local://heuristic",
                    api_key_env=None,
                    enabled=True,
                    description="Built-in offline instruction parser and planner",
                )

        if not self.models:
            mid = self.model.profile_id or "default"
            pid = self.model.provider_id if self.model.provider_id in self.providers else next(iter(self.providers))
            self.models[mid] = ModelProfileConfig(
                profile_id=mid,
                name=f"{self.model.model_id} ({pid})",
                provider_id=pid,
                model_id=self.model.model_id,
                context_window_tokens=self.model.context_window_tokens,
                supported_inputs=list(self.model.supported_inputs),
                supports_tool_calling=self.model.supports_tool_calling,
                supports_structured_output=self.model.supports_structured_output,
                supports_streaming=self.model.supports_streaming,
                temperature=self.model.temperature,
                max_tokens=self.model.max_tokens,
                timeout_seconds=self.model.timeout_seconds,
                max_retries=self.model.max_retries,
                enabled=True,
            )
            if "local-planner" not in self.models and "local_heuristic" in self.providers:
                self.models["local-planner"] = ModelProfileConfig(
                    profile_id="local-planner",
                    name="Local Planner Model",
                    provider_id="local_heuristic",
                    model_id="local-heuristic-v1",
                    endpoint_override="local://heuristic",
                    context_window_tokens=4096,
                    supported_inputs=["text"],
                    supports_tool_calling=False,
                    supports_structured_output=True,
                    supports_streaming=False,
                    enabled=True,
                )

        if self.active_model not in self.models and self.models:
            self.active_model = next(iter(self.models))
        if self.default_model not in self.models and self.models:
            self.default_model = self.active_model

    def _sync_effective_model(self) -> None:
        """Keep `self.model` synchronized with the currently selected `active_model` profile."""
        if self.active_model in self.models:
            prof = self.models[self.active_model]
            prov = self.providers.get(prof.provider_id)
            if prov is not None:
                inline_key = self.model.api_key if self.model.profile_id == prof.profile_id else None
                self.model = ModelConfig(
                    provider=prov.adapter_type,
                    model_id=prof.model_id,
                    endpoint=prof.endpoint_override or prov.endpoint,
                    api_key_env=prov.api_key_env or "AGENT911_API_KEY",
                    api_key=inline_key,
                    timeout_seconds=prof.timeout_seconds,
                    max_retries=prof.max_retries,
                    temperature=prof.temperature,
                    max_tokens=prof.max_tokens,
                    allow_local_fallback=self.allow_local_fallback,
                    provider_id=prov.provider_id,
                    profile_id=prof.profile_id,
                    context_window_tokens=prof.context_window_tokens,
                    supported_inputs=list(prof.supported_inputs),
                    supports_tool_calling=prof.supports_tool_calling,
                    supports_structured_output=prof.supports_structured_output,
                    supports_streaming=prof.supports_streaming,
                    custom_headers=dict(prov.custom_headers),
                )

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace_root).resolve()

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p.resolve() if p.is_absolute() else (self.workspace_path / p).resolve()

    @property
    def db_path(self) -> Path:
        p = Path(self.database_path)
        return p.resolve() if p.is_absolute() else (self.workspace_path / p).resolve()

    @property
    def capabilities_path(self) -> Path:
        p = Path(self.custom_capabilities_dir)
        return p.resolve() if p.is_absolute() else (self.workspace_path / p).resolve()

    @property
    def credentials_vault_path(self) -> Path:
        return (self.data_path / "credentials.json").resolve()

    @property
    def credential_store(self) -> CredentialStore:
        return CredentialStore(self.credentials_vault_path)

    def validate(self) -> None:
        if not self.workspace_root or not isinstance(self.workspace_root, str):
            raise ConfigurationError("workspace_root must be a non-empty path string.")
        ws = Path(self.workspace_root)
        if ws.exists() and not ws.is_dir():
            raise ConfigurationError(
                f"Configured workspace_root '{self.workspace_root}' exists and is not a directory."
            )
        if self.max_recovery_attempts_per_step < 0:
            raise ConfigurationError("max_recovery_attempts_per_step cannot be negative.")
        if self.command_timeout_seconds <= 0:
            raise ConfigurationError("command_timeout_seconds must be greater than 0.")
        if self.max_context_chars < 512:
            raise ConfigurationError("max_context_chars must be at least 512 characters.")

        for pid, pcfg in self.providers.items():
            pcfg.validate()
        for mid, mcfg in self.models.items():
            mcfg.validate()
            if mcfg.provider_id not in self.providers:
                raise ConfigurationError(
                    f"Model profile '{mid}' references unknown provider '{mcfg.provider_id}'."
                )

        if self.active_model not in self.models:
            raise ConfigurationError(
                f"Active model profile '{self.active_model}' does not exist in configured models."
            )
        if self.default_model not in self.models:
            raise ConfigurationError(
                f"Default model profile '{self.default_model}' does not exist in configured models."
            )
        if self.fallback_model is not None and self.fallback_model not in self.models:
            raise ConfigurationError(
                f"Fallback model profile '{self.fallback_model}' does not exist in configured models."
            )
        self.model.validate()

    # ------------------------------------------------------------------
    # Provider & Model Registry Management Operations
    # ------------------------------------------------------------------

    def add_or_update_provider(
        self,
        provider: ProviderConfig | Dict[str, Any],
        api_key: Optional[str] = None,
    ) -> ProviderConfig:
        """Add or update a provider configuration after validating it."""
        cfg = (
            provider
            if isinstance(provider, ProviderConfig)
            else ProviderConfig(**provider)
        )
        cfg.validate()
        self.providers[cfg.provider_id] = cfg
        if api_key and api_key.strip():
            self.credential_store.set_credential(cfg.provider_id, api_key.strip())
        self._sync_effective_model()
        return cfg

    def remove_provider(self, provider_id: str) -> None:
        """Remove a provider if no active/default model depends on it."""
        if provider_id not in self.providers:
            raise ConfigurationError(f"Provider '{provider_id}' does not exist.")
        dependent_models = [
            mid for mid, m in self.models.items() if m.provider_id == provider_id
        ]
        if any(m in (self.active_model, self.default_model) for m in dependent_models):
            raise ConfigurationError(
                f"Cannot remove provider '{provider_id}' because active or default model "
                f"({dependent_models}) depends on it.",
                next_step="Switch active/default model to another provider before removing this provider.",
            )
        for mid in dependent_models:
            del self.models[mid]
        del self.providers[provider_id]
        self.credential_store.remove_credential(provider_id)

    def add_or_update_model(
        self, model_profile: ModelProfileConfig | Dict[str, Any]
    ) -> ModelProfileConfig:
        """Add or update a model profile configuration after validating it."""
        cfg = (
            model_profile
            if isinstance(model_profile, ModelProfileConfig)
            else ModelProfileConfig(**model_profile)
        )
        cfg.validate()
        if cfg.provider_id not in self.providers:
            raise ConfigurationError(
                f"Cannot configure model '{cfg.profile_id}': provider '{cfg.provider_id}' is not registered.",
                next_step=f"Add provider '{cfg.provider_id}' first.",
            )
        self.models[cfg.profile_id] = cfg
        self._sync_effective_model()
        return cfg

    def remove_model(self, profile_id: str) -> None:
        """Remove a configured model profile unless it is currently active or default."""
        if profile_id not in self.models:
            raise ConfigurationError(f"Model profile '{profile_id}' does not exist.")
        if profile_id in (self.active_model, self.default_model):
            raise ConfigurationError(
                f"Cannot remove model profile '{profile_id}' while it is the active or default model.",
                next_step="Select a different active and default model first.",
            )
        if self.fallback_model == profile_id:
            self.fallback_model = None
        del self.models[profile_id]

    def select_active_model(
        self,
        profile_id: str,
        set_as_default: bool = False,
    ) -> ModelProfileConfig:
        """
        Switch the active model at runtime after validating model and provider state.
        Preserves prior state if switching fails.
        """
        if profile_id not in self.models:
            raise ConfigurationError(
                f"Cannot switch to model '{profile_id}': model profile is not configured.",
                changes_applied=False,
                next_step=f"Choose from configured models: {sorted(self.models.keys())}",
            )
        target_model = self.models[profile_id]
        if not target_model.enabled:
            raise ConfigurationError(
                f"Cannot switch to model '{profile_id}' because it is currently disabled.",
                changes_applied=False,
                next_step=f"Enable model '{profile_id}' before selecting it.",
            )
        prov = self.providers.get(target_model.provider_id)
        if prov is None:
            raise ConfigurationError(
                f"Model '{profile_id}' references missing provider '{target_model.provider_id}'.",
                changes_applied=False,
            )
        if not prov.enabled:
            raise ConfigurationError(
                f"Cannot switch to model '{profile_id}' because its provider '{prov.provider_id}' is disabled.",
                changes_applied=False,
                next_step=f"Enable provider '{prov.provider_id}' first.",
            )

        self.active_model = profile_id
        if set_as_default:
            self.default_model = profile_id
        self._sync_effective_model()
        return target_model

    def resolve_model_Config_for_profile(self, profile_id: str) -> ModelConfig:
        """Build an effective ModelConfig for a given profile_id."""
        if profile_id not in self.models:
            raise ConfigurationError(f"Model profile '{profile_id}' is not configured.")
        prof = self.models[profile_id]
        if not prof.enabled:
            raise ConfigurationError(f"Model profile '{profile_id}' is disabled.")
        prov = self.providers.get(prof.provider_id)
        if prov is None or not prov.enabled:
            raise ConfigurationError(
                f"Provider '{prof.provider_id}' for model '{profile_id}' is missing or disabled."
            )
        inline_key = (
            self.model.api_key if self.model.profile_id == prof.profile_id else None
        )
        return ModelConfig(
            provider=prov.adapter_type,
            model_id=prof.model_id,
            endpoint=prof.endpoint_override or prov.endpoint,
            api_key_env=prov.api_key_env or "AGENT911_API_KEY",
            api_key=inline_key,
            timeout_seconds=prof.timeout_seconds,
            max_retries=prof.max_retries,
            temperature=prof.temperature,
            max_tokens=prof.max_tokens,
            allow_local_fallback=self.allow_local_fallback,
            provider_id=prov.provider_id,
            profile_id=prof.profile_id,
            context_window_tokens=prof.context_window_tokens,
            supported_inputs=list(prof.supported_inputs),
            supports_tool_calling=prof.supports_tool_calling,
            supports_structured_output=prof.supports_structured_output,
            supports_streaming=prof.supports_streaming,
            custom_headers=dict(prov.custom_headers),
        )

    def select_model_for_operation(
        self,
        operation_category: str = "general",
        required_context_tokens: Optional[int] = None,
        require_tool_calling: bool = False,
        require_structured_output: bool = False,
        input_type: str = "text",
        allow_capability_routing: bool = False,
    ) -> Dict[str, Any]:
        """
        Capability-aware model selection using configured metadata and routing rules (Section 10).
        Never overrides explicit active_model selection unless allow_capability_routing=True or
        a configurable model_routing_rules entry is set for operation_category.
        """
        def _check_profile_suitability(prof: ModelProfileConfig) -> Tuple[bool, List[str]]:
            reasons: List[str] = []
            if not prof.enabled:
                return False, [f"Model profile '{prof.profile_id}' is disabled."]
            prov = self.providers.get(prof.provider_id)
            if prov is None or not prov.enabled:
                return False, [f"Provider '{prof.provider_id}' is missing or disabled."]
            if (
                required_context_tokens is not None
                and prof.context_window_tokens is not None
                and prof.context_window_tokens < required_context_tokens
            ):
                reasons.append(
                    f"context_window_tokens ({prof.context_window_tokens}) < required ({required_context_tokens})"
                )
            if require_tool_calling and prof.supports_tool_calling is False:
                reasons.append("supports_tool_calling is False")
            if require_structured_output and prof.supports_structured_output is False:
                reasons.append("supports_structured_output is False")
            if (
                input_type != "text"
                and "unknown" not in prof.supported_inputs
                and input_type not in prof.supported_inputs
            ):
                reasons.append(f"input_type '{input_type}' not in {prof.supported_inputs}")
            return len(reasons) == 0, reasons

        # 1. Check if an explicit routing rule exists for this operation_category
        routed_id = self.model_routing_rules.get(operation_category)
        if routed_id and routed_id in self.models:
            routed_prof = self.models[routed_id]
            ok, problems = _check_profile_suitability(routed_prof)
            if ok:
                return {
                    "selected_profile_id": routed_id,
                    "suitable": True,
                    "reason": f"Selected via model_routing_rules['{operation_category}'] -> '{routed_id}'.",
                    "limitations": [],
                }

        # 2. Check the currently active model profile
        active_prof = self.models.get(self.active_model)
        if active_prof is not None:
            ok, problems = _check_profile_suitability(active_prof)
            if ok:
                return {
                    "selected_profile_id": self.active_model,
                    "suitable": True,
                    "reason": f"Active model '{self.active_model}' satisfies all operation requirements.",
                    "limitations": [],
                }
            if not allow_capability_routing:
                return {
                    "selected_profile_id": self.active_model,
                    "suitable": False,
                    "reason": (
                        f"Active model '{self.active_model}' does not satisfy requirements "
                        f"({'; '.join(problems)}), and automatic switching is disabled to respect explicit selection."
                    ),
                    "limitations": problems,
                }

        # 3. If allow_capability_routing is True, search enabled models for the best suitable candidate
        suitable_candidates: List[Tuple[int, str]] = []
        for cand_id, cand_prof in self.models.items():
            ok, _ = _check_profile_suitability(cand_prof)
            if ok:
                score = 0
                if (
                    required_context_tokens is not None
                    and cand_prof.context_window_tokens is not None
                    and cand_prof.context_window_tokens >= required_context_tokens
                ):
                    score += 2
                if require_tool_calling and cand_prof.supports_tool_calling is True:
                    score += 2
                if require_structured_output and cand_prof.supports_structured_output is True:
                    score += 2
                suitable_candidates.append((score, cand_id))

        if suitable_candidates:
            suitable_candidates.sort(key=lambda item: item[0], reverse=True)
            best_id = suitable_candidates[0][1]
            return {
                "selected_profile_id": best_id,
                "suitable": True,
                "reason": (
                    f"Capability-aware routing selected '{best_id}' because active model "
                    f"'{self.active_model}' did not meet requirements."
                ),
                "limitations": [],
            }

        return {
            "selected_profile_id": None,
            "suitable": False,
            "reason": "No enabled model profile satisfies the requested capability/context requirements.",
            "limitations": ["No compatible model profile found in models registry."],
        }

    def ensure_directories(self) -> None:
        """Ensure workspace and runtime directories exist."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.capabilities_path.mkdir(parents=True, exist_ok=True)

    def save(self, config_path: Optional[str | Path] = None) -> Path:
        """
        Atomically persist configuration in schema_version 2 format without exposing secrets.
        """
        self.validate()
        target = (
            Path(config_path).resolve()
            if config_path is not None
            else (self.workspace_path / "agent911.config.local.json").resolve()
        )
        payload = {
            "schema_version": 2,
            "workspace_root": self.workspace_root,
            "data_dir": self.data_dir,
            "database_path": self.database_path,
            "custom_capabilities_dir": self.custom_capabilities_dir,
            "max_recovery_attempts_per_step": self.max_recovery_attempts_per_step,
            "command_timeout_seconds": self.command_timeout_seconds,
            "max_context_chars": self.max_context_chars,
            "max_model_calls_per_task": self.max_model_calls_per_task,
            "max_subtasks_per_task": self.max_subtasks_per_task,
            "max_task_duration_seconds": self.max_task_duration_seconds,
            "allow_local_fallback": self.allow_local_fallback,
            "default_model": self.default_model,
            "active_model": self.active_model,
            "fallback_model": self.fallback_model,
            "model_routing_rules": dict(self.model_routing_rules),
            "providers": {
                pid: asdict(pcfg) for pid, pcfg in self.providers.items()
            },
            "models": {
                mid: mcfg.to_dict() for mid, mcfg in self.models.items()
            },
        }
        atomic_write_text(target, json.dumps(payload, indent=2) + "\n")
        return target

    def to_safe_dict(self) -> Dict[str, Any]:
        cstore = self.credential_store
        return {
            "schema_version": self.schema_version,
            "workspace_root": str(self.workspace_path),
            "data_dir": str(self.data_path),
            "database_path": str(self.db_path),
            "custom_capabilities_dir": str(self.capabilities_path),
            "max_recovery_attempts_per_step": self.max_recovery_attempts_per_step,
            "command_timeout_seconds": self.command_timeout_seconds,
            "max_context_chars": self.max_context_chars,
            "max_model_calls_per_task": self.max_model_calls_per_task,
            "max_subtasks_per_task": self.max_subtasks_per_task,
            "max_task_duration_seconds": self.max_task_duration_seconds,
            "allow_local_fallback": self.allow_local_fallback,
            "default_model": self.default_model,
            "active_model": self.active_model,
            "fallback_model": self.fallback_model,
            "model_routing_rules": dict(self.model_routing_rules),
            "model": self.model.to_safe_dict(cstore),
            "providers": {
                pid: pcfg.to_safe_dict(cstore) for pid, pcfg in self.providers.items()
            },
            "models": {
                mid: mcfg.to_dict() for mid, mcfg in self.models.items()
            },
            "validation_warnings": list(self.validation_warnings),
        }

    @classmethod
    def load(
        cls,
        config_path: Optional[str | Path] = None,
        workspace_override: Optional[str | Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> AgentConfig:
        """
        Load configuration from JSON file (v1 or v2), environment variables, and overrides.
        Preserves valid provider/model entries when an unrelated non-active entry is invalid.
        """
        raw: Dict[str, Any] = {}
        model_raw: Dict[str, Any] = {}
        providers_raw: Dict[str, Any] = {}
        models_raw: Dict[str, Any] = {}
        warnings: List[str] = []

        target_cfg_file: Optional[Path] = None
        if config_path is not None:
            target_cfg_file = Path(config_path)
            if not target_cfg_file.exists():
                raise ConfigurationError(f"Configuration file not found: {target_cfg_file}")
        else:
            ws_base = Path(workspace_override) if workspace_override else Path(".")
            candidate_local = ws_base / "agent911.config.local.json"
            if candidate_local.exists():
                target_cfg_file = candidate_local

        if target_cfg_file is not None:
            try:
                loaded = json.loads(target_cfg_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ConfigurationError(
                    f"Invalid JSON in configuration file '{target_cfg_file}': {exc}"
                ) from exc
            if not isinstance(loaded, dict):
                raise ConfigurationError(
                    f"Configuration file '{target_cfg_file}' must contain a JSON object at top level."
                )
            if isinstance(loaded.get("model"), dict):
                model_raw.update(loaded.pop("model"))
            if isinstance(loaded.get("providers"), dict):
                providers_raw.update(loaded.pop("providers"))
            if isinstance(loaded.get("models"), dict):
                models_raw.update(loaded.pop("models"))
            raw.update(loaded)

        # Environment variable overrides
        if os.environ.get("AGENT911_WORKSPACE"):
            raw["workspace_root"] = os.environ["AGENT911_WORKSPACE"]
        if os.environ.get("AGENT911_DB_PATH"):
            raw["database_path"] = os.environ["AGENT911_DB_PATH"]
        if os.environ.get("AGENT911_PROVIDER"):
            model_raw["provider"] = os.environ["AGENT911_PROVIDER"]
        if os.environ.get("AGENT911_MODEL_ID"):
            model_raw["model_id"] = os.environ["AGENT911_MODEL_ID"]
        if os.environ.get("AGENT911_ENDPOINT"):
            model_raw["endpoint"] = os.environ["AGENT911_ENDPOINT"]
        if os.environ.get("AGENT911_TIMEOUT"):
            try:
                model_raw["timeout_seconds"] = float(os.environ["AGENT911_TIMEOUT"])
            except ValueError as exc:
                raise ConfigurationError(
                    f"Invalid float in AGENT911_TIMEOUT: {os.environ['AGENT911_TIMEOUT']}"
                ) from exc

        if workspace_override is not None:
            raw["workspace_root"] = str(workspace_override)

        if overrides:
            if "model" in overrides and isinstance(overrides["model"], dict):
                model_raw.update(overrides["model"])
            if "providers" in overrides and isinstance(overrides["providers"], dict):
                providers_raw.update(overrides["providers"])
            if "models" in overrides and isinstance(overrides["models"], dict):
                models_raw.update(overrides["models"])
            for k, v in overrides.items():
                if k not in ("model", "providers", "models"):
                    raw[k] = v

        allowed_top_keys = {
            "schema_version",
            "workspace_root",
            "data_dir",
            "database_path",
            "custom_capabilities_dir",
            "max_recovery_attempts_per_step",
            "command_timeout_seconds",
            "max_context_chars",
            "max_model_calls_per_task",
            "max_subtasks_per_task",
            "max_task_duration_seconds",
            "allow_local_fallback",
            "default_model",
            "active_model",
            "fallback_model",
            "model_routing_rules",
        }
        unknown_top = set(raw.keys()) - allowed_top_keys
        if unknown_top:
            raise ConfigurationError(f"Unknown configuration keys: {sorted(unknown_top)}")

        allowed_model_keys = {
            "provider",
            "model_id",
            "endpoint",
            "api_key_env",
            "api_key",
            "timeout_seconds",
            "max_retries",
            "temperature",
            "max_tokens",
            "allow_local_fallback",
            "provider_id",
            "profile_id",
            "context_window_tokens",
            "supported_inputs",
            "supports_tool_calling",
            "supports_structured_output",
            "supports_streaming",
            "custom_headers",
        }
        unknown_model = set(model_raw.keys()) - allowed_model_keys
        if unknown_model:
            raise ConfigurationError(
                f"Unknown model configuration keys: {sorted(unknown_model)}"
            )

        if "allow_local_fallback" in model_raw and "allow_local_fallback" not in raw:
            raw["allow_local_fallback"] = bool(model_raw["allow_local_fallback"])

        # Validate individual provider & model entries, isolating invalid non-active entries
        valid_providers: Dict[str, ProviderConfig] = {}
        for pid, pdata in providers_raw.items():
            try:
                if isinstance(pdata, ProviderConfig):
                    pcfg = pdata
                elif isinstance(pdata, dict):
                    copied = dict(pdata)
                    copied.setdefault("provider_id", pid)
                    pcfg = ProviderConfig(**copied)
                else:
                    raise ConfigurationError(f"Provider '{pid}' must be an object.")
                pcfg.validate()
                valid_providers[pid] = pcfg
            except Exception as exc:
                warnings.append(f"Ignored invalid provider '{pid}': {exc}")

        valid_models: Dict[str, ModelProfileConfig] = {}
        for mid, mdata in models_raw.items():
            try:
                if isinstance(mdata, ModelProfileConfig):
                    mcfg = mdata
                elif isinstance(mdata, dict):
                    copied = dict(mdata)
                    copied.setdefault("profile_id", mid)
                    mcfg = ModelProfileConfig(**copied)
                else:
                    raise ConfigurationError(f"Model '{mid}' must be an object.")
                mcfg.validate()
                if valid_providers and mcfg.provider_id not in valid_providers:
                    raise ConfigurationError(
                        f"Model '{mid}' references unavailable provider '{mcfg.provider_id}'."
                    )
                valid_models[mid] = mcfg
            except Exception as exc:
                warnings.append(f"Ignored invalid model '{mid}': {exc}")

        model_cfg = ModelConfig(**model_raw)
        config = cls(
            model=model_cfg,
            providers=valid_providers,
            models=valid_models,
            validation_warnings=warnings,
            **raw,
        )
        config.validate()
        return config
