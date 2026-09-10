"""Resolve checkpoint names once, before weights, into reproducible model settings."""
from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from halt.errors import CapabilityError, ConfigurationError
from halt.profiles.qwen3 import QWEN3_MODEL_ID, QWEN3_REVISION

if TYPE_CHECKING:
    from halt.config import ResolvedConfig


def _hub() -> Any:
    try:
        hub = import_module("huggingface_hub")
    except ImportError as exc:
        raise ConfigurationError("Model trials require the transformers extra; from the checkout run python -m pip install -e '.[transformers]'.") from exc
    return hub


def resolve_model_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Validate small configuration metadata; never download model weights here.

    The tested default stays pinned. Other names/branches resolve to a commit at
    execution time, which the caller saves in the resolved experiment manifest.
    Offline use requires a pinned revision and cached metadata.
    """
    resolved = dict(settings)
    model_id = resolved.setdefault("name", QWEN3_MODEL_ID)
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigurationError("model.name must be nonempty text")
    for key in ("revision", "tokenizer_revision"):
        value = resolved.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConfigurationError(f"model.{key} must be a nonempty revision string")
    if not isinstance(resolved.get("dtype", "float32"), str) or resolved.get("dtype", "float32") not in {"float32", "float16", "bfloat16"}:
        raise ConfigurationError("model.dtype must be float32, float16, or bfloat16")
    if not isinstance(resolved.get("device", "cpu"), str) or not resolved.get("device", "cpu").strip():
        raise ConfigurationError("model.device must be nonempty text")
    if type(resolved.get("local_files_only", False)) is not bool:
        raise ConfigurationError("model.local_files_only must be a boolean")
    for key, default in (("temperature", 0.6), ("top_p", 0.95)):
        value = resolved.get(key, default)
        if isinstance(value, bool) or not isinstance(value, float | int) or not math.isfinite(value):
            raise ConfigurationError(f"model.{key} must be a finite number")
        if (key == "temperature" and value < 0) or (key == "top_p" and not 0 < value <= 1):
            raise ConfigurationError("model.temperature must be nonnegative and model.top_p must be in (0, 1]")
    if type(resolved.get("top_k", 20)) is not int or resolved.get("top_k", 20) < 0:
        raise ConfigurationError("model.top_k must be a nonnegative integer")
    if resolved.get("model_profile", "qwen3_thinking") != "qwen3_thinking":
        raise CapabilityError("Available model profile: qwen3_thinking (decoder-only Qwen3).")
    hub = _hub()

    def revision(value: str | None) -> str:
        if value and re.fullmatch(r"[0-9a-f]{40}", value):
            return value
        if resolved.get("local_files_only", False):
            raise ConfigurationError("Offline model selection needs an immutable 40-character --revision and cached model files.")
        try:
            commit = hub.HfApi().model_info(model_id, revision=value or "main", timeout=20).sha
        except Exception as exc:
            raise ConfigurationError(f"Could not resolve a model revision for {model_id!r}. Check the model name, revision, network, and access permissions ({type(exc).__name__}).") from exc
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ConfigurationError("The model service did not return an immutable revision.")
        return commit

    requested = resolved.get("revision") or (QWEN3_REVISION if model_id == QWEN3_MODEL_ID else None)
    resolved["revision"] = revision(requested)
    tokenizer = resolved.get("tokenizer_revision")
    resolved["tokenizer_revision"] = resolved["revision"] if not tokenizer or tokenizer == requested else revision(tokenizer)
    try:
        path = hub.hf_hub_download(model_id, "config.json", revision=resolved["revision"],
            cache_dir=resolved.get("cache_dir"), local_files_only=resolved.get("local_files_only", False))
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("model configuration must be a JSON object")
    except Exception as exc:
        raise ConfigurationError(f"Could not read model configuration for {model_id!r} at the resolved revision ({type(exc).__name__}).") from exc
    if metadata.get("model_type") != "qwen3" or metadata.get("is_encoder_decoder", False):
        raise CapabilityError(f"Model {model_id!r} has model_type={metadata.get('model_type')!r}; this backend supports decoder-only Qwen3. A new model family needs a backend/profile extension.")
    return resolved


def resolve_model_config(config: ResolvedConfig) -> ResolvedConfig:
    if config.backend["name"] != "transformers":
        return config
    return replace(config, model=resolve_model_settings(config.model))
