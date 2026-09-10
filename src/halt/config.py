"""Strict sectioned configuration, with JSON/TOML in the dependency-free core."""
from __future__ import annotations

import inspect
import json
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from halt.errors import ConfigurationError
from halt.registry import MethodRegistry
from halt.tasks import MultipleChoiceTask, NumericTask, Task
from halt.types import SCHEMA_VERSION, Budget, to_data

_SECTIONS = {"schema_version", "backend", "model", "method", "runtime", "budget", "task",
             "evaluation", "calibration"}
_KEYS = {
    "backend": {"name", "parameters"},
    "model": {"name", "revision", "tokenizer_revision", "model_profile", "device", "dtype",
              "temperature", "top_p", "top_k", "local_files_only", "trust_remote_code", "cache_dir"},
    "method": {"name", "parameters"},
    "runtime": {"seed", "capture"},
    "task": {"type", "question", "choices"},
    "evaluation": {"dataset", "adapter", "split", "data_revision", "methods", "baseline",
                   "bootstrap_samples", "seed", "resume", "session", "protocol", "limit",
                   "question_column", "answer_column", "id_column", "choices_column", "choice_columns"},
    "calibration": {"parameter_grid", "accuracy_tolerance", "objective", "recipe_frozen"},
}


def _unknown(data: dict[str, Any], allowed: set[str], section: str) -> None:
    extra = set(data) - allowed
    if extra:
        raise ConfigurationError(f"unknown {section} fields: {', '.join(sorted(extra))}")


@dataclass(frozen=True)
class ResolvedConfig:
    backend: dict[str, Any] = field(default_factory=lambda: {"name": "scripted", "parameters": {}})
    model: dict[str, Any] = field(default_factory=dict)
    method: dict[str, Any] = field(default_factory=lambda: {"name": "full_reasoning", "parameters": {}})
    runtime: dict[str, Any] = field(default_factory=lambda: {"seed": 0, "capture": "none"})
    budget: Budget = field(default_factory=Budget)
    task: dict[str, Any] = field(default_factory=lambda: {
        "type": "mcq", "question": "A shop has 12 apples and sells 5. How many remain?",
        "choices": {"A": "5", "B": "6", "C": "7", "D": "8"}})
    evaluation: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return dict(to_data(self))


def validate_config(data: dict[str, Any], *, base_dir: Path | None = None) -> ResolvedConfig:
    if not isinstance(data, dict):
        raise ConfigurationError("configuration must be an object")
    _unknown(data, _SECTIONS, "top-level")
    if str(data.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ConfigurationError("unsupported configuration schema_version")
    merged = ResolvedConfig().to_dict()
    for section, allowed in _KEYS.items():
        if section in data:
            if not isinstance(data[section], dict):
                raise ConfigurationError(f"{section} must be an object")
            _unknown(data[section], allowed, section)
            if section == "task":
                merged[section] = dict(data[section])
            else:
                merged[section].update(data[section])
    budget_data = data.get("budget", {})
    if not isinstance(budget_data, dict):
        raise ConfigurationError("budget must be an object")
    _unknown(budget_data, {f.name for f in fields(Budget)}, "budget")
    merged["budget"] = Budget(**budget_data)
    if merged["backend"]["name"] not in {"scripted", "transformers"}:
        raise ConfigurationError("backend.name must be scripted or transformers")
    for name in ("backend", "method"):
        if not isinstance(merged[name].get("parameters"), dict):
            raise ConfigurationError(f"{name}.parameters must be an object")
    if merged["runtime"]["capture"] not in {"none", "metadata", "full"}:
        raise ConfigurationError("runtime.capture must be none, metadata or full")
    if type(merged["runtime"]["seed"]) is not int:
        raise ConfigurationError("runtime.seed must be an integer")
    if merged["model"].get("trust_remote_code"):
        raise ConfigurationError("trust_remote_code is not supported by HALT's reference configuration")
    merged["model"].pop("trust_remote_code", None)
    if merged["backend"]["name"] == "scripted":
        from halt.backends.scripted import ScriptedBackend
        if merged["model"]:
            raise ConfigurationError("model settings are inapplicable to the scripted backend")
        try:
            inspect.signature(ScriptedBackend).bind(**merged["backend"]["parameters"])
        except TypeError as exc:
            raise ConfigurationError(f"invalid backend parameters: {exc}") from exc
    else:
        from halt.backends.transformers import TransformersBackend
        from halt.profiles.qwen3 import QWEN3_MODEL_ID, QWEN3_REVISION
        supplied_model = merged["model"]
        defaults = {"name": QWEN3_MODEL_ID, "model_profile": "qwen3_thinking", "device": "cpu",
                    "dtype": "float32", "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                    "local_files_only": False}
        if supplied_model.get("name", QWEN3_MODEL_ID) == QWEN3_MODEL_ID:
            defaults["revision"] = QWEN3_REVISION
        merged["model"] = {**defaults, **supplied_model}
        merged["model"].setdefault("tokenizer_revision", merged["model"].get("revision"))
        overlap = set(merged["backend"]["parameters"]) & set(merged["model"])
        if overlap:
            raise ConfigurationError(f"model parameters must occur only in model section: {sorted(overlap)}")
        arguments = {**merged["model"], **merged["backend"]["parameters"]}
        arguments["model_id"] = arguments.pop("name")
        try:
            inspect.signature(TransformersBackend.from_pretrained).bind(**arguments)
        except TypeError as exc:
            raise ConfigurationError(f"invalid model/backend parameters: {exc}") from exc
    make_task(merged["task"])
    registry = MethodRegistry()
    registry.create(merged["method"]["name"], merged["method"]["parameters"])
    evaluation = merged["evaluation"]
    for column in ("question_column", "answer_column", "id_column", "choices_column"):
        if column in evaluation and (not isinstance(evaluation[column], str) or not evaluation[column].strip()):
            raise ConfigurationError(f"evaluation.{column} must be a nonempty column name")
    if "choice_columns" in evaluation and (not isinstance(evaluation["choice_columns"], dict)
            or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v
                   for k, v in evaluation["choice_columns"].items())):
        raise ConfigurationError("evaluation.choice_columns must map answer labels to column names")
    for method in evaluation.get("methods", []):
        if not isinstance(method, dict):
            raise ConfigurationError("evaluation.methods entries must be objects")
        _unknown(method, {"name", "parameters", "id"}, "evaluation.methods")
        registry.create(method["name"], method.get("parameters", {}))
    if evaluation.get("dataset") and base_dir:
        path = Path(evaluation["dataset"])
        evaluation["dataset"] = str((base_dir / path).resolve() if not path.is_absolute() else path)
    if evaluation.get("protocol", "common_protocol") not in {"common_protocol", "source_recipe"}:
        raise ConfigurationError("evaluation.protocol must be common_protocol or source_recipe")
    if "bootstrap_samples" in evaluation and (type(evaluation["bootstrap_samples"]) is not int or evaluation["bootstrap_samples"] < 1):
        raise ConfigurationError("evaluation.bootstrap_samples must be a positive integer")
    return ResolvedConfig(**merged)


def load_config(path: str | Path) -> ResolvedConfig:
    source = Path(path)
    content = source.read_text(encoding="utf-8-sig")
    if source.suffix.lower() == ".toml":
        data = tomllib.loads(content)
    else:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            if source.suffix.lower() not in {".yaml", ".yml"}:
                raise ConfigurationError(f"invalid JSON configuration: {exc}") from exc
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as missing:
                raise ConfigurationError("YAML requires pip install 'halt-reasoning[evaluation]'; JSON/TOML need no extra") from missing
            data = yaml.safe_load(content)
    return validate_config(data, base_dir=source.parent)


def make_task(data: dict[str, Any]) -> Task:
    kind = data.get("type", "mcq")
    if kind in {"mcq", "mcq_jsonl"}:
        return MultipleChoiceTask(data["question"], data["choices"])
    if kind in {"numeric", "numeric_jsonl"}:
        if "choices" in data and data["choices"]:
            raise ConfigurationError("numeric tasks cannot include choices")
        return NumericTask(data["question"])
    raise ConfigurationError(f"unsupported task type {kind!r}")


def make_backend(config: ResolvedConfig) -> Any:
    factory: Any
    if config.backend["name"] == "scripted":
        from halt.backends.scripted import ScriptedBackend
        factory = ScriptedBackend
        parameters = dict(config.backend["parameters"])
    else:
        from halt.backends.transformers import TransformersBackend
        factory = TransformersBackend.from_pretrained
        parameters = {**config.model, **config.backend["parameters"]}
        parameters["model_id"] = parameters.pop("name", "Qwen/Qwen3-0.6B")
    try:
        inspect.signature(factory).bind(**parameters)
        return factory(**parameters)
    except TypeError as exc:
        raise ConfigurationError(f"invalid backend configuration: {exc}") from exc
