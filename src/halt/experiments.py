"""Small command-line experiments built on the same configuration and evaluator API."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from halt.config import ResolvedConfig, load_config, validate_config
from halt.errors import ConfigurationError
from halt.evaluation.datasets import load_dataset
from halt.evaluation.identifiers import method_id as _method_id
from halt.profiles.qwen3 import QWEN3_MODEL_ID
from halt.registry import MethodRegistry

DATA_OPTIONS = ("question_column", "answer_column", "id_column", "choices_column", "choice_columns")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter", choices=("auto", "mcq", "numeric", "mcq_jsonl", "numeric_jsonl"))
    parser.add_argument("--limit", type=positive_int, help="evaluate only the first N questions; validate the entire file")
    for field in DATA_OPTIONS[:-1]:
        parser.add_argument("--" + field.replace("_", "-"))
    parser.add_argument("--choice-columns", help="answer-label mappings, e.g. A=option_a,B=option_b")


def model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("transformers", "scripted"))
    parser.add_argument("--model", help="Hugging Face model name; currently decoder-only Qwen3")
    parser.add_argument("--revision", help="model commit, tag, or branch; saved as an immutable commit")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--device", help="e.g. cpu or cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true", default=None)


def dataset_options(args: argparse.Namespace) -> dict[str, Any]:
    options = {key: getattr(args, key) for key in DATA_OPTIONS[:-1] if getattr(args, key, None) is not None}
    mapping = getattr(args, "choice_columns", None)
    if mapping is not None:
        pairs = [part.partition("=") for part in mapping.split(",")]
        if any(not key.strip() or not separator or not value.strip() for key, separator, value in pairs):
            raise ConfigurationError("--choice-columns must be A=column_a,B=column_b")
        labels = [key.strip() for key, _, _ in pairs]
        if len(set(labels)) != len(labels):
            raise ConfigurationError("--choice-columns contains duplicate answer labels")
        options["choice_columns"] = {key.strip(): value.strip() for key, _, value in pairs}
    return options


def _model_overrides(data: dict[str, Any], args: argparse.Namespace) -> None:
    model = data.setdefault("model", {})
    if getattr(args, "backend", None) and args.backend != data.get("backend", {}).get("name"):
        data["backend"] = {"name": args.backend}
        if args.backend == "scripted":
            model = data["model"] = {}
    if getattr(args, "model", None):
        if getattr(args, "backend", None) == "scripted":
            raise ConfigurationError("--model cannot be combined with --backend scripted")
        if data["backend"]["name"] != "transformers":
            data["backend"] = {"name": "transformers"}
        if args.model != model.get("name"):
            model.pop("revision", None)
            model.pop("tokenizer_revision", None)
        model["name"] = args.model
    for key in ("revision", "tokenizer_revision", "device", "dtype", "cache_dir", "local_files_only"):
        value = getattr(args, key, None)
        if value is not None:
            if data["backend"]["name"] == "scripted":
                raise ConfigurationError(f"--{key.replace('_', '-')} requires the transformers backend")
            model[key] = value
            if key == "revision" and getattr(args, "tokenizer_revision", None) is None:
                model["tokenizer_revision"] = value


def _matching_methods(entries: list[dict[str, Any]], selector: str) -> list[dict[str, Any]]:
    exact = [entry for entry in entries if _method_id(entry) == selector]
    return exact or [entry for entry in entries if entry["name"] == selector]


def trial_config(args: argparse.Namespace, *, benchmark: bool = True) -> ResolvedConfig:
    supplied = bool(getattr(args, "config", None))
    data = load_config(args.config).to_dict() if supplied else {
        "backend": {"name": "transformers"},
        "model": {"name": QWEN3_MODEL_ID, "cache_dir": ".model-cache"},
        "budget": {"max_reasoning_tokens": 128, "max_answer_tokens": 64,
                   "max_probe_output_tokens": 128, "max_total_generated_tokens": 320, "max_probe_calls": 8},
    }
    _model_overrides(data, args)
    if not benchmark:
        return validate_config(data)
    evaluation = data.setdefault("evaluation", {})
    dataset_path = getattr(args, "dataset", None)
    if dataset_path:
        evaluation["dataset"] = str(Path(dataset_path).resolve())
        # A replacement file must not inherit the old file's importer/mappings.
        evaluation["adapter"] = "auto"
        evaluation["data_revision"] = "local"
        for key in DATA_OPTIONS:
            evaluation.pop(key, None)
    if not evaluation.get("dataset"):
        raise ConfigurationError("Supply --dataset questions.csv or --config experiment.json with evaluation.dataset.")
    if getattr(args, "adapter", None):
        evaluation["adapter"] = args.adapter
    evaluation.setdefault("adapter", "auto")
    evaluation.update(dataset_options(args))
    if getattr(args, "limit", None) is not None:
        evaluation["limit"] = args.limit
    dataset = load_dataset(evaluation["dataset"], adapter=evaluation["adapter"],
        limit=evaluation.get("limit"), **{k: evaluation[k] for k in DATA_OPTIONS if k in evaluation})
    # Keep a representative inference-only task for doctor/requirements inspection.
    visible = dataset.items[0].task.visible()
    data["task"] = {"type": "numeric" if visible["adapter"] == "numeric_v1" else "mcq", "question": visible["question"]}
    if "choices" in visible:
        data["task"]["choices"] = visible["choices"]
    configured = evaluation.get("methods", [])
    requested = getattr(args, "methods", None)
    if requested:
        names = [name.strip() for part in requested for name in part.split(",") if name.strip()]
        if not names or len(names) != len(set(names)):
            raise ConfigurationError("--methods needs distinct method names; use a config with distinct IDs for parameter sweeps")
        selected = []
        for name in names:
            matches = _matching_methods(configured, name)
            if len(matches) > 1:
                raise ConfigurationError(f"method {name!r} matches multiple configured variants; select their evaluation IDs")
            fallback = data.get("method", {})
            selected.append(deepcopy(matches[0]) if matches else (
                deepcopy(fallback) if fallback.get("name") == name else {"name": name}))
        evaluation["methods"] = selected
    elif not supplied:
        other = "answer_convergence" if data["task"]["type"] == "numeric" else "halt_cot"
        evaluation["methods"] = [{"name": "full_reasoning"}, {"name": other}]
    if not evaluation.get("methods"):
        method = data.get("method", {"name": "full_reasoning"})
        evaluation["methods"] = [{"name": "full_reasoning"}] + ([method] if method["name"] != "full_reasoning" else [])
    baseline = getattr(args, "baseline", None) or evaluation.get("baseline", "full_reasoning")
    baseline_matches = _matching_methods(evaluation["methods"], baseline)
    if len(baseline_matches) > 1:
        raise ConfigurationError(f"baseline {baseline!r} matches multiple variants; select an evaluation ID")
    if not baseline_matches:
        configured_baselines = _matching_methods(configured, baseline)
        if len(configured_baselines) > 1:
            raise ConfigurationError(f"baseline {baseline!r} matches multiple configured variants; select an evaluation ID")
        baseline_entry = deepcopy(configured_baselines[0]) if configured_baselines else {"name": baseline}
        evaluation["methods"].insert(0, baseline_entry)
    else:
        baseline_entry = baseline_matches[0]
    for assignment in getattr(args, "param", []) or []:
        left, separator, value = assignment.partition("=")
        method_id, dot, parameter = left.partition(".")
        if not separator or not dot or not parameter:
            raise ConfigurationError("--param must be METHOD.PARAMETER=JSON_VALUE, e.g. halt_cot.threshold=0.5")
        matches = _matching_methods(evaluation["methods"], method_id)
        if len(matches) != 1:
            raise ConfigurationError(f"--param refers to an unselected or ambiguous method: {method_id!r}")
        try:
            parsed = json.loads(value)
        except ValueError as exc:
            raise ConfigurationError("--param values must be JSON numbers, booleans, strings, or arrays") from exc
        matches[0].setdefault("parameters", {})[parameter] = parsed
    evaluation["baseline"] = _method_id(baseline_entry)
    selected_ids = [_method_id(entry) for entry in evaluation["methods"]]
    if len(set(selected_ids)) != len(selected_ids):
        raise ConfigurationError("duplicate evaluation method IDs; use distinct id fields for parameter sweeps")
    data.setdefault("runtime", {})
    if getattr(args, "seed", None) is not None:
        data["runtime"]["seed"] = args.seed
    budget = data.setdefault("budget", {})
    changed_budget = False
    for key in ("max_reasoning_tokens", "max_answer_tokens", "max_probe_output_tokens", "max_probe_calls"):
        if getattr(args, key, None) is not None:
            budget[key] = getattr(args, key)
            changed_budget = changed_budget or key != "max_probe_calls"
    if getattr(args, "max_total_generated_tokens", None) is not None:
        budget["max_total_generated_tokens"] = args.max_total_generated_tokens
    elif changed_budget and not supplied:
        budget["max_total_generated_tokens"] = sum(budget.get(k, 0) for k in ("max_reasoning_tokens", "max_answer_tokens", "max_probe_output_tokens"))
    config = validate_config(data)
    if not 0 <= config.runtime["seed"] < 2**63:
        raise ConfigurationError("runtime.seed must be in [0, 2**63)")
    # Rule-specific task checks precede optional dependencies, network, and weights.
    registry = MethodRegistry()
    for entry in evaluation["methods"]:
        method = registry.create(entry["name"], entry.get("parameters", {}))
        validate = getattr(method, "validate_task", None)
        if callable(validate):
            for item in dataset.items:
                validate(item.task)
    return config
