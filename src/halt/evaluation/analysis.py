"""Dependency-free loading and comparisons of HALT's versioned measurement rows."""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from halt.errors import ConfigurationError
from halt.types import SCHEMA_VERSION, stable_hash

RESULTS_SCHEMA_VERSION = "1.0"
WORK_FIELDS = ("reasoning_tokens", "answer_tokens", "probe_output_tokens", "total_generated_tokens",
               "input_tokens", "forced_context_tokens", "probe_calls", "recomputed_prefix_tokens",
               "scored_tokens", "embedding_calls", "forward_calls", "cancellation_requests")
JSON_FIELDS = ("choices", "component_seconds", "method_spec", "hardware", "diagnostics")
BOOL_FIELDS = ("correct", "adaptive_session")
INT_FIELDS = ("seed", "order", "stop_position", *WORK_FIELDS)
FLOAT_FIELDS = ("elapsed_seconds",)
RESULT_FIELDS = ("results_schema_version", "comparison_id", "identity", "question_id", "method_id",
    "method_configuration_id", "seed", "order", "dataset_sha256", "dataset_split", "model_id",
    "model_revision", "tokenizer_revision", "backend", "halt_source_sha256", "evidence_kind",
    "timing_scope", "question", "choices", "reference", "answer", "correct", "status", "error",
    "stop_reason", "stop_position", "adaptive_session", "session_id", *WORK_FIELDS,
    "elapsed_seconds", "component_seconds", "usage_measurement", "method_spec", "hardware",
    "diagnostics", "trace_reference")


def measurement_row(record: dict[str, Any]) -> dict[str, Any]:
    """Project a full evaluator record into one stable row; never discard failed runs."""
    if "results_schema_version" in record:
        _validate_row(record)
        return {key: record.get(key) for key in RESULT_FIELDS}
    manifest, result = record["manifest"], record["result"]
    backend, dataset = manifest["backend"], manifest["dataset"]
    runtime = dict(manifest.get("runtime", {}))
    seed = runtime.pop("seed", result.get("provenance", {}).get("seed", 0))
    # Capture level changes neither the question nor generation settings.
    runtime.pop("capture", None)
    identity_parts = {key: manifest.get(key) for key in (
        "environment", "backend", "budget", "model_configuration", "backend_configuration",
        "dataset", "protocol")}
    identity_parts["runtime"] = runtime
    evaluation = record.get("evaluation", {})
    durations = result.get("durations", {})
    elapsed = durations.get("end_to_end", durations.get("total"))
    if elapsed is None and durations:
        elapsed = sum(durations.values())
    evidence = record["evidence_kind"]
    session = record.get("session_before")
    usage = result["usage"]
    row = {"results_schema_version": RESULTS_SCHEMA_VERSION,
        "comparison_id": stable_hash(identity_parts), "identity": record["identity"],
        "question_id": record["item_id"], "method_id": record["method_id"],
        "method_configuration_id": stable_hash({"configuration": manifest.get("method_configuration"),
                                                "spec": manifest.get("method_spec")}),
        "seed": seed, "order": manifest.get("order", 0), "dataset_sha256": dataset["content_sha256"],
        "dataset_split": dataset.get("split", ""), "model_id": backend.get("model_id", ""),
        "model_revision": backend.get("model_revision", ""),
        "tokenizer_revision": backend.get("tokenizer_revision", ""), "backend": backend.get("name", ""),
        "halt_source_sha256": manifest.get("environment", {}).get("halt_source_sha256", ""),
        "evidence_kind": evidence,
        "timing_scope": "controller_only" if evidence == "replay" else (
            "simulated_runtime" if evidence == "simulated" else "inference_end_to_end"),
        "question": evaluation.get("question", ""), "choices": evaluation.get("choices", {}),
        "reference": evaluation.get("reference", ""), "answer": result.get("answer"),
        "correct": record["correct"], "status": result["status"], "error": result.get("error"),
        "stop_reason": result["stop"]["reason"], "stop_position": result["stop"]["reasoning_position"],
        "adaptive_session": session is not None, "session_id": session.get("session_id", "") if session else "",
        **{key: usage.get(key, 0) for key in WORK_FIELDS}, "elapsed_seconds": elapsed,
        "component_seconds": durations, "usage_measurement": usage.get("measurement", "unknown"),
        "method_spec": manifest.get("method_spec", {}), "hardware": backend.get("hardware", {}),
        "diagnostics": result["stop"].get("diagnostics", {}), "trace_reference": result.get("trace_reference")}
    _validate_row(row)
    return row


def _validate_row(row: dict[str, Any]) -> None:
    if row.get("results_schema_version") != RESULTS_SCHEMA_VERSION:
        raise ConfigurationError(f"unsupported results schema {row.get('results_schema_version')!r}")
    missing = set(RESULT_FIELDS) - row.keys()
    if missing:
        raise ConfigurationError("missing result fields: " + ", ".join(sorted(missing)))
    for key in BOOL_FIELDS:
        if type(row[key]) is not bool:
            raise ConfigurationError(f"result {key} must be a boolean")
    if row["correct"] and row["status"] != "completed":
        raise ConfigurationError("only completed results can be marked correct")
    for key in JSON_FIELDS:
        if not isinstance(row[key], dict):
            raise ConfigurationError(f"result {key} must be an object")
    if row["evidence_kind"] not in {"real", "simulated", "replay"}:
        raise ConfigurationError("result evidence_kind must be real, simulated, or replay")
    for key in INT_FIELDS:
        if key in row and (type(row[key]) is not int or row[key] < 0):
            raise ConfigurationError(f"result {key} must be a nonnegative integer")
    elapsed = row.get("elapsed_seconds")
    if elapsed is not None and (isinstance(elapsed, bool) or not isinstance(elapsed, int | float)
                                or not math.isfinite(elapsed) or elapsed < 0):
        raise ConfigurationError("result elapsed_seconds must be finite, nonnegative, or null")
    generated = sum(row[key] for key in ("reasoning_tokens", "answer_tokens", "probe_output_tokens"))
    if row["total_generated_tokens"] != generated:
        raise ConfigurationError("total_generated_tokens must include reasoning, answer, and probe output")


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Load a run directory, results CSV/JSONL, or legacy per_example JSONL into plain dicts.

    Returned records work directly with ``pandas.DataFrame`` without importing pandas here.
    A directory means results.jsonl, falling back to legacy per_example.jsonl.
    """
    source = Path(path)
    if source.is_dir():
        source = source / ("results.jsonl" if (source / "results.jsonl").exists() else "per_example.jsonl")
    try:
        rows = []
        if source.suffix.lower() == ".csv":
            with source.open(newline="", encoding="utf-8-sig") as handle:
                for number, raw in enumerate(csv.DictReader(handle), 2):
                    try:
                        row: dict[str, Any] = dict(raw)
                        for key in JSON_FIELDS:
                            row[key] = json.loads(row[key])
                        for key in BOOL_FIELDS:
                            if row[key] not in {"true", "false", "True", "False"}:
                                raise ValueError(f"{key} must be true or false")
                            row[key] = row[key].lower() == "true"
                        for key in INT_FIELDS:
                            row[key] = int(row[key])
                        for key in FLOAT_FIELDS:
                            row[key] = float(row[key]) if row[key] != "" else None
                        for key in ("answer", "error", "trace_reference"):
                            row[key] = row[key] or None
                        rows.append(measurement_row(row))
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ConfigurationError(f"invalid results CSV row {number}: {exc}") from exc
        else:
            for number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
                if line.strip():
                    try:
                        rows.append(measurement_row(json.loads(line)))
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ConfigurationError(f"invalid results JSONL line {number}: {exc}") from exc
    except OSError as exc:
        raise ConfigurationError(f"cannot load results {source}: {exc}") from exc
    if not rows:
        raise ConfigurationError("results file contains no observations")
    return rows


def paired_bootstrap(differences: list[float], *, samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not differences or samples < 1:
        raise ConfigurationError("bootstrap needs paired observations and positive sample count")
    rng = random.Random(seed)
    values = sorted(statistics.fmean(rng.choices(differences, k=len(differences))) for _ in range(samples))
    return values[math.floor(0.025 * (samples - 1))], values[math.ceil(0.975 * (samples - 1))]


def compare(records: Iterable[dict[str, Any]], *, baseline: str = "full_reasoning",
            bootstrap_samples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Compare compatible runs, paired by question and seed, with failures in the denominator.

    Reject mixed datasets/models/settings/code and duplicate observations. For separate
    experiments use ``grouped_compare``. Adaptive or repeated-question multi-seed runs
    retain point estimates but have no iid item bootstrap interval.
    """
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ConfigurationError("bootstrap_samples must be a positive integer")
    data = [measurement_row(record) for record in records]
    if not data:
        raise ConfigurationError("comparison requires at least one observation")
    if len({r["comparison_id"] for r in data}) != 1 or len({r["evidence_kind"] for r in data}) != 1:
        raise ConfigurationError("incompatible experiments; compare each comparison_id separately with grouped_compare")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in data:
        groups.setdefault(record["method_id"], []).append(record)
    for name, group in groups.items():
        keys = {(r["question_id"], r["seed"]) for r in group}
        if len(keys) != len(group):
            raise ConfigurationError(f"duplicate question/seed observations for method {name!r}")
        if len({r["method_configuration_id"] for r in group}) != 1:
            raise ConfigurationError(f"method {name!r} has different configurations; assign distinct method IDs")
    reference = {(r["question_id"], r["seed"]): r for r in groups.get(baseline, [])}
    if not reference:
        raise ConfigurationError(f"baseline {baseline!r} is absent from evaluated methods")
    rows = []
    for name, group in groups.items():
        n = len(group)
        row: dict[str, Any] = {"method_id": name, "sample_count": n,
            "accuracy": statistics.fmean(r["correct"] for r in group),
            "parse_failure_rate": statistics.fmean(r["answer"] is None and r["status"] != "abstained" for r in group),
            "incomplete_answer_rate": statistics.fmean(r["status"] == "answer_incomplete" for r in group),
            "abstention_rate": statistics.fmean(r["status"] == "abstained" for r in group),
            "failure_count": sum(r["status"] != "completed" for r in group),
            "status_counts": dict(Counter(r["status"] for r in group)),
            "natural_completion_rate": statistics.fmean(r["stop_reason"] in {"natural_completion", "natural_end", "natural_reasoning_end"} for r in group),
            "budget_limited_rate": statistics.fmean(r["status"] == "budget_exhausted" or "budget" in r["stop_reason"] for r in group),
            "stop_reasons": dict(Counter(r["stop_reason"] for r in group)),
            "stop_positions": [r["stop_position"] for r in group],
            "evidence_kind": sorted({r["evidence_kind"] for r in group}),
            "recipe": group[0]["method_spec"], "resources": group[0]["hardware"],
            "batch_size": 1, "concurrency": 1, "peak_memory": None, "peak_memory_measurement": "not measured"}
        for key in WORK_FIELDS:
            row[f"mean_{key}"] = statistics.fmean(r[key] for r in group)
            row[f"total_{key}"] = sum(r[key] for r in group)
        durations = [r["component_seconds"] for r in group]
        components = sorted(set().union(*(set(d) for d in durations)))
        row["mean_component_seconds"] = {key: statistics.fmean(d.get(key, 0.0) for d in durations) for key in components}
        latency = [r["elapsed_seconds"] for r in group if r["elapsed_seconds"] is not None]
        row["timing_sample_count"] = len(latency)
        row["mean_latency_seconds"] = statistics.fmean(latency) if latency else None
        row["timing_scope"] = group[0]["timing_scope"]
        if len(latency) >= 20:
            row["p50_latency_seconds"] = statistics.median(latency)
            row["p95_latency_seconds"] = sorted(latency)[math.ceil(0.95 * len(latency)) - 1]
        paired = [(r, reference[r["question_id"], r["seed"]]) for r in group if (r["question_id"], r["seed"]) in reference]
        differences = [float(a["correct"]) - float(b["correct"]) for a, b in paired]
        row["paired_sample_count"] = len(paired)
        row["unpaired_sample_count"] = n - len(paired)
        row["baseline_unmatched_sample_count"] = len(reference) - len(paired)
        row["paired_accuracy_difference"] = statistics.fmean(differences) if differences else None
        row["correct_to_wrong"] = sum(b["correct"] and not a["correct"] for a, b in paired)
        row["wrong_to_correct"] = sum(a["correct"] and not b["correct"] for a, b in paired)
        adaptive = any(r["adaptive_session"] for r in group) or any(r["adaptive_session"] for r in reference.values())
        repeated_questions = len({a["question_id"] for a, _ in paired}) != len(paired)
        row["adaptive_session"] = any(r["adaptive_session"] for r in group)
        row["paired_accuracy_ci95"] = None if adaptive or repeated_questions or not differences else paired_bootstrap(differences, samples=bootstrap_samples, seed=seed)
        interval = row["paired_accuracy_ci95"]
        row["paired_accuracy_ci95_low"] = interval[0] if interval is not None else None
        row["paired_accuracy_ci95_high"] = interval[1] if interval is not None else None
        denominator = sum(b["total_generated_tokens"] for _, b in paired)
        numerator = sum(a["total_generated_tokens"] for a, _ in paired)
        row["generated_reduction_baseline_denominator"] = denominator
        row["aggregate_generated_token_reduction"] = 1 - numerator / denominator if denominator else None
        row["uncertainty_note"] = (
            "Ordered dependent session; iid item bootstrap is inapplicable. Repeat independent sessions for uncertainty." if adaptive else
            "Repeated questions across seeds; iid observation bootstrap is inapplicable. Use an analysis accounting for repeated questions." if repeated_questions else
            "No matched question/seed pairs; paired effects and intervals are unavailable." if not differences else
            "Paired percentile bootstrap over examples; small samples may give uninformative intervals.")
        rows.append(row)
    first = data[0]
    evidence = first["evidence_kind"]
    return {"schema_version": SCHEMA_VERSION, "results_schema_version": RESULTS_SCHEMA_VERSION,
        "comparison_id": first["comparison_id"], "baseline": baseline, "methods": rows,
        "experiment": {key: first[key] for key in ("model_id", "model_revision", "backend", "dataset_sha256", "dataset_split", "evidence_kind")},
        "seeds": sorted({r["seed"] for r in data}),
        "interpretation": ("Simulated demonstration: generated fragments and timings are synthetic mechanics, not model performance." if evidence == "simulated" else
            "Replay demonstration: timings measure controller work only, not model inference." if evidence == "replay" else
            "Measured runs on the specified dataset and model. Accuracy includes failures; timing includes probes and finalization, excluding model loading. Small trials do not establish general efficiency.")}


def grouped_compare(records: Iterable[dict[str, Any]], *, by: Sequence[str] = ("comparison_id",),
                    baseline: str = "full_reasoning", bootstrap_samples: int = 2000,
                    seed: int = 0) -> list[dict[str, Any]]:
    """Produce separate comparisons by field(s); each group still requires compatible runs."""
    if not by or isinstance(by, str):
        raise ConfigurationError("by must be a nonempty sequence of field names, e.g. ('comparison_id',)")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        row = measurement_row(record)
        try:
            key = tuple(row[field] for field in by)
            groups.setdefault(key, []).append(row)
        except (KeyError, TypeError) as exc:
            raise ConfigurationError("group fields must be existing scalar result fields") from exc
    return [{"group": dict(zip(by, key, strict=True)), "comparison": compare(group, baseline=baseline,
             bootstrap_samples=bootstrap_samples, seed=seed)} for key, group in groups.items()]
