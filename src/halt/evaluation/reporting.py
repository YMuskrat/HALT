from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from halt.errors import ConfigurationError
from halt.types import SCHEMA_VERSION

_WORK = ("reasoning_tokens", "answer_tokens", "probe_output_tokens", "total_generated_tokens",
         "input_tokens", "forced_context_tokens", "probe_calls", "recomputed_prefix_tokens",
         "scored_tokens", "embedding_calls", "forward_calls")


def paired_bootstrap(differences: list[float], *, samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not differences or samples < 1:
        raise ConfigurationError("bootstrap needs paired observations and positive sample count")
    rng = random.Random(seed)
    values = sorted(statistics.fmean(rng.choices(differences, k=len(differences))) for _ in range(samples))
    return values[math.floor(0.025 * (samples - 1))], values[math.ceil(0.975 * (samples - 1))]


def summarize(records: list[dict[str, Any]], *, baseline: str, bootstrap_samples: int = 2000,
              seed: int = 0) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["method_id"], []).append(record)
    reference = {r["item_id"]: r for r in groups.get(baseline, [])}
    if not reference:
        raise ConfigurationError(f"baseline {baseline!r} is absent from evaluated methods")
    rows = []
    for name, group in groups.items():
        n = len(group)
        row: dict[str, Any] = {"method_id": name, "sample_count": n,
            "accuracy": statistics.fmean(r["correct"] for r in group),
            "parse_failure_rate": statistics.fmean(r["result"]["answer"] is None and r["result"]["status"] != "abstained" for r in group),
            "incomplete_answer_rate": statistics.fmean(r["result"]["status"] == "answer_incomplete" for r in group),
            "abstention_rate": statistics.fmean(r["result"]["status"] == "abstained" for r in group),
            "natural_completion_rate": statistics.fmean(r["result"]["stop"]["reason"] in {"natural_completion", "natural_end", "natural_reasoning_end"} for r in group),
            "budget_limited_rate": statistics.fmean(r["result"]["status"] == "budget_exhausted" or "budget" in r["result"]["stop"]["reason"] for r in group),
            "stop_reasons": dict(Counter(r["result"]["stop"]["reason"] for r in group)),
            "stop_positions": [r["result"]["stop"]["reasoning_position"] for r in group],
            "evidence_kind": sorted({r["evidence_kind"] for r in group}),
            "recipe": group[0]["manifest"]["method_spec"],
            "resources": group[0]["manifest"]["backend"].get("hardware", {}),
            "batch_size": 1, "concurrency": 1,
            "peak_memory": None,
            "peak_memory_measurement": "not measured"}
        for key in _WORK:
            row[f"mean_{key}"] = statistics.fmean(r["result"]["usage"].get(key, 0) for r in group)
            row[f"total_{key}"] = sum(r["result"]["usage"].get(key, 0) for r in group)
        durations = [r["result"].get("durations", {}) for r in group]
        row["mean_component_seconds"] = {key: statistics.fmean(x.get(key, 0.0) for x in durations) for key in sorted(set().union(*(set(x) for x in durations)))}
        latency = [d.get("end_to_end", d.get("total", sum(d.values()))) for d in durations]
        row["mean_latency_seconds"] = statistics.fmean(latency)
        row["timing_scope"] = "controller_only" if "replay" in row["evidence_kind"] else ("simulated_runtime" if row["evidence_kind"] == ["simulated"] else "inference_end_to_end")
        if n >= 20:
            ordered = sorted(latency)
            row["p50_latency_seconds"] = statistics.median(ordered)
            row["p95_latency_seconds"] = ordered[math.ceil(0.95 * n) - 1]
        paired = [(r, reference[r["item_id"]]) for r in group if r["item_id"] in reference]
        differences = [float(a["correct"]) - float(b["correct"]) for a, b in paired]
        row["paired_sample_count"] = len(paired)
        row["paired_accuracy_difference"] = statistics.fmean(differences) if differences else None
        row["correct_to_wrong"] = sum(b["correct"] and not a["correct"] for a, b in paired)
        row["wrong_to_correct"] = sum(a["correct"] and not b["correct"] for a, b in paired)
        adaptive = any(r.get("session_before") is not None for r in group)
        row["adaptive_session"] = adaptive
        row["paired_accuracy_ci95"] = None if adaptive or not differences else paired_bootstrap(differences, samples=bootstrap_samples, seed=seed)
        denominator = sum(b["result"]["usage"]["total_generated_tokens"] for _, b in paired)
        numerator = sum(a["result"]["usage"]["total_generated_tokens"] for a, _ in paired)
        row["generated_reduction_baseline_denominator"] = denominator
        row["aggregate_generated_token_reduction"] = 1 - numerator / denominator if denominator else None
        row["uncertainty_note"] = "Ordered dependent session; iid item bootstrap is inapplicable. Repeat independent sessions for uncertainty." if adaptive else "Paired percentile bootstrap over examples; small samples may give uninformative intervals."
        rows.append(row)
    return {"schema_version": SCHEMA_VERSION, "baseline": baseline, "methods": rows,
            "interpretation": "Original toy data and simulated runs demonstrate mechanics only. No general efficiency claim follows from this report."}


def write_reports(records: list[dict[str, Any]], output_dir: str | Path, *, baseline: str,
                  bootstrap_samples: int = 2000, seed: int = 0) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summary = summarize(records, baseline=baseline, bootstrap_samples=bootstrap_samples, seed=seed)
    (directory / "per_example.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    fields = [key for key, value in summary["methods"][0].items() if not isinstance(value, dict | list | tuple)]
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["methods"])
    lines = ["# HALT evaluation", "", summary["interpretation"], "",
             "| Method | Evidence | N | Accuracy | Paired change (95% CI) | Mean generated tokens | Mean probe calls |", "|---|---|---:|---:|---|---:|---:|"]
    for row in summary["methods"]:
        interval = row["paired_accuracy_ci95"]
        change = f"{row['paired_accuracy_difference']:+.3f}"
        change += f" [{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval is not None else " (ordered session; no iid CI)"
        lines.append(f"| {row['method_id']} | {', '.join(row['evidence_kind'])} | {row['sample_count']} | {row['accuracy']:.3f} | {change} | {row['mean_total_generated_tokens']:.1f} | {row['mean_probe_calls']:.1f} |")
    lines.extend(["", "Latency scope and complete work/resource records are in summary.json. Percentiles are omitted below 20 examples. Every output is evaluated against an evaluator-only reference label. Resolved configurations and run identities are in manifest.json and per_example.jsonl.", ""])
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
