"""Empirical split-declared threshold search, without a risk guarantee."""
from __future__ import annotations

import itertools
import json
import tempfile
from pathlib import Path
from typing import Any

from halt.config import ResolvedConfig, make_backend, validate_config
from halt.errors import ConfigurationError
from halt.evaluation.datasets import load_dataset
from halt.evaluation.evaluator import environment_identity, evaluate
from halt.types import SCHEMA_VERSION, stable_hash, to_data


def calibrate(config: ResolvedConfig, output: str | Path) -> dict[str, Any]:
    settings = config.calibration
    if config.evaluation.get("split") != "calibration":
        raise ConfigurationError("calibration requires evaluation.split='calibration'; never tune on held-out test labels")
    if settings.get("recipe_frozen") is not True:
        raise ConfigurationError("set calibration.recipe_frozen=true after selecting the recipe on development data")
    if config.method.get("parameters", {}).get("adaptive"):
        raise ConfigurationError("REFRAIN online session adaptation is separate from offline empirical calibration")
    grid = settings.get("parameter_grid")
    if not isinstance(grid, dict) or not grid or any(not isinstance(x, list) or not x for x in grid.values()):
        raise ConfigurationError("calibration.parameter_grid must map parameter names to nonempty lists")
    tolerance = settings.get("accuracy_tolerance", 0.0)
    if not isinstance(tolerance, float | int) or not 0 <= tolerance <= 1:
        raise ConfigurationError("accuracy_tolerance must be between zero and one")
    objective = settings.get("objective", "mean_total_generated_tokens")
    if objective not in {"mean_total_generated_tokens", "mean_input_tokens", "mean_forward_calls", "mean_latency_seconds"}:
        raise ConfigurationError("unsupported empirical calibration objective")
    settings_list = [{**config.method["parameters"], **dict(zip(grid, values, strict=True))}
                     for values in itertools.product(*grid.values())]
    backend = make_backend(config)
    data = config.to_dict()
    data["evaluation"]["methods"] = [{"name": "full_reasoning"}] + [
        {"name": config.method["name"], "parameters": parameters, "id": f"calibration_{i}"}
        for i, parameters in enumerate(settings_list)]
    data["evaluation"]["baseline"] = "full_reasoning"
    data["evaluation"]["resume"] = False
    with tempfile.TemporaryDirectory(prefix="halt-calibration-") as directory:
        summary = evaluate(validate_config(data), directory, backend=backend)
    baseline = summary["methods"][0]
    eligible = [row for row in summary["methods"][1:] if row["accuracy"] >= baseline["accuracy"] - tolerance]
    chosen = min(eligible, key=lambda row: (row[objective], -row["accuracy"], row["method_id"])) if eligible else None
    dataset = load_dataset(config.evaluation["dataset"], adapter=config.evaluation.get("adapter", "mcq_jsonl"),
                           split="calibration", revision=config.evaluation.get("data_revision", "local"),
                           limit=config.evaluation.get("limit"))
    artifact = {"schema_version": SCHEMA_VERSION, "kind": "empirical_calibration", "statistical_guarantee": False,
        "environment": environment_identity(), "backend": to_data(backend.info), "dataset": dataset.identity(),
        "resolved_configuration": config.to_dict(), "objective": objective, "accuracy_tolerance": tolerance,
        "tested_parameters": settings_list, "results": summary,
        "chosen_parameters": settings_list[int(chosen["method_id"].split("_")[-1])] if chosen else None,
        "status": "selected" if chosen else "no_feasible_setting",
        "instruction": "Freeze chosen parameters and use a disjoint held-out split. Observed tolerance is not a statistical correctness guarantee."}
    artifact["artifact_id"] = stable_hash(artifact)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    return artifact
