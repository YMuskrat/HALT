"""Independent evaluation repeats with separate adaptive state and checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from halt.config import ResolvedConfig, make_backend
from halt.errors import ConfigurationError
from halt.evaluation.evaluator import environment_identity, evaluate
from halt.evaluation.uncertainty import cluster_bootstrap
from halt.types import stable_hash


def evaluate_repeats(config: ResolvedConfig, output_dir: str | Path, *, seeds: Sequence[int],
                     backend_factory: Callable[[], Any] | None = None) -> list[dict[str, Any]]:
    """Evaluate distinct seeds with fresh backends, sessions and checkpoint directories.

    The same dataset order is used in every repeat. Callers supplying a factory
    must return a fresh backend each time. Resume requires exactly the original
    config, seed list, dataset bytes and source environment identity.
    """
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ConfigurationError("repeat seeds must be distinct nonnegative integers")
    if not config.evaluation.get("dataset"):
        raise ConfigurationError("evaluation.dataset is required")
    dataset = Path(config.evaluation["dataset"])
    try:
        digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    except OSError as exc:
        raise ConfigurationError(f"cannot read repeat dataset: {exc}") from exc
    identity = {"version": 1, "configuration": config.to_dict(), "seeds": list(seeds),
                "dataset_sha256": digest, "environment": environment_identity()}
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "repeats.json"
    if manifest.exists():
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ConfigurationError("invalid repeat manifest") from exc
        if saved != identity or not config.evaluation.get("resume", True):
            raise ConfigurationError("repeat output already exists with incompatible identity or resume disabled")
    else:
        if any(root.iterdir()):
            raise ConfigurationError("new repeat output directory must be empty")
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump(identity, handle, indent=2, allow_nan=False)
    summaries = []
    for index, seed in enumerate(seeds):
        settings = copy.deepcopy(config.evaluation)
        settings["session"] = {**settings.get("session", {}),
            "session_id": stable_hash({"identity": identity, "repeat": index, "seed": seed})[:32]}
        repeated = replace(config, evaluation=settings, runtime={**config.runtime, "seed": seed})
        backend = backend_factory() if backend_factory else make_backend(repeated)
        summary = evaluate(repeated, root / f"repeat-{index:04d}", backend=backend)
        summaries.append(summary)
    return summaries


def summarize_repeats(summaries: Sequence[dict[str, Any]], *, samples: int = 2000,
                      seed: int = 0) -> dict[str, Any]:
    """Aggregate repeat effects equally; never bootstrap questions within sessions.

    Input must contain one result per independently initialized repeat. This
    function checks identity/seed compatibility, not physical RNG independence.
    """
    if not summaries:
        raise ConfigurationError("at least one repeat summary is required")
    first = summaries[0]
    expected = {method["method_id"] for method in first["methods"]}
    observed_seeds: set[int] = set()
    for summary in summaries:
        if (summary["comparison_id"] != first["comparison_id"] or
                summary["baseline"] != first["baseline"] or
                summary["experiment"] != first["experiment"]):
            raise ConfigurationError("repeat summaries describe incompatible experiments")
        seeds = summary["seeds"]
        if len(seeds) != 1 or seeds[0] in observed_seeds:
            raise ConfigurationError("repeat summaries require one distinct seed each")
        observed_seeds.add(seeds[0])
        if {method["method_id"] for method in summary["methods"]} != expected:
            raise ConfigurationError("repeat summaries must contain the same methods")
    methods = []
    for name in sorted(expected):
        rows = [next(m for m in summary["methods"] if m["method_id"] == name) for summary in summaries]
        if any(row["recipe"] != rows[0]["recipe"] for row in rows):
            raise ConfigurationError("repeat method recipes differ")
        effects = [row["paired_accuracy_difference"] for row in rows]
        complete = all(effect is not None and row["unpaired_sample_count"] == 0 and
                       row["baseline_unmatched_sample_count"] == 0 for row, effect in zip(rows, effects, strict=True))
        interval = (cluster_bootstrap([[float(effect)] for effect in effects], samples=samples, seed=seed)
                    if complete and len(rows) >= 2 else None)
        methods.append({"method_id": name, "repeat_count": len(rows),
            "mean_accuracy": statistics.fmean(row["accuracy"] for row in rows),
            "failure_count": sum(row["failure_count"] for row in rows),
            "mean_paired_accuracy_difference": statistics.fmean(effects) if complete else None,
            "paired_accuracy_ci95": interval,
            "uncertainty_unit": "independent_repeat",
            "uncertainty_note": "Equal-weight repeat percentile bootstrap; requires independent initializations."
                if interval is not None else "Need at least two fully paired independent repeats for an interval."})
    return {"version": 1, "experiment": first["experiment"], "baseline": first["baseline"],
            "methods": methods, "seeds": sorted(observed_seeds),
            "interpretation": first["interpretation"]}
