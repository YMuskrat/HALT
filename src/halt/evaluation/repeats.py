"""Independent evaluation repeats with separate adaptive state and checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from halt.config import ResolvedConfig, make_backend
from halt.errors import ConfigurationError
from halt.evaluation.evaluator import environment_identity, evaluate
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
