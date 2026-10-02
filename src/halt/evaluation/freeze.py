"""Freeze and verify benchmark inputs before allocating inference compute."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from halt.config import ResolvedConfig, load_config
from halt.errors import ConfigurationError
from halt.evaluation.datasets import load_dataset
from halt.evaluation.evaluator import environment_identity
from halt.types import stable_hash


def benchmark_identity(config: ResolvedConfig) -> dict[str, Any]:
    settings = config.evaluation
    if not settings.get("dataset"):
        raise ConfigurationError("evaluation.dataset is required")
    if config.backend["name"] == "transformers":
        for key in ("revision", "tokenizer_revision"):
            if not re.fullmatch(r"[0-9a-fA-F]{40}", str(config.model.get(key, ""))):
                raise ConfigurationError(f"freeze requires immutable model {key} (40-character commit)")
    source = Path(settings["dataset"])
    try:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        raise ConfigurationError(f"cannot read benchmark dataset: {exc}") from exc
    dataset = load_dataset(source, adapter=settings.get("adapter", "mcq_jsonl"),
        split=settings.get("split", "development"), revision=settings.get("data_revision", "local"),
        limit=settings.get("limit"), **{key: settings[key] for key in
            ("question_column", "answer_column", "id_column", "choices_column", "choice_columns") if key in settings})
    payload = {"version": 1, "configuration": config.to_dict(), "dataset_file_sha256": digest,
               "selected_dataset": dataset.identity(), "environment": environment_identity()}
    return {**payload, "identity": stable_hash(payload)}


def freeze_benchmark(config: ResolvedConfig, destination: str | Path) -> dict[str, Any]:
    """Create a new lock; never overwrite an existing reviewed benchmark plan."""
    identity = benchmark_identity(config)
    with Path(destination).open("x", encoding="utf-8") as handle:
        json.dump(identity, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return identity


def verify_benchmark(config: ResolvedConfig, lock: str | Path) -> None:
    try:
        saved = json.loads(Path(lock).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot read benchmark lock: {exc}") from exc
    if saved != benchmark_identity(config):
        raise ConfigurationError("benchmark inputs differ from the frozen lock")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("lock")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.verify:
            verify_benchmark(config, args.lock)
            print("Benchmark inputs match the frozen lock.")
        else:
            freeze_benchmark(config, args.lock)
            print("Benchmark inputs frozen; no model inference was performed.")
    except (ConfigurationError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
