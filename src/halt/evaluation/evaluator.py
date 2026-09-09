from __future__ import annotations

import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

from halt.api import HaltRunner
from halt.config import ResolvedConfig, make_backend
from halt.errors import ConfigurationError
from halt.evaluation.datasets import load_dataset
from halt.evaluation.reporting import write_reports
from halt.registry import MethodRegistry
from halt.types import SCHEMA_VERSION, stable_hash, to_data


def environment_identity() -> dict[str, Any]:
    try:
        version = metadata.version("halt-reasoning")
    except metadata.PackageNotFoundError:
        version = "0.1.0rc1-uninstalled"
    # A content revision works in an unpacked sdist and changes with local source edits.
    source = Path(__file__).resolve().parents[1]
    files = sorted(source.rglob("*.py"))
    code_hash = stable_hash({str(p.relative_to(source)): stable_hash(p.read_text(encoding="utf-8")) for p in files})
    return {"halt_version": version, "halt_source_sha256": code_hash, "python": platform.python_version(),
            "plugins": MethodRegistry().list()}


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def evaluate(config: ResolvedConfig, output_dir: str | Path, *, backend: Any = None) -> dict[str, Any]:
    settings = config.evaluation
    if not settings.get("dataset"):
        raise ConfigurationError("evaluation.dataset is required")
    dataset = load_dataset(settings["dataset"], adapter=settings.get("adapter", "mcq_jsonl"),
                           split=settings.get("split", "development"), revision=settings.get("data_revision", "local"),
                           limit=settings.get("limit"))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    backend = backend if backend is not None else make_backend(config)
    runner = HaltRunner(backend)
    registry = MethodRegistry()
    method_settings = settings.get("methods") or [
        {"name": "full_reasoning"}, {"name": "fixed_reasoning_budget", "parameters": {"tokens": 4}},
        {"name": "fixed_reasoning_budget", "parameters": {"tokens": 8}}, {"name": "immediate_answer"},
        config.method,
    ]
    methods = []
    names: set[str] = set()
    for entry in method_settings:
        parameters = entry.get("parameters", {})
        name = entry.get("id", entry["name"] + ("_" + str(parameters["tokens"]) if entry["name"] == "fixed_reasoning_budget" else ""))
        if name in names:
            if entry == config.method and name == "full_reasoning":
                continue
            raise ConfigurationError(f"duplicate evaluation method ID {name!r}; assign distinct id fields")
        names.add(name)
        method = registry.create(entry["name"], parameters)
        for item in dataset.items:
            runner.inspect(item.task, method, config.budget)
        methods.append((name, entry, method))
    environment = environment_identity()
    resolved = config.to_dict()
    resolved["evaluation"] = {**settings, "methods": [entry for _, entry, _ in methods]}
    manifest = {"schema_version": SCHEMA_VERSION, "resolved_configuration": resolved,
                "dataset": dataset.identity(), "backend": to_data(backend.info), "environment": environment}
    _write_json(directory / "manifest.json", manifest)
    old: dict[str, Any] = {}
    checkpoint = directory / "checkpoint.jsonl"
    if settings.get("resume", True) and checkpoint.exists():
        for number, line in enumerate(checkpoint.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line)
                old[record["identity"]] = record
            except (ValueError, KeyError) as exc:
                raise ConfigurationError(f"invalid resume checkpoint line {number}; retain or repair the interrupted final line explicitly") from exc
    records = []
    for name, entry, method in methods:
        session = None
        if entry["name"] == "refrain" and entry.get("parameters", {}).get("adaptive", False):
            from halt.session import RefrainSession
            session_parameters = dict(settings.get("session", {}))
            session_parameters.setdefault("session_id", stable_hash({"configuration": resolved,
                "dataset": dataset.identity(), "environment": environment, "method": name})[:32])
            session = RefrainSession(**session_parameters)
        for order, item in enumerate(dataset.items):
            before = to_data(session.to_dict()) if session is not None else None
            run_manifest = {"schema_version": SCHEMA_VERSION, "environment": environment,
                "backend": to_data(backend.info), "method_spec": to_data(method.spec),
                "method_configuration": entry, "budget": to_data(config.budget),
                "runtime": config.runtime, "model_configuration": config.model,
                "backend_configuration": config.backend, "dataset": dataset.identity(),
                "item_id": item.item_id, "prompt_sha256": stable_hash(item.task.render()),
                "task_adapter": item.task.visible()["adapter"], "order": order,
                "session_before": before, "protocol": settings.get("protocol", "common_protocol")}
            identity = stable_hash(run_manifest)
            saved = old.get(identity)
            if saved is not None:
                if stable_hash(saved["manifest"]) != identity:
                    raise ConfigurationError("resume manifest integrity check failed")
                if session is not None:
                    session = type(session).from_dict(saved["session_after"])
                records.append(saved)
                continue
            trace = directory / "traces" / f"{identity}.jsonl" if config.runtime["capture"] != "none" else None
            if trace is not None:
                trace.parent.mkdir(parents=True, exist_ok=True)
            result = runner.run(item.task, method=method, budget=config.budget,
                seed=config.runtime["seed"], capture=config.runtime["capture"], trace_path=trace,
                session=session, run_id=identity[:32])
            mode = backend.info.execution_mode
            record = {"schema_version": SCHEMA_VERSION, "identity": identity, "method_id": name,
                "item_id": item.item_id, "correct": item.correct(result.answer) and str(result.status) == "completed",
                "evidence_kind": "simulated" if mode == "simulated" else ("replay" if backend.info.name == "replay" or "replay" in mode or mode == "recorded_observations" else "real"),
                "manifest": run_manifest, "session_before": before,
                "session_after": to_data(session.to_dict()) if session is not None else None,
                "result": result.to_dict()}
            records.append(record)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
    return write_reports(records, directory, baseline=settings.get("baseline", "full_reasoning"),
                         bootstrap_samples=settings.get("bootstrap_samples", 2000), seed=settings.get("seed", 0))
