"""Offline-first command line entry point."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from halt.config import ResolvedConfig, load_config, make_backend, make_task
from halt.errors import HaltError
from halt.experiments import dataset_arguments, dataset_options, model_arguments, trial_config
from halt.registry import MethodRegistry
from halt.types import Budget, to_data


def _emit(data: Any, output: str | Path | None = None) -> None:
    text = json.dumps(to_data(data), indent=2, allow_nan=False)
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _doctor(config: ResolvedConfig, *, load_model: bool = False) -> dict[str, Any]:
    from halt.api import HaltRunner

    method = MethodRegistry().create(config.method["name"], config.method["parameters"])
    result: dict[str, Any] = {"schema_version": "1.0", "resolved_configuration": config.to_dict(),
        "method_requirements": to_data(method.spec.requirements), "method_spec": to_data(method.spec),
        "validation": "static", "model_loaded": False,
        "optional_dependencies": {name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "yaml", "sentence_transformers")}}
    if config.backend["name"] == "scripted" or load_model:
        backend = make_backend(config)
        result["inspection"] = HaltRunner(backend).inspect(make_task(config.task), method, config.budget)
        result["detected_capabilities"] = sorted(backend.info.capabilities)
        result["model_loaded"] = config.backend["name"] != "scripted"
        result["validation"] = "model_and_tokenizer" if load_model else "static_scripted"
        result["unsupported_assumptions"] = []
    else:
        result["detected_capabilities"] = None
        result["unsupported_assumptions"] = ["Actual tokenizer, delimiters, contextual candidate IDs and model capabilities require --load-model."]
    return result


def _demo(scenario: str) -> Any:
    from halt.api import HaltRunner
    from halt.backends.scripted import ScriptedBackend
    from halt.methods.base import BaseMethod
    from halt.methods.baselines import FullReasoning
    from halt.types import (
        Continue,
        EventKind,
        Finalize,
        MethodSpec,
        RequestSignals,
        ScoreCandidates,
        WorkLimit,
    )

    class ProbeDemo(BaseMethod):
        spec = MethodSpec("demo_probe", variant="original_demo")

        def reset(self, context: Any) -> None:
            super().reset(context)
            self._decision: Any = Continue()

        def observe(self, event: Any) -> None:
            if event.kind == EventKind.STEP_BOUNDARY:
                self._decision = RequestSignals((ScoreCandidates(request_id="demo-score", prefix=event.prefix,
                    candidates=("A", "B", "C", "D"), max_work=WorkLimit(scored_tokens=4)),))
            elif event.kind == EventKind.PROBE_COMPLETED:
                self._decision = Finalize("demo_probe_completed")
            elif event.kind in {EventKind.PROBE_DENIED, EventKind.PROBE_FAILED}:
                self._decision = Continue()

        def decide(self) -> Any:
            return self._decision

    config = ResolvedConfig()
    backend = ScriptedBackend(answer_eos=scenario != "incomplete", fail_at_token=2 if scenario == "error" else None)
    method = ProbeDemo() if scenario == "probe" else FullReasoning()
    budget = Budget(max_reasoning_tokens=3 if scenario == "budget" else 64,
                    max_answer_tokens=3 if scenario == "incomplete" else 16,
                    max_probe_output_tokens=16, max_total_generated_tokens=96)
    return HaltRunner(backend).run(make_task(config.task), method, budget).to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="halt", description="Auditable inference-time reasoning termination")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run a deterministic offline lifecycle demonstration")
    demo.add_argument("--backend", choices=["scripted"], default="scripted")
    demo.add_argument("--scenario", choices=["probe", "natural", "budget", "incomplete", "error"], default="probe")
    methods = commands.add_parser("methods", help="discover methods without loading model dependencies")
    actions = methods.add_subparsers(dest="action", required=True)
    actions.add_parser("list")
    inspect = actions.add_parser("inspect")
    inspect.add_argument("method_id")
    check = actions.add_parser("check", help="scripted plugin conformance")
    check.add_argument("method_id")
    doctor = commands.add_parser("doctor", help="validate configuration; model loading requires an explicit flag")
    doctor.add_argument("--config")
    model_arguments(doctor)
    doctor.add_argument("--load-model", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--config", help="advanced configuration; explicit flags override its settings")
    benchmark.add_argument("--dataset", help="local CSV or JSONL questions")
    benchmark.add_argument("--output-dir", help="default: a new timestamped directory under runs/")
    dataset_arguments(benchmark)
    model_arguments(benchmark)
    benchmark.add_argument("--methods", nargs="+", help="method names or configured evaluation IDs, separated by commas or spaces")
    benchmark.add_argument("--baseline", help="reference method; automatically included")
    benchmark.add_argument("--param", action="append", default=[], metavar="METHOD.PARAM=JSON")
    benchmark.add_argument("--seed", type=int)
    for name in ("reasoning", "answer", "probe-output", "total-generated"):
        benchmark.add_argument(f"--max-{name}-tokens", type=int)
    benchmark.add_argument("--max-probe-calls", type=int)
    benchmark.add_argument("--quiet", action="store_true", help="hide per-question progress")
    benchmark.add_argument("--json", action="store_true", help="print the summary as JSON instead of a table")
    datasets = commands.add_parser("dataset", help="validate local data without loading a model")
    dataset_actions = datasets.add_subparsers(dest="action", required=True)
    dataset_check = dataset_actions.add_parser("check")
    dataset_check.add_argument("path")
    dataset_arguments(dataset_check)
    dataset_check.add_argument("--json", action="store_true")
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--config", required=True)
    calibration.add_argument("--output", required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--trace", required=True)
    replay.add_argument("--method", required=True)
    replay.add_argument("--parameters", default=None, help="JSON method parameters; defaults to recorded configuration for the same method")
    new = commands.add_parser("new-method")
    new.add_argument("method_id")
    new.add_argument("--output")
    new.add_argument("--contribute", action="store_true", help="generate a contribution inside a HALT source checkout")
    new.add_argument("--template", choices=("boundary", "moving_average"), default="boundary")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            _emit(_demo(args.scenario))
        elif args.command == "methods":
            registry = MethodRegistry()
            if args.action == "list":
                _emit(registry.list())
            elif args.action == "inspect":
                _emit(registry.inspect(args.method_id))
            else:
                from halt.sdk import check_conformance
                checked = check_conformance(args.method_id)
                _emit(checked)
                return 0 if checked["passed"] else 1
        elif args.command == "doctor":
            from halt.profiles.selection import resolve_model_config
            config = trial_config(args, benchmark=False)
            if args.load_model:
                config = resolve_model_config(config)
            _emit(_doctor(config, load_model=args.load_model))
        elif args.command == "run":
            from halt.api import HaltRunner
            config = load_config(args.config)
            from halt.profiles.selection import resolve_model_config
            config = resolve_model_config(config)
            method = MethodRegistry().create(config.method["name"], config.method["parameters"])
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            trace = output.with_suffix(".trace.jsonl") if config.runtime["capture"] != "none" else None
            result = HaltRunner(make_backend(config)).run(make_task(config.task), method, config.budget,
                seed=config.runtime["seed"], capture=config.runtime["capture"], trace_path=trace)
            data = result.to_dict()
            data["resolved_configuration"] = config.to_dict()
            _emit(data, output)
            print(f"{result.status}: {output}")
            return 0 if str(result.status) == "completed" else 1
        elif args.command == "benchmark":
            from halt.evaluation import evaluate
            from halt.evaluation.reporting import render_comparison
            from halt.profiles.selection import resolve_model_config
            config = trial_config(args)
            config = resolve_model_config(config)
            output_dir = args.output_dir or str(Path("runs") / datetime.now(UTC).strftime("experiment-%Y%m%d-%H%M%S-%f"))
            if not args.quiet:
                model = config.model.get("name", config.backend["name"])
                print(f"Model: {model}; dataset: {config.evaluation['dataset']}", file=sys.stderr)

            def progress(completed: int, total: int, record: dict[str, Any]) -> None:
                if not args.quiet:
                    print(f"[{completed}/{total}] {record['method_id']} / {record['item_id']}: {record['result']['status']}", file=sys.stderr)

            summary = evaluate(config, output_dir, progress=progress)
            if args.json:
                _emit(summary)
            else:
                print(render_comparison(summary))
                print(f"Results saved to {Path(output_dir).resolve()}")
                print("Open report.html; reuse results.csv, results.jsonl, summary.csv, and manifest.json.")
        elif args.command == "dataset":
            from halt.evaluation.datasets import validate_dataset
            report = validate_dataset(args.path, adapter=args.adapter or "auto", limit=args.limit, **dataset_options(args))
            data = report.to_dict()
            if args.json:
                _emit(data)
            elif report.valid:
                assert report.dataset is not None
                print(f"Valid dataset: {len(report.dataset.items)} selected questions ({report.dataset.adapter}).")
            else:
                print("Dataset validation failed:", file=sys.stderr)
                for error in data["errors"]:
                    location = f"Row {error['row']}" if error["row"] is not None else "File"
                    print(f"  {location}: {error['message']}", file=sys.stderr)
            return 0 if report.valid else 1
        elif args.command == "calibrate":
            from halt.evaluation.calibration import calibrate
            calibrated = calibrate(load_config(args.config), args.output)
            _emit({"output": args.output, "status": calibrated["status"], "chosen_parameters": calibrated["chosen_parameters"]})
        elif args.command == "replay":
            from halt.api import HaltRunner
            from halt.backends.replay import ReplayBackend
            from halt.tasks import MultipleChoiceTask, NumericTask
            backend = ReplayBackend(args.trace)
            recorded = backend.started["payload"]
            context = recorded["context"]
            visible = context["task_visible"]
            task = MultipleChoiceTask(visible["question"], visible["choices"]) if visible["adapter"] == "mcq_v1" else NumericTask(visible["question"])
            parameters = json.loads(args.parameters) if args.parameters is not None else (
                recorded["configuration"] if context["method_id"] == args.method else {})
            method = MethodRegistry().create(args.method, parameters)
            result = HaltRunner(backend).run(task, method, Budget(**context["budget"]), seed=context["seed"])
            _emit(result.to_dict())
        elif args.command == "new-method":
            from halt.sdk import scaffold_method
            print(scaffold_method(args.method_id, args.output, contribute=args.contribute, template=args.template))
        return 0
    except (HaltError, OSError, ImportError, ValueError, KeyError) as exc:
        print(f"halt: {exc}", file=sys.stderr)
        return 2
