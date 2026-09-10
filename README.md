# HALT

HALT is a Python library for controlling when a language model ends inference-time
reasoning and produces its answer. Methods request signals and make decisions;
the shared runtime owns generation, isolated probes, budgets, finalization and
operation-level measurements.

This is an implementation candidate. Research integrations have explicit source
pins and common-protocol adaptations; none has reproduced a paper's benchmark
results or received author review. See [implementation status](IMPLEMENTATION_STATUS.md),
[source ledger](SOURCE_LEDGER.md), and [method audits](docs/audits/README.md).

Python 3.11–3.13 is the CI target. Core imports and the scripted demo need no
Torch, GPU, network access or API key.

```sh
git clone https://github.com/YMuskrat/HALT.git
cd HALT
python -m pip install -e ".[dev]"
halt demo --backend scripted
halt methods list
halt methods inspect halt_cot
halt doctor --config configs/halt_cot_scripted_demo.json
halt run --config configs/halt_cot_scripted_demo.json --output runs/demo.json
```

```python
from halt import Budget, HaltRunner, MultipleChoiceTask
from halt.backends import ScriptedBackend
from halt.methods import HaltCoT

task = MultipleChoiceTask(
    question="A shop has 12 apples and sells 5. How many remain?",
    choices={"A": "5", "B": "6", "C": "7", "D": "8"},
)
result = HaltRunner(ScriptedBackend()).run(
    task, method=HaltCoT(threshold=0.6, consecutive=2), budget=Budget(),
)
print(result.answer, result.stop.reason, result.usage.total_generated_tokens)
```

`HaltRunner.stream(...)` yields typed events. Close an abandoned stream or use
`CancellationToken`. `run(...)` returns a `RunResult` with a distinct terminal
status, method stop decision, partial answer, usage ledger and provenance.
Callbacks run synchronously. Methods are copied and reset for each request.

The initial backend uses one decoder-only Qwen3 sequence and isolated full-prefix
recomputation. Model and tokenizer revisions are pinned. Install optional dependencies:

```sh
python -m pip install -e ".[transformers]"
python examples/real_model.py --help
halt doctor --config configs/halt_cot_qwen3_demo.yaml
```

Static `doctor` does not load model weights. Explicit model checks and measured
CPU integration evidence are documented in [the backend audit](docs/backend_transformers.md).
Small-model mechanics checks do not establish research quality or efficiency.

Try your own multiple-choice or numeric questions in CSV or JSONL:

```sh
halt dataset check data/example_questions.csv
halt benchmark --dataset data/example_questions.csv --output-dir runs/my-trial
```

The trial uses pinned Qwen3-0.6B and compares full reasoning with a compatible
stopping method. Install the `transformers` extra above for inference, or add
`--backend scripted` for an offline demonstration. Choose methods with
`--methods full_reasoning,halt_cot,thinkbrake` and change a setting with
`--param halt_cot.threshold=0.5`. Use `--model` to select another supported dense
Qwen3 checkpoint; HALT records its resolved model and tokenizer commits.

Each trial prints a small comparison table and saves an interactive `report.html`,
per-question `results.csv` / `results.jsonl`, `summary.csv`, and `manifest.json`.
Open the HTML locally or use the exported numbers in your own plots and tables:

```python
from halt.evaluation import load_results, compare, render_comparison

rows = load_results("runs/my-trial")
print(render_comparison(compare(rows, baseline="full_reasoning")))
# Optional: pandas.DataFrame(rows).to_parquet("my-results.parquet")
```

See [dataset formats and column mapping](docs/datasets.md),
[model trials and experiment settings](docs/experiments.md), and
[results and custom comparisons](docs/results_format.md).

The available baselines are `full_reasoning`, `fixed_reasoning_budget` and
`immediate_answer`. The research controllers are HALT-CoT candidate entropy,
ThinkBrake log margin, Answer Convergence answer consistency, REFRAIN fixed or
session-adaptive semantic redundancy, and sequential DEER Qwen3 trial likelihood.
Each has a distinct recipe and limitations in its method card. vLLM and compatible
HTTP endpoints remain roadmap work and are not advertised as supported backends.

```sh
halt benchmark --config configs/benchmark_mcq.yaml --output-dir runs/mcq
halt calibrate --config configs/calibrate_halt_cot.yaml --output runs/calibration.json
halt replay --trace tests/fixtures/example_trace.jsonl --method halt_cot
halt new-method my_stopper --output runs/my_stopper
python -m pip install -e runs/my_stopper
```

To contribute a method directly through a pull request, run
`halt new-method my_stopper --contribute --template moving_average` in a checkout.
HALT generates the method, test, registration card, and demo configuration.
Usually you edit just the method and its test; edit attribution when adapting
existing work. Every method uses the same `HaltMethod` contract. Inherit
`BaseMethod` for general event handling or the optional `ProbeMethod` for signal
requests and boundary callbacks. See the [author guide](docs/author_guide.md) and
[available signals](docs/signals.md).

The original toy dataset proves mechanics only. Evaluation keeps reference labels
outside inference objects, reports all probe work, and separates real inference,
scripted execution and replay controller timing. Replay only supports recorded
prefixes and operations; it raises `ReplayUnavailable` for missing counterfactuals.
Use `capture="full", trace_path="runs/trace.jsonl"` to record a replayable trace;
full capture includes task inputs and model output and is opt-in. Metadata capture
redacts text, and the default writes no trace.

```sh
python -m pytest -m "not model"
python -m ruff check .
python -m mypy src/halt
python -m build
python -m twine check dist/*.whl dist/*.tar.gz
```

See [contributing](CONTRIBUTING.md), [architecture](docs/architecture.md),
[benchmark protocol](docs/benchmark_protocol.md) and [release notes](RELEASE_NOTES.md).
Working repository/import/CLI names are HALT/`halt`/`halt`; the working distribution
name is `halt-reasoning`. PyPI package-name availability and publication require
a separate release action. Original HALT code is MIT; research authors retain
credit for their methods. No author outreach or PyPI release has been performed.
