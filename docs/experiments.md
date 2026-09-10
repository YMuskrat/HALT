# Dataset and model trials

From a HALT checkout, install `python -m pip install -e ".[transformers]"` for
inference. The dependency-free core also supports `--backend scripted` to exercise
the workflow with simulated outputs.

```sh
halt dataset check data/example_questions.csv
halt benchmark --dataset data/example_questions.csv --output-dir runs/trial
halt benchmark --dataset data/example_questions.csv --methods full_reasoning,halt_cot,thinkbrake --param halt_cot.threshold=0.5 --output-dir runs/threshold-05
```

Each run validates the entire dataset before resolving checkpoint metadata or
loading weights. `--limit 10` selects the first ten valid rows after validation.
CSV and JSONL use common column names automatically; see [dataset mapping](datasets.md).
The defaults compare `full_reasoning` with `halt_cot` for multiple-choice questions
or `answer_convergence` for numeric questions. A selected `--baseline` is included
automatically. The `full_reasoning` baseline still obeys the experiment's reasoning
budget; a small cap can truncate it.

Default trial budgets are 128 reasoning tokens, 64 answer tokens, 128 probe output
tokens, 320 total generated tokens, and eight probe calls per question and method.
These are small mechanics trials. Increase them for your task and inspect failure
and budget-limit rates before interpreting accuracy or savings:

```sh
halt benchmark --dataset questions.csv --max-reasoning-tokens 1024 --max-answer-tokens 128 --max-probe-output-tokens 512 --max-probe-calls 16 --seed 19
```

For trials without a config, changing component token budgets recomputes the total
unless you pass `--max-total-generated-tokens` explicitly. With a config, its total
cap remains in force until explicitly overridden. Outputs default to a timestamped
directory under `runs/`. `--quiet` hides progress and `--json` prints the summary
as JSON. A successful benchmark command means the experiment and reports finished;
individual failed or incomplete answers remain visible in the results.

## Choosing a model

```sh
halt doctor --model Qwen/Qwen3-0.6B
halt benchmark --dataset questions.csv --model Qwen/Qwen3-0.6B --device cpu
```

The tested default is Qwen3-0.6B at commit
`c1899de289a04d12100db370d81485cdf75e47ca`. Other dense decoder-only Qwen3
checkpoints and compatible fine-tunes can be selected by Hugging Face name.
Omitted revisions for other names, and explicit branches/tags, resolve to an
immutable commit saved in the experiment. Use `--revision` and
`--tokenizer-revision` when needed. Changing a model name clears inherited
checkpoint pins from an advanced configuration.

The backend checks model metadata before weights, then verifies the tokenizer,
native chat template, and thinking delimiter. Passing these checks does not mean
a new checkpoint has been benchmarked or validated for a paper's recipe. This
backend does not support arbitrary model families, Qwen3 MoE, or remote chat APIs.
A different family needs a backend/profile contribution.

`--device cuda --dtype bfloat16` selects an available compatible GPU; CPU defaults
to float32. Use `--cache-dir` to choose storage. Offline inference requires cached
files and pinned revisions: add `--local-files-only`. Static `doctor` does not
contact the hub or load weights; `doctor --load-model` performs model checks.

## Reusable experiments and parameter sweeps

Existing JSON/TOML/YAML configs remain supported. Explicit CLI flags override the
config; replacing `--dataset` resets its old importer and column mappings.
YAML requires the evaluation extra. Use distinct IDs to compare multiple settings
of one implementation:

```json
{
  "backend": {"name": "scripted"},
  "evaluation": {
    "dataset": "../data/example_questions.csv",
    "adapter": "auto",
    "methods": [
      {"name": "full_reasoning"},
      {"name": "halt_cot", "id": "entropy_05", "parameters": {"threshold": 0.5}},
      {"name": "halt_cot", "id": "entropy_08", "parameters": {"threshold": 0.8}}
    ]
  }
}
```

Save this in `configs/my_comparison.json`, then run
`halt benchmark --config configs/my_comparison.json --output-dir runs/sweep`.
For real inference select `--backend transformers --model Qwen/Qwen3-0.6B`.
`--param entropy_05.threshold=0.4` can override a selected ID. Reusing an output
directory reuses exact matching resume identities; changed settings or code trigger
new measurements instead of reusing stale observations.

## Original prompt recipes

Stopping rules select named recipes through the shared interface. The Transformers
backend accepts additional versioned recipes in `backend.parameters.answer_recipes`:

```json
{
  "backend": {
    "name": "transformers",
    "parameters": {
      "answer_recipes": {
        "my_answer_v1": {"suffix": "\nAnswer:", "close_reasoning": true}
      }
    }
  }
}
```

A method can use `finalization_recipe="my_answer_v1"` in its `MethodSpec`, or
request the recipe on a probe. Settings describe the appended suffix, reasoning
closure, boxed-answer prefix reconstruction, exact terminal suffix removal for
isolated trials, stopping at the reasoning end, and generated-token scoring.
Main finalization requires closure and forbids suffix removal. Candidate scoring
validates contextual tokenization for its chosen cue. Invalid settings fail before
weights load. Existing recipe names are preserved as aliases; use a new versioned
name when changing behavior. The full recipe catalog is saved in backend provenance.

Use [exported results](results_format.md) to build your own comparisons. Keep model,
dataset, budgets, seed policy and settings consistent across methods, and report
probe work alongside reasoning length.
