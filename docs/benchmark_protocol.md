# Evaluation and calibration protocol

`halt benchmark --config configs/benchmark_mcq.yaml --output-dir runs/mcq` compares natural full reasoning, two fixed reasoning budgets, immediate answers and named research configurations. The supplied original six-item dataset and deterministic backend are a mechanics check. Predetermined answers deliberately include mistakes: full reasoning's answer is never used as ground truth. `numeric_jsonl` is evaluated separately and never constructs candidate sets from reference answers.

JSONL rows contain `id`, `question`, `answer`, and, for `mcq_jsonl`, `choices`. IDs must be unique and every reference must satisfy the task normalizer. Only the constructed `MultipleChoiceTask` or `NumericTask` enters the runtime. Reference labels stay in `EvaluationItem` and are used after inference. Unknown fields are rejected to avoid silently rendering private labels or data metadata.

Use one fixed model and tokenizer revision, prompt adapter, sampling configuration, answer parser, final-answer allowance and total-work policy across a common-protocol comparison. Source-specific recipe runs must be configured and reported as `source_recipe`; their differences are not hidden. Dataset bytes, subset order, split and revision are included in each identity. HALT code content, plugin versions, fully resolved method and backend settings, prompt hashes, budgets, seeds and capture levels also participate.

Checkpoints append after each completed run. Resume uses the entire hashed manifest, so changing a threshold, prompt, source file, model or dataset cannot reuse a stale result. Ordered adaptive sessions also include controller state before each request and restore the recorded state after it. A deterministic evaluation session ID enables safe restart. A malformed interrupted checkpoint line fails explicitly rather than silently discarding unrelated results.

Reports include per-example JSONL, summary JSON/CSV and Markdown. They distinguish simulated, replay and real evidence; replay time measures controller execution only. Summary JSON includes correctness, parsing/incomplete/abstention rates, paired accuracy changes, transition counts, every generated-token category, probes, repeated prefix input, scoring and embeddings, component durations, stopping positions/reasons, hardware and recipe metadata. p50/p95 latency are omitted below 20 examples. Peak memory is reported as unmeasured unless an actual measurement is available.

Independent examples use a seeded paired percentile bootstrap over correctness differences, with explicitly limited small-sample interpretation. Adaptive session rows retain request order and omit iid item intervals. Repeated independently initialized sessions or a justified dependence-aware analysis are necessary for their uncertainty.

`halt calibrate --config configs/calibrate_halt_cot.yaml --output runs/calibration.json` performs empirical parameter search on an explicitly named calibration split, after `recipe_frozen=true` confirms recipe selection on development data. The artifact records every tested setting, measured cost, observed accuracy tolerance, uncertainty, selected parameters, versions and dataset hashes. It supplies no risk guarantee. Freeze selected parameters before final evaluation on disjoint held-out data. Source-defined online REFRAIN adaptation is a separate protocol.

Default capture persists operational results and hashes. `metadata` records scalar operational events, while `full` explicitly records prompt-bearing observations needed for replay. Benchmark outputs belong under ignored `runs/`, not in the public fixture dataset. Never add credentials or private benchmark inputs to committed examples.

The optional ARC loader (`halt.evaluation.huggingface.load_arc`) targets the
[official ARC dataset schema](https://huggingface.co/datasets/allenai/ai2_arc/blob/210d026faf9955653af8916fad021475a3f00453/README.md)
and requires an immutable dataset commit. Its `choices` define the inference candidates;
`answerKey` is retained only in `EvaluationItem`. The implementation follows the
[versioned Datasets loading API](https://huggingface.co/docs/datasets/v3.6.0/loading).
The audited data-only Parquet revision is `210d026faf9955653af8916fad021475a3f00453`.
An earlier card revision required a legacy remote dataset script and was deliberately
replaced with this source-pinned data-only loader; remote code execution is unnecessary.

```sh
python -m pip install -e ".[evaluation]"
python examples/prepare_arc.py --split validation --limit 50 --output runs/arc_validation.jsonl
```

The exporter writes a dataset identity sidecar. Point a benchmark configuration to
the JSONL file and retain the sidecar's full source revision in `data_revision`.
Use validation/development data before any calibration and a disjoint test partition
after freezing the recipe. A larger ARC benchmark on a suitable model is still
required before making general quality or efficiency claims.
