# Results and custom comparisons

Every benchmark saves its measurements locally. The terminal table, HTML report,
and Python comparison API use the same rows. No analysis service or additional
Python dependency is required.

| File | Contents |
|---|---|
| `results.csv` | One row per question, method, and seed, including failed runs. Opens in CSV tools, pandas, R, and spreadsheets. |
| `results.jsonl` | The same fields as the CSV, plus `details` containing each original evaluator record, manifest, operation ledger, and session state. |
| `summary.csv` | One row per method with overall and paired measurements. |
| `summary.json` | Summary plus uncertainty notes, stop distributions, component timing, and experiment identity. |
| `report.html` | Standalone summary, accuracy/runtime plot, work breakdown, and searchable, sortable individual results. Open directly in a browser. |
| `report.md` | Compact comparison suitable for reviewing alongside a contribution. |
| `manifest.json` | Resolved experiment settings, dataset identity, model/tokenizer revisions, backend information, code identity, and environment. |
| `per_example.jsonl` | Original evaluator records, retained for compatibility. |
| `checkpoint.jsonl` | Append-only completed observations used for exact-identity resume. It can also be loaded for analysis after an interrupted run. |

The HTML itself makes no network requests. Its links refer to neighboring exports;
share the output directory when recipients need both the report and underlying
numbers. Exports include question text and evaluator reference answers. Those
labels remain separate from the inference task supplied to the model.

## Load results in Python

```python
from halt.evaluation import load_results, compare, render_comparison

rows = load_results("runs/my_trial")
summary = compare(rows, baseline="full_reasoning")
print(render_comparison(summary))

# Optional: use your preferred analysis library.
# import pandas as pd
# frame = pd.DataFrame(rows)
# frame.to_parquet("my_measurements.parquet", index=False)
```

`load_results` accepts a directory, `results.csv`, `results.jsonl`, or the legacy
`per_example.jsonl`/`checkpoint.jsonl`. It returns a list of ordinary Python
dictionaries with consistent types. JSON object columns in CSV are decoded;
booleans and numbers are parsed; unknown measurements stay `None`. To inspect
the complete operation ledger, read `details.result.usage.operations` directly
from `results.jsonl`; the convenient rows returned by `load_results` omit that
large nested field.

Run the included example without installing pandas or a notebook server:

```sh
python examples/analyze_results.py runs/my_trial --output runs/my_trial/custom_comparison.json
```

It prints the standard comparison and lists correct-to-wrong changes relative to
the selected baseline. Edit the example to ask your own questions. Question IDs
let you join external metadata such as difficulty or subject to the results.
No model execution happens while loading or comparing saved observations.

For an editable notebook, open [analyze_results.ipynb](../examples/analyze_results.ipynb).
It loads the same public API, checks paired coverage and failures, and includes optional
pandas tables/CSV export and matplotlib charts. Set its results path and baseline first.
The notebook ships with empty outputs and never executes a model.

## Stable individual-result schema

The exported schema has its own `results_schema_version`, currently `"1.0"`.
Existing fields keep their meaning within this major version; future releases
can add fields. Breaking semantic or type changes require a new major version.
Readers should ignore additional fields, and the HALT loader rejects unsupported
versions. This is distinct from the nested runtime record's `schema_version`.

| Fields | Meaning |
|---|---|
| `comparison_id` | Content identity of dataset/order, backend/model, generation settings, budgets, code/environment, and protocol. It excludes method configuration and trial seed so methods and repeated seeds can be paired. Capture verbosity is also excluded. |
| `identity` | Original exact per-run identity, including method configuration, question, seed, and adaptive starting state. |
| `question_id`, `method_id`, `seed`, `order` | Join keys and zero-based position in the ordered dataset. Assign distinct method IDs when comparing parameter variants. |
| `method_configuration_id` | Hash of the method configuration and declared specification; prevents silently pooling different settings under one name. |
| `dataset_sha256`, `dataset_split` | Dataset content identity and declared split. Full adapter/mapping/order information is in the manifest. |
| `model_id`, `model_revision`, `tokenizer_revision`, `backend`, `halt_source_sha256` | Model, tokenizer, backend, and implementation provenance. |
| `question`, `choices`, `reference`, `answer` | Evaluator-only question context, choices, reference, and normalized predicted answer. Older records may have empty question/reference fields. Missing predictions are `null` in JSONL and empty in CSV. |
| `correct`, `status`, `error` | Boolean correctness, actual terminal status, and error text when present. An incomplete run is not counted as correct. |
| `stop_reason`, `stop_position`, `diagnostics` | Why and at which committed reasoning-token position the method stopped, plus its optional custom measurements. |
| `adaptive_session`, `session_id` | Whether the method carries state between questions, and the session identifier. Full state is in JSONL `details`. |
| `evidence_kind` | `real`, `simulated`, or `replay`; never pool them as equivalent model evidence. |
| `timing_scope`, `usage_measurement`, `trace_reference` | Timing interpretation, backend accounting qualification, and optional trace location. |
| `method_spec`, `hardware` | Declared method contract and recorded backend hardware metadata. |

CSV uses UTF-8, a header row, decimal numbers, lowercase `true`/`false`, and empty
cells for nulls. The `choices`, `component_seconds`, `method_spec`, `hardware`, and
`diagnostics` columns contain JSON objects. Preserve question IDs as text when
importing into spreadsheets (for example, ID `001` should retain its zeros).

## What the measurements count

| Field | Unit and scope |
|---|---|
| `reasoning_tokens` | Generated tokens in the main reasoning phase. |
| `answer_tokens` | Generated tokens in the main final-answer phase. |
| `probe_output_tokens` | Generated tokens from all probe branches, including discarded branches and a probe answer later accepted directly. |
| `total_generated_tokens` | Exactly `reasoning_tokens + answer_tokens + probe_output_tokens`. Each token is counted once. |
| `input_tokens` | Token positions processed across recorded model invocations, including probes and repeated prefix processing. This is not the number of unique prompt tokens. |
| `recomputed_prefix_tokens` | Prefix positions recomputed by the backend. These overlap `input_tokens`; do not add the columns together. |
| `forced_context_tokens` | Inserted context such as a reasoning closure/answer cue, rather than sampled output. |
| `scored_tokens` | Token positions evaluated for requested scoring operations. Scoring and input counts describe different aspects of work, not extra generated tokens. |
| `probe_calls`, `forward_calls`, `embedding_calls`, `cancellation_requests` | Recorded counts of the corresponding operations. |
| `elapsed_seconds` | Per-run wall time including reasoning, probing, controller work, and answer finalization. Excludes model loading, dataset import, and report writing. Uses the runtime's `end_to_end`/`total` measurement, with a component-sum fallback for older records. Missing timing is null, never fabricated as zero. |
| `component_seconds` | Original component durations. An `end_to_end`/`total` entry overlaps its components; do not sum all entries. |

The full operation ledger records scope and backend limitations. Fewer reasoning
tokens do not guarantee less total work or lower latency. Scripted tokens are
synthetic fragments, not real tokenizer counts. Scripted timing is controller and
fixture execution; replay timing measures controller work only. The report labels
both cases explicitly. Timing is descriptive for the recorded machine and run
conditions; model loading is a separate setup cost.

## Comparisons, failures, and uncertainty

`compare` keeps failures, abstentions, and incomplete answers in the accuracy
denominator. `failure_count` in the API is the total number of rows with a status
other than `completed`; the display calls it **Not completed**. `status_counts`
retains the individual outcomes.

Overall accuracy and mean work use every available row for that method. Paired
accuracy changes and generated-token reductions use only matching
`(question_id, seed)` pairs. `paired_sample_count`, `unpaired_sample_count`, and
`baseline_unmatched_sample_count` make missing coverage visible. The generated
reduction is `1 - sum(method generated tokens) / sum(baseline generated tokens)`
over matched pairs, not an average of per-question percentages. The result is
null when the baseline denominator is zero.

Accuracy differences and confidence limits in files are fractions; the display
converts them to percentage points. The API uses a paired item bootstrap only
for a single observation per question without adaptive dependence. It suppresses
that interval when either side uses an ordered adaptive session or questions
repeat across seeds, and records the reason in `uncertainty_note`. Use a study
design and analysis accounting for that dependence when estimating uncertainty.
Small samples may produce uninformative intervals. Latency percentiles are only
reported with at least 20 measured durations, and missing times are excluded with
`timing_sample_count` recorded.

The API rejects duplicate question/method/seed rows, conflicting configurations
under a method ID, and mixed comparison identities. This prevents silently
combining different models, datasets, budgets, or implementation revisions.
The comparison ID checks recorded settings; it cannot ensure isolated hardware,
independent questions, or faithful paper reproduction.

For multiple experiments, keep them separate:

```python
from halt.evaluation import load_results, grouped_compare

rows = load_results("runs/model_a") + load_results("runs/model_b")
comparisons = grouped_compare(rows)  # Groups by comparison_id by default.
for result in comparisons:
    print(result["group"], result["comparison"]["methods"])
```

An explicit grouping such as `by=("model_id", "dataset_sha256")` is allowed,
but every group must still satisfy compatibility checks and contain its baseline.
For custom filtering, retain the matching baseline rows. Inspect unmatched counts
before interpreting a filtered comparison.
