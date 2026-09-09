# Recorded real-model evidence

These are integration/mechanics checks, not paper reproductions or an established
dataset benchmark. All model results below use Qwen/Qwen3-0.6B model and tokenizer
commit `c1899de289a04d12100db370d81485cdf75e47ca`, Transformers 4.53.2, Torch
2.7.1+cpu, CPython 3.12.4 on Windows, CPU float32, eager attention and four model
threads. No GPU was available. Each backend run uses one sequence; unrelated host
activity was not isolated, so timings are descriptive rather than a controlled
performance study.

`tests/test_transformers.py` passed its two opted-in model tests plus four offline
profile tests. Checks include raw candidate scores, complete vocabulary scalar
entropy, answer probes, identical sampled continuation with/without a rejected
probe, one forced delimiter, final-answer generation, cleanup and multi-token
candidate rejection. No mutable KV cache is shared. The final code additionally
restricts closure detection to the generated assistant suffix, so a delimiter
mentioned by a user cannot silently suppress forced closure.

`examples/backend_smoke.py` ran a six-method comparison on **one original question**,
“What is 2 + 2? Keep reasoning brief.” The reference label B was used only for
evaluation. Sampling: temperature 0.6, top-p 0.95, top-k 20, seed 19. Shared budget:
128 reasoning, 48 answer, 128 probe-output, 304 total-generated tokens and 8 probe
calls. Full reasoning reached its declared reasoning budget. Method demonstration
settings deliberately exercise stopping: HALT-CoT threshold 2.1 bits/consecutive 1,
ThinkBrake threshold 100 nats, Answer Convergence consecutive 2/probe limit 16.
These thresholds are not calibrated recommendations.

| Method | Status/answer | Main reasoning | Answer output | Probe output | Probe calls | Repeated/input work total | Wall seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| Immediate answer | completed / B | 0 | 6 | 0 | 0 | 388 | 11.062 |
| Fixed budget 16 | completed / B | 16 | 5 | 0 | 0 | 1,264 | 34.438 |
| Full reasoning, cap 128 | completed / B | 128 | 5 | 0 | 0 | 15,320 | 322.547 |
| HALT-CoT common recipe | completed / B | 2 | 0 | 0 | 1 | 158 | 2.734 |
| ThinkBrake common recipe | completed / B | 2 | 7 | 0 | 1 | 563 | 9.891 |
| Answer Convergence common recipe | completed / B | 55 | 0 | 22 | 3 | 6,218 | 94.578 |

The full [recorded operation ledgers](verification/real_model_smoke.json) retain
input, scoring, forced-context and generated work. An earlier run produced valid
whole-choice strings (`B. 4`, `**B. 4**`) that the stricter initial parser rejected.
The final run uses a parser accepting the complete supplied option string only
when it exactly matches visible choice text; ambiguous or mismatched text still
fails. This parser change is an explicit experimental condition.

The [adaptive/branch fixture](verification/adaptive_model_smoke.json) used the same
model plus MiniLM encoder revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`:

- REFRAIN completed a budget-limited run with answer B, 11 main/answer tokens and
  one valid session update. Its answer-only geometric token likelihood was
  0.9769207; first-sample length penalty 0.0011; reward 0.9758207. It did not need
  to trigger semantic redundancy on this short question. Reflection/history
  stopping gates are separately verified with deterministic fixtures.
- MiniLM encoded three explicit texts with normalized vectors; identical texts
  produce identical vectors. Encoder provenance and full vectors are retained.
- DEER's `Wait` suffix was explicitly inserted into a backend fixture, then a real
  greedy trial returned a boxed B and generated `</think>`. Its four raw token
  log probabilities and unchanged main prefix are recorded. This is an induced
  transition component check, not evidence that DEER naturally stopped on a
  benchmark task or matched author speedups.

Reproduce with optional dependencies and cached pinned weights:

```powershell
python -m pip install -e '.[transformers,embeddings,dev]'
python examples/real_model.py --cache-dir .model-cache
$env:HALT_RUN_MODEL_TESTS = '1'
$env:HALT_MODEL_CACHE = '.model-cache'
python -m pytest tests/test_transformers.py -q
python examples/backend_smoke.py --output runs/backend_smoke_final.json
python examples/adaptive_model_smoke.py
```

On this workspace, the model checks used system `python` with `PYTHONPATH=src;.model-deps`
to reuse existing CPU Torch/Transformers and an isolated sentence-transformers 4.1.0
install. Development/core/wheel checks used `.venv`; Python 3.11 offline checks used
`.py311-venv`. Other Python versions, model
sizes, hardware and declared dependency ranges are not locally verified. No authors
reviewed these runs, and no paper-result replication or general savings claim follows.
