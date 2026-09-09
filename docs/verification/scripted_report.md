# HALT evaluation

Original toy data and simulated runs demonstrate mechanics only. No general efficiency claim follows from this report.

| Method | Evidence | N | Accuracy | Paired change (95% CI) | Mean generated tokens | Mean probe calls |
|---|---|---:|---:|---|---:|---:|
| full_reasoning | simulated | 6 | 0.500 | +0.000 [+0.000, +0.000] | 14.0 | 0.0 |
| fixed_reasoning_budget_4 | simulated | 6 | 0.500 | +0.000 [+0.000, +0.000] | 6.0 | 0.0 |
| fixed_reasoning_budget_8 | simulated | 6 | 0.500 | +0.000 [+0.000, +0.000] | 10.0 | 0.0 |
| immediate_answer | simulated | 6 | 0.500 | +0.000 [+0.000, +0.000] | 2.0 | 0.0 |
| halt_cot | simulated | 6 | 0.500 | +0.000 [+0.000, +0.000] | 7.0 | 1.0 |
| thinkbrake | simulated | 6 | 0.500 | +0.000 [+0.000, +0.000] | 10.0 | 2.0 |
| answer_convergence | simulated | 6 | 0.500 | +0.000 [+0.000, +0.000] | 18.0 | 2.0 |
| deer | simulated | 6 | 0.500 | +0.000 [+0.000, +0.000] | 14.0 | 0.0 |

Latency scope and complete work/resource records are in summary.json. Percentiles are omitted below 20 examples. Every output is evaluated against an evaluator-only reference label. Resolved configurations and run identities are in manifest.json and per_example.jsonl.
