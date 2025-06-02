# Repeated questions and uncertainty

`compare(rows, uncertainty="question")` computes a paired percentile bootstrap
by resampling question IDs, retaining every matched seed for each selected
question. `grouped_compare` accepts the same option. The default remains `iid`,
which withholds intervals when repeated questions or adaptive sessions are found.

The point estimate is the observation-weighted mean paired accuracy difference.
Each resampled draw uses the total difference divided by its observation count;
questions with more observed seeds therefore carry more weight. Prefer balanced
seed coverage. Unmatched observations are reported but excluded from paired effects.
Failures remain in accuracy denominators.

At least two independent matched questions are required. An interval from very
few questions can be uninformative; this method does not correct dataset selection
bias or make synthetic output into model-performance evidence. The report records
the number of independent questions and the uncertainty unit.

Questions inside an adaptive session are dependent. Selecting `question` does
not enable an interval for them. Repeat independently initialized sessions and
analyze whole-session effects instead. Do not pool their questions into an iid
bootstrap. No formal risk-control guarantee is implied by either analysis.

```python
from halt.evaluation import compare, load_results

summary = compare(load_results("runs/trial/results.jsonl"),
                  uncertainty="question", bootstrap_samples=2000, seed=7)
```
