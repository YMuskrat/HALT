# Adding a stopping method

Every built-in and contributed method uses the same `reset(context)`, `observe(event)`, and
`decide()` contract. HALT handles inference, budgets, signal execution, accounting, datasets,
and comparison reports. Your class owns its stopping rule, parameters, and per-question state.
An original idea does not need a paper citation.

## Contribute through a pull request

Fork and clone the repository, create a branch, and install the development package:

```console
python -m pip install -e ".[dev]"
halt new-method my_stopper --contribute --template moving_average
```

Run the generator from the HALT checkout root, or pass that root with `--output`. The default
`boundary` template counts reasoning boundaries. The `moving_average` template requests raw
full-vocabulary entropy in nats, keeps a rolling window, and stops when its mean is strictly
below a threshold. It is an original teaching example with demonstration settings.

The generator creates four files:

| File | What you change |
|---|---|
| `src/halt/methods/my_stopper.py` | Write the rule, parameters, and required capabilities. |
| `tests/methods/test_my_stopper.py` | Add small examples for continuing, stopping, invalid evidence, and reset. |
| `src/halt/resources/method_cards/my_stopper.json` | Fill the description, authors, limitations, and any applicable sources. |
| `configs/methods/my_stopper.json` | An immediately runnable scripted example; adjust parameters if useful. |

Normally you edit the implementation and test, then fill the short card. The card includes
an explicit local implementation target, so discovery and wheel packaging work automatically.
There is no shared registry or export file to edit. The card describes your method; the Python
class implements it. Existing files are preserved if a destination already exists.

Check and try the generated contribution:

```console
python -m pytest tests/methods/test_my_stopper.py
halt methods check my_stopper
halt methods inspect my_stopper
halt run --config configs/methods/my_stopper.json --output runs/my_stopper.json
halt benchmark --dataset questions.csv --methods full_reasoning,my_stopper
```

The generated `run` configuration is scripted, so it requires no model download. The simple
benchmark command uses real inference by default; install the `transformers` extra first or
use `--backend scripted` for a clearly labeled demonstration. See the
[dataset walkthrough](datasets.md) for input formats and the [results guide](results_format.md)
for saved results. Once your tests and comparison are ready, open a PR containing the four
generated files and describe the rule, validation, and supported combinations. A large model
experiment is not needed to demonstrate a small decision rule's correctness.

## Choose the public class that fits the rule

Import from `halt.methods`:

```python
from halt.methods import BaseMethod, ProbeMethod
```

`BaseMethod` is the general foundation. Override the three lifecycle functions for algorithms
that count tokens, inspect text, consume main-stream events, or need a different scheduling
policy. For a token-count example, read [FixedReasoningBudget](../src/halt/methods/baselines.py).
Nothing requires an algorithm to use probes or reasoning boundaries.

`ProbeMethod` is an optional helper. It implements the same lifecycle, handles request IDs,
prefix matching, terminal results, and decision storage, and offers these hooks:

| Hook | Purpose |
|---|---|
| `reset(context)` | Call `super().reset(context)` and initialize private per-question history. |
| `on_boundary(event)` | Request measurements at your declared reasoning boundary. |
| `on_event(event)` | Handle token, score, and other non-probe events. |
| `on_probe(result)` | Update the rule from one valid result. |
| `on_probes(results)` | Optionally make a joint decision from a completed batch. |
| `invalid_probe()` | Clear evidence or counters when a batch is denied, fails, or is invalid. |

The helper does not choose measurements, frequency, thresholds, or equations. For answer
agreement using the same interface, read [AnswerConvergence](../src/halt/methods/answer_convergence.py).
The old internal `_research.ProbeMethod` import remains compatible, but new code should use
the public import above.

## Request a signal and decide

Inside `on_boundary`, request a typed measurement:

```python
from halt.types import ScoreNextToken, WorkLimit

def on_boundary(self, event):
    self.request_signal(
        ScoreNextToken,
        include_entropy=True,
        include_end_margin=False,
        max_work=WorkLimit(scored_tokens=1),
    )
```

HALT supplies the current exact prefix and a unique request ID. The runtime schedules the
work, checks the remaining budgets, and records its actual cost. In `on_probe`, check the
value type, frame, units, and finite values appropriate to your rule, update your history,
and call `self.finalize("low_mean_entropy", diagnostics={...})` when ready. The generated
moving-average implementation and its test show the complete code.

For a batch, construct requests with `self.signal(...)` and pass them together:

```python
self.request(
    self.signal(ScoreNextToken, include_entropy=True, max_work=WorkLimit(scored_tokens=1)),
    self.signal(EmbedSteps, steps=(event.payload.text,), max_work=WorkLimit()),
)
```

This second example also requires `EmbedSteps` from `halt.types` and an embeddings-capable
backend. All requests bind the same observation prefix. The helper waits for the entire
batch, then calls `on_probes` in original request order. Its default implementation calls
`on_probe` for each result. Override `on_probes` when a rule combines measurements. One
batch is outstanding at a time. Any invalid, denied, or failed member invalidates the entire
batch and calls `invalid_probe` once; no partial evidence is delivered. Duplicate and unknown
terminal results are ignored, and reused request IDs or stale requested prefixes are errors.
For another failure policy or concurrent event handling, implement the lifecycle directly
with `BaseMethod`.

`self.finalize(...)` asks HALT to generate a final answer. For an answer already obtained
from a validated probe, use `self.return_answer(answer, source_result, reason, diagnostics=...)`
and declare `can_return_answer=True`. The runtime verifies the actual answer provenance.
General decisions remain `Continue`, `RequestSignals`, `Finalize`, `ReturnAnswer`, and
`Abstain`; declare `can_abstain=True` before using abstention.

Read the [signal reference](signals.md) for available requests, result meanings, capabilities,
units, and limitations. Full raw logits, hidden states, and attention maps are not currently
method signals. A new measurement needs a typed request/result, capability checks, backend
implementation, and accounting tests; it should not introduce a method-name conditional in
the generation loop.

## Declare identity and keep state isolated

Expose a `MethodSpec` with a unique lowercase snake_case identifier, API version `1`, actual
required capabilities, boundary policy, and finalization recipe. Candidate next-token scoring
also declares `signal_operations=("candidate_next_token",)` so supported candidate tokenization
can be checked before a run. Backend capability checks are not evidence that every model is
compatible; record combinations you have tested.

Use a dataclass for constructor parameters so `configuration()` records them. Put mutable run
state in private instance fields initialized by `reset`, and call the parent reset. HALT clones
the configured instance for every question; do not use globals, class-level histories, neural
clients, or reference labels. An unchanged state must produce the same `decide()` result.
Method diagnostics are optional scalar/list/dictionary data saved with the stop event. Standard
comparison measurements come from HALT's ledger rather than author-supplied performance totals.

`halt methods check` checks lightweight API mechanics and repeated-run isolation on scripted
data. It cannot prove scientific fidelity, accuracy, or real-model compatibility. Add focused
tests for the rule, including threshold equality, insufficient history, invalid values, and
fresh-question state. Use a disjoint evaluation split after selecting thresholds; preserve
failed/incomplete results and probe work in comparisons.

## Share an independent plugin

The same templates also generate an independently installable package:

```console
halt new-method my_stopper --template moving_average --output examples/my_stopper
python -m pip install -e examples/my_stopper
python -m pytest examples/my_stopper/tests
halt methods check my_stopper
```

The `halt.methods` packaging entry point makes it available to the same runner and benchmark.
The authoritative method card is packaged beside the implementation at
`src/halt_plugin_my_stopper/method_card.json`; the scaffold also writes an initial root copy
for browsing. `halt methods inspect` reads the packaged card. Registry listing discovers names
and distribution versions without importing plugin modules. Names must be unique across
built-ins, contributions, and plugins.

Original methods need authors, a description, and limitations. When adapting a paper or code,
add citations and immutable source revisions where available, retain required license notices,
and distinguish direct reuse, independent implementation, and adaptations. No author review or
paper-result reproduction is implied by a scaffold or passing conformance checks.
