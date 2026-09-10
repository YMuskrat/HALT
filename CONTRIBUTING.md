# Contributing a method

A contribution needs a named strategy, version, configuration, requirements,
decision logic and behavior fixtures. HALT supplies generation, signals, branch
isolation, budgets, finalization and report formats. Read the
[author guide](docs/author_guide.md) and [source audit conventions](docs/audits/README.md).

1. Fork the repository, create a branch, install `python -m pip install -e ".[dev]"`,
   and run `halt new-method my_stopper --contribute`. Add `--template moving_average`
   for an example that consumes entropy signals and maintains a rolling history.
2. Edit the generated method and its test. The scaffold also supplies a registration
   card and demo configuration; there is no registry source file to edit. Declare
   capabilities, exact signal semantics and finalization recipe. Add citations and
   license details if adapting existing work; original algorithms need no paper.
3. Add fixtures with hand-computed expected decisions, including failed/denied probes
   and the relevant boundary positions. Never turn invalid signals into a confident stop.
4. Run the plugin's tests, HALT conformance checks and a documented backend smoke check.
   Compare source equations or executable upstream fixtures before claiming fidelity.
5. Evaluate against the task's gold labels kept in the evaluator. Freeze experimental
   recipes and calibration splits before held-out evaluation. Report all probe work.

Submit the generated files in a pull request. Most original methods only require
hand editing the implementation and its test. External plugins remain an option:
use `halt new-method my_stopper --output runs/my_stopper`, then install the generated
package with `python -m pip install -e runs/my_stopper`.

All methods expose `spec` and implement `reset(context)`, `observe(event)` and `decide()`.
Inherit `BaseMethod`, or the optional public `ProbeMethod` helper to receive typed
signals through callbacks. See the [signal reference](docs/signals.md). They must not
load models, make HTTP calls, mutate caches or call provider clients. `decide()`
is deterministic when state has not changed. Store only compact histories. Use
typed signal requests tied to the exact `PrefixRef`. The runtime delivers a scheduled
event and a terminal completion/failure/denial event. A `Continue` advances main
generation before the next decision cycle; a repeated request cannot loop indefinitely.

Each run gets a private copy of the configured method. REFRAIN uses an explicit
sequential session; never place user/request state in module globals. Session reward
uses model answer likelihood and length, never evaluation correctness labels.

List unsupported backend/profile combinations explicitly. A reference recipe must
reject substitutions; a changed recipe must say what changed. Wrap upstream code only
after checking its license and interface. Upstream transport/decoding code belongs in
a backend/provider adapter, not in the decision callback.

Core checks:

```sh
python -m pip install -e ".[dev]"
python -m pytest -m "not model"
python -m ruff check .
python -m mypy src/halt
python -m build
python -m twine check dist/*
```

The [independent example plugin](examples/external_plugin) is installed separately
in CI. No core file edits are needed to discover an external method. `methods list`
loads entry-point metadata only, so it does not import heavy plugin dependencies.

Do not claim author review, paper-result reproduction or general savings based on
component fixtures. Prepare source-linked technical questions for author review;
sending them is a separate action. Credit research authors, code contributors and
reviewers for the work they actually performed.
