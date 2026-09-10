# Implementation status — 0.1.0rc1 candidate

HALT is executable and installable. The core runtime, five research controllers,
three baselines, offline and real backends, contributor plugin workflow, evaluation,
calibration and replay are implemented. **The full research/release specification
is not declared complete:** source-exact experimental parity and broader benchmark/
platform evidence remain unmet gates. No PyPI release or author outreach occurred.

The contributor/experiment update adds CSV/JSONL import and validation, one-command
dataset/model trials, immutable checkpoint resolution, a public neutral method
helper, pull-request scaffolds, configurable answer recipes, reusable result exports,
an interactive local HTML report, and Python/notebook comparison examples.
The update passed **177 offline tests**, Ruff, mypy (47 source files), wheel/sdist
validation and clean-wheel trial/plugin checks. Six backend/profile tests passed
with cached Qwen3 weights, and the real REFRAIN/induced-DEER checks passed again.
A two-method CLI model trial retained both budget-incomplete answers in its reports.
See [update verification](docs/verification/experiment_update.md) and the
[public CI runs](https://github.com/YMuskrat/HALT/actions/workflows/ci.yml).

| Milestone | Implemented and checked | Remaining gate |
|---|---|---|
| M0 | Primary-source ledger, immutable source/model pins, method audits/cards, license/credit separation; workspace was empty | ThinkBrake generation code absent; no verified official REFRAIN repository; source ambiguities remain explicit |
| M1 | Typed per-run runtime; scripted backend; baselines; event stream; probes/finalization; central work budget/ledger; CLI; failure and cleanup fixtures | None for offline mechanics |
| M2 | Pinned Qwen3-0.6B / Transformers 4.53.2 CPU decode, candidate/next-token/sequence scoring, isolated probes, RNG continuation, forced answer and accounting | Other models/cache modes/hardware are unverified; no optimized KV reuse |
| M3 | HALT-CoT, ThinkBrake and Answer Convergence distinct controllers; source-derived fixtures; each stopped and returned a valid answer in the real one-item smoke comparison | These are named common-protocol adaptations, not exact upstream experimental reproductions or calibrated quality evidence |
| M4 | REFRAIN fixed detector and explicit adaptive session; per-arm UCB/reward/persistence/order fixtures; DEER Qwen3 trial rule/rollback fixtures; real encoder/reward and induced DEER branch checks | No end-to-end published-result reproduction; REFRAIN paper ambiguities and DEER token-suffix restriction are documented |
| M5 | Entry-point and contribution-card registry; boundary/moving-average scaffolds; MCQ/numeric CSV/JSONL and optional pinned ARC adapter; resume identities; reusable CSV/JSONL and HTML reports; comparison API; calibration; opt-in replay | Larger established-dataset/model benchmark and independently initialized adaptive-session uncertainty study not run |
| M6 | Documentation, support matrix, source cards included in wheel, clean install, local Python 3.11/3.12 checks, public Linux/Windows Python 3.11–3.13 CI and release notes | No author review, PyPI-name availability claim or PyPI publishing |

Actual evidence is in [backend verification](docs/backend_verification.md),
[method fixtures](tests/unit/test_research_methods.py),
[session fixtures](tests/unit/test_refrain_session.py),
[runtime tests](tests/test_runtime.py), and the generated release verification record
at `docs/verification/release_checks.json`. The scripted report is explicitly marked
simulated; the six-method real comparison is one original arithmetic question, not
an accuracy/efficiency benchmark. Full reasoning's answer was never treated as gold.

Initial-candidate verification included 78 passing offline tests under each of CPython 3.12.4 and 3.11.0,
the separately installed plugin fixture, Ruff and mypy. Six backend/profile tests
passed (including two opted-in real-model tests). The six real-method comparison
runs all completed with the correct one-item answer; REFRAIN's real session reward
and the induced DEER branch fixture also passed. The optional ARC adapter fetched
and exported two validation items from data-only revision
`210d026faf9955653af8916fad021475a3f00453` with Datasets 3.6.0 and Hub 0.33.4.

Use the workspace environment on Windows:

```powershell
.venv\Scripts\halt.exe demo --backend scripted
.venv\Scripts\halt.exe methods inspect halt_cot
.venv\Scripts\halt.exe run --config configs/halt_cot_scripted_demo.json --output runs/demo.json
.venv\Scripts\halt.exe benchmark --config configs/benchmark_mcq.yaml --output-dir runs/mcq
.venv\Scripts\halt.exe calibrate --config configs/calibrate_halt_cot.yaml --output runs/calibration.json
.venv\Scripts\halt.exe replay --trace tests/fixtures/example_trace.jsonl --method halt_cot
.venv\Scripts\halt.exe methods check example_stopper
.venv\Scripts\python.exe scripts/check_commands.py
.venv\Scripts\python.exe -m pytest -m "not model"
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src/halt
.venv\Scripts\python.exe scripts/build_verify.py
```

Fresh environment installation: `python -m pip install -e ".[dev]"`. Real-model
commands and the separate existing CPU dependency environment are documented in
`docs/backend_verification.md`. Model files live in ignored `.model-cache/`; they
are not distributed. Build artifacts live in ignored `dist/`.

Reviewable deviations/limits:

- Research cards are functional independent **common_protocol** implementations;
  no reference recipe, author endorsement or reproduced speedup is claimed. HALT-CoT
  changes numbered-step prompting and alias scoring; ThinkBrake fixes underspecified
  boundary/processing choices; Answer Convergence changes NLTK/offline segmentation
  and prompting/sampling; REFRAIN records prior-length-mean and cue choices; DEER
  uses an explicit sequential greedy Qwen3 variant. See source-linked audit notes.
- Full-prefix recomputation is correct and costly. Input work, token counts, scoring,
  embedding calls and latency are distinct metrics. No billed-cost provider or hard
  monetary ceiling is implemented. Local cancellation/deadlines cannot preempt an
  active synchronous model forward pass. No untested cache or HTTP capability is claimed.
- DEER removes an exact terminal `Wait` token suffix for its trial. A token that also
  contains extra text/whitespace can be rejected as an unsupported branch shape.
  The real fixture deliberately induces its supported prefix, and is labeled accordingly.
- Replayed tokens describe recorded output; replay timing and work describe controller
  execution. Missing probes/transitions raise `ReplayUnavailable`. Replay does not
  infer counterfactual answers or latency. Full capture is required; metadata capture
  redacts text and cannot be replayed.
- Only Python 3.11.0/3.12.4 and the recorded dependency versions are locally verified. The
  supported CI targets are 3.11–3.13. Larger model evaluation was not attempted on
  this CPU-only host; the one-item checks do not establish general savings or quality.
- Default decoding and model profile use the pinned native Qwen3 template. The shared
  answer cue is an explicit HALT recipe. Whole-choice final outputs are accepted only
  when they exactly match an inference-visible option. Source-specific recipes and
  their parser differences remain separately recorded.

vLLM, generic HTTP, async concurrency, logits interventions, trained predictors and
multiple trajectories are later work, not stub integrations. No web dashboard,
hosted service, training pipeline or unrelated agent framework was added.
