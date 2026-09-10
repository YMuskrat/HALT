# Signals available to methods

Signals are versioned typed data in `halt.types`. They describe a specific observed prefix or
an explicitly isolated branch. The runtime supplies visible inference inputs and execution
budgets; dataset reference answers remain in the evaluator.

## Events that require no probe request

| Event | Payload and use |
|---|---|
| `TokenCommitted` | `TokenOutput`: token ID, decoded text, EOS/end-of-reasoning indicators. Count or inspect generated reasoning. |
| `StepBoundary` | `StepBoundary`: text, index, and policy. Boundary policies include blank lines, newlines, sentences, fixed token chunks, or every token. |
| `NextTokenScores` | Reserved event kind; the current runtime does not emit it. Request `ScoreNextToken` for numerical features. A `ScoreFrame` identifies provenance and is not a logit array. |
| Lifecycle events | Run start, phase changes, and run finish identify context and phase. |

`BaseMethod.observe` receives events directly. `ProbeMethod` routes boundaries to `on_boundary`,
other non-probe reasoning/lifecycle events to `on_event`, and probe events through its matching
logic. It clears pending work at answering/finished phases. Methods may compute moving averages,
trends, patience counters, text features, or combinations from their own event history.

## Requested measurements

The five request/result types below are currently implemented by the scripted backend and the
Qwen3 Transformers backend, with the listed requirements. Scripted values are synthetic; replay
can only return matching recorded measurements. Real embedding requests additionally require
an embedding provider.

| Request | Result | Meaning and units | Required capability to declare |
|---|---|---|---|
| `ScoreCandidates` | `CandidateScores` | Candidate next-token log probabilities or teacher-forced sequence log scores. Natural-log units (nats); distribution conditional on the supplied candidate set after normalization. | `selected_token_scores` for next-token mode; `sequence_scoring` for sequence mode. |
| `ScoreNextToken` | `NextTokenFeatures` | Raw next-token features at the exact reasoning prefix: selected token log probabilities, top token log probability, reasoning-end log probability, optional full-vocabulary entropy in nats. | `full_vocab_scores` for full-vocabulary entropy; the end-margin feature also needs the profile's reasoning-end token. |
| `ProbeAnswer` | `ParsedAnswer` | Isolated provisional answer, parsed validity/completeness, optional answer token log probabilities in nats. Generation is bounded by `max_work.generated_tokens`. | `prefix_branch`, `exact_prefix_resume`, and a compatible answer-transition recipe. |
| `EmbedSteps` | `StepEmbeddings` | Step vectors plus encoder/revision/normalization metadata. Cosine similarity is dimensionless; compare vectors from the same encoder and policy. | `embeddings`. |
| `ScoreAnswerSequence` | `AnswerSequenceScore` | Teacher-forced answer and suffix token log probabilities in nats, with explicit normalization setting. | `sequence_scoring`. |

All methods also declare applicable stream, boundary, and answer-transition requirements.
Branch computations need prefix-branch/resume support. `MethodSpec.requirements` can express
alternatives; `HaltRunner.inspect` and `halt doctor` report compatibility before generation.
Some checks depend on the loaded tokenizer or visible task. In particular,
`signal_operations=("candidate_next_token",)` enables candidate tokenization preflight.

`ScoreCandidates` is measured at its answer recipe's branch prefix. `ProbeResult.prefix`
identifies the source reasoning prefix, while `CandidateScores.frame.prefix` may identify the
forced answer branch. Inspect both. `ScoreNextToken` uses the unchanged reasoning prefix.
Scores identify raw versus processed values and vocabulary coverage through `ScoreFrame`.
Do not treat the entropy of normalized candidate scores as full-vocabulary entropy.

The top continuation field currently includes the entire vocabulary, including the
reasoning-end token. Subtracting the reasoning-end log probability gives a margin in nats;
it does not exclude special tokens. Probe sampling settings are separate from the main
continuation and use separately recorded RNG seeds. Probe answer likelihood fields depend on
the selected recipe and answer span; consult [backend details](backend_transformers.md) before
claiming equivalence to a paper's confidence score.

## Request plumbing and limits

In a `ProbeMethod` hook, request a measurement without constructing transport identifiers:

```python
self.request_signal(
    ScoreNextToken,
    include_entropy=True,
    include_end_margin=False,
    max_work=WorkLimit(scored_tokens=1),
)
```

Import `ScoreNextToken` and `WorkLimit` from `halt.types`. `signal(RequestType, ...)` builds one
request without queuing it; `request(request_a, request_b)` queues a batch at the current exact
prefix. `request_signal(RequestType, ...)` combines the single-request steps. An optional
`recipe=` chooses the branch formatting; by default it uses the method's finalization recipe.
Next-token features do not perform an answer transition.

`WorkLimit` bounds generated, scored, and processed input tokens. Scored tokens count scored
sequence positions/candidates under the operation's ledger convention, not arithmetic over
every vocabulary entry. One next-token entropy computation therefore scores one position; its
forward and input work are recorded separately. A zero generated-token bound allows scoring
but cannot produce a provisional answer. Limits do not grant extra work: runtime budgets,
reserved final-answer tokens, per-prefix caps, and capability checks still apply. Embedding
calls are recorded separately from language-model token work.

Requests carry IDs and `PrefixRef` values that identify the run, trajectory, position, exact
prefix identity, and model revision. `ProbeMethod` matches request IDs, exact source/event
prefixes, result types, and top-level validity before delivering results. A complete valid
batch reaches `on_probes` in request order; by default it calls `on_probe` for each member.
Any denied, failed, or mismatched member invalidates the batch and calls `invalid_probe` once.
Unknown and duplicate terminal events are ignored. This helper intentionally supports one
outstanding batch; use `BaseMethod` for another scheduling or failure policy.

The method still validates semantics needed by its algorithm: finite numbers, the expected
score frame and processing, parsed completeness, vector dimensions, and a sufficient history
window. Missing measurements never stand for low entropy, answer agreement, or successful
similarity. The generated moving-average template clears its history on invalid evidence and
uses a full window before applying a strict threshold.

## Extending signals

There is no method API for full raw logit arrays, hidden states, attention maps, or arbitrary
model callbacks. Adding a new measurement requires a typed request and result, documented
units and provenance, capability validation, backend implementation, budget accounting, and
focused tests. Model execution stays in the backend; the stopping method receives values.
Existing methods continue to share the same `reset / observe / decide` interface.
