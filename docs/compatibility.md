# Executed capabilities and limits

| Backend/profile | Available operations | Verification |
|---|---|---|
| Scripted / synthetic Qwen3-shaped tokens | Reasoning/answer streams, all five typed probes, synthetic embeddings, cancellation | Offline method/runtime fixtures; signals are intentionally simulated |
| Replay / captured profile | Exactly the recorded tokens, transitions and signal requests | Decision replay fixtures; missing counterfactuals raise `ReplayUnavailable`; controller timing only |
| Transformers 4.53.2 / Qwen3-0.6B pinned revision | Single-sequence decode, exact candidate scores, raw full-vocabulary scalar entropy, selected log margin, sequence scoring, isolated answer branches, finalization | CPU float32 mechanics and method smoke runs; see backend evidence |
| Transformers + explicit pinned MiniLM provider | Embedding signals for REFRAIN | Optional provider; no embedding capability is advertised when absent |

The actual Qwen3 model/tokenizer revision is
`c1899de289a04d12100db370d81485cdf75e47ca`. Qwen3 profiles validate decoder-only
architecture, native chat template and the single-token reasoning delimiter.
Other model revisions require explicit selection and do not inherit these test results.

HALT-CoT requires at least two inference-visible candidates. Contextual single-token
labels must be distinct; complete-sequence scoring is a separately declared adaptation.
ThinkBrake needs exact delimiter and top-token scores at the committed boundary.
Answer Convergence needs isolated answer generation and exact prefix resumption.
REFRAIN needs an explicit encoder; its adaptive recipe also requires a sequential
session and valid answer-span likelihood. DEER currently supports a Qwen3 greedy
trial at an exactly removable terminal `Wait` token suffix. The source's broader
string-stop behavior is not advertised as arbitrary tokenizer support.

All research controllers are independent common-protocol implementations. Audits
describe changed prompting, segmentation, sampling, normalization or finalization.
Source-pinned does not mean source-parity verified, paper-reproduced or author-reviewed.

No vLLM/HTTP backend, async server, model training, multi-trajectory controller or
logits intervention is exposed. These remain roadmap work. CPU import requires
no Torch; optional extras load only when explicitly selected. Hard monetary ceilings
are rejected because no billing provider is implemented. Local deadlines and cancellation
are cooperative at forward-call boundaries. No server-side cancellation claim exists.

Python 3.11.0 and 3.12.4 pass the offline suite locally. CI is configured for Python 3.11, 3.12 and 3.13 on
Linux and Windows; unexecuted CI jobs are not represented as local passes. Only the
exact direct dependency versions recorded in `requirements-dev.lock` and
`requirements-model-tested.txt` were exercised here; broader declared ranges are
installation compatibility bounds, not a fully tested matrix.
