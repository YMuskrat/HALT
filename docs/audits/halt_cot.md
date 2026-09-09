# HALT-CoT integration audit

Yassir Laaouach, *HALT-CoT: Model-Agnostic Early Stopping for Chain-of-Thought
Reasoning via Answer Entropy*, 2025 Muslims in ML workshop.
[Paper](https://openreview.net/forum?id=CX5c7C1CZa).
[Pinned author release](https://huggingface.co/yass4/halt-cot/tree/9ddbe6c81cf2ec073a8c59672c414d2960d31c6b).

`EntropyHaltingController.observe` in upstream `halt_cot/core.py` increments the
streak iff entropy is strictly below theta; equality resets it. The current argmax
answer is returned, with insertion-order tie breaking. `answer_distribution`
uses raw candidate logits, logsumexp over each candidate's first-token spelling
aliases, then candidate-set softmax. Units default to bits, with nats configurable.

HALT maps this controller to `HaltCoT.observe`, via exact `ScoreCandidates`
requests after newline steps. It requires at least two distinct candidates and
rejects nonfinite observations. Invalid/failed/denied probes reset the streak.
State lasts one run; `decide()` only returns a stored decision.

The `halt_cot_qwen_common_v1` recipe changes upstream numbered-step prompting to
the Qwen3 thinking template and supplies distinct contextual single-token labels,
without alias aggregation. Candidate collisions and multi-token labels fail
preflight. Optional complete-sequence scoring is a separately named adaptation.
It does not inherit upstream numerical candidate shortcuts or generate candidates
from held-out labels. Finalization returns the scored label directly.

Upstream is MIT. Its algorithm is independently implemented with attribution;
no model weights are in that source repository. The release's citation year 2026
differs from the workshop record's 2025; both identities remain recorded.

Fixtures: uniform/concentrated scores, strict equality, streak reset, ties,
invalid probes, and current-answer selection. Backend collision tests are shared.
Open review items: reproduce upstream alias-scoring recipes on their original
models and compare a common dataset. Author review: unreviewed.
