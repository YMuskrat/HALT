# Answer Convergence: answer-consistency audit

Xin Liu and Lu Wang, *Answer Convergence as a Signal for Early Stopping in
Reasoning*, [arXiv:2506.02536v1](https://arxiv.org/html/2506.02536v1), 2025-06-03.
[Official code commit](https://github.com/launchnlp/reasoning_earlystop/tree/0ff83811e409bc28cf2e06b5cd51ac4eac553f80).

Section 4.1 checks answers from consecutive sentence chunks with greedy probes
after a reasoning-end token. `early_stopping_via_consistency.py` compares exact
extracted answer strings and stops at k identical answers, default 3.
`collect_partial_reasoning_res.py` uses NLTK sentence segmentation, cumulative
prefixes, a custom R1-Qwen template, a boxed answer suffix, presence penalty 1,
and newline-driven EOS. Its implementation collects full traces offline before
performing truncation.

`AnswerConvergence.observe` implements that k-consecutive criterion on isolated
online probes. Invalid, incomplete, missing and empty answers reset history.
It returns the current validated probe answer without another generation.
Task normalization must preserve different option labels and numeric values.

`answer_consistency_sentence_common_v1` explicitly changes the NLTK/offline
segmentation to HALT's deterministic streaming sentence policy, uses the selected
model profile and task normalization, and does not apply the source's presence
penalty/newline truncation. It is a functioning answer-consistency adaptation,
not exact source parity. No claim covers think-token adjustment or Learn-to-Stop.
No root source license was found; no code was copied. State lasts one run.

Fixtures exercise insufficient history, agreement, disagreement, invalid resets,
prefix identity and current-probe return. Exact NLTK prompt/sampler parity and
paper benchmark reproduction remain open. Author review: unreviewed.
