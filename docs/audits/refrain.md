# REFRAIN integration audit

Renliang Sun, Wei Cheng, Dawei Li, Haifeng Chen, Wei Wang, *Stop When Enough:
Adaptive Early-Stopping for Chain-of-Thought Reasoning*,
[arXiv:2510.10103v1](https://arxiv.org/html/2510.10103v1), 2025-10-11.
No official source repository was verified. Implementation is independent.

Algorithm 1 maps to `Refrain.observe`: blank-line steps; Appendix A reflection
phrases; a provisional-answer cue in a strictly earlier step; maximum cosine
similarity to strictly earlier step embeddings; stop when similarity >= tau.
No self-similarity is included. Default encoder is all-MiniLM-L6-v2. HALT includes
the four listed categories but excludes the explicitly separate new-category
ablation. Casefolding and the exact cue strings are explicit recipe choices.

Algorithm 2 maps to `RefrainSession`: each arm has its own latest-W reward deque.
Cold-start arms follow configured order. Otherwise use mean reward plus
C sqrt(2 log(min(k, W*number_of_arms))/arm_count); ties follow configured order.
The reward is exp(mean(answer-token log probability)) minus lambda*L/prior_mean_L;
the first valid sample instead uses 0.0001*L. Save/load retains arm buffers,
configuration, seed, request order and completed IDs. Only successful valid runs
update once. Concurrent access is rejected and abort releases the active slot.

The paper does not resolve whether the running mean includes the current sample;
HALT uses the prior completed-sample mean and records that choice. Its prose says
larger tau encourages earlier stopping, contradicting Eq.5; HALT follows Eq.5.
W, C, lambda and exact cue expansion are user-visible experimental values.

`refrain_fixed_common_v1` is a detector ablation. `refrain_adaptive_common_v1`
requires an explicit session plus answer-only scoring. They remain adaptations
until encoder, forced-closure span, prompting and source ambiguities have a full
reference comparison. Fixtures cover history gates, excluded self-similarity,
per-arm windows, warmup, reward, persistence, retries and concurrency.
Author review: unreviewed.
