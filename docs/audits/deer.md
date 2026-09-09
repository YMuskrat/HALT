# DEER Qwen3 integration audit

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Zheng Lin,
Li Cao, Weiping Wang, *Dynamic Early Exit in Reasoning Models*,
[arXiv:2504.15895v1](https://arxiv.org/abs/2504.15895v1), 2025-04-22.
The actual variant is the later author
[`vllm-deer-qwen3.py` at c9dd19fbffa27f841cfe47502d015b63811b4d1b](https://github.com/iie-ycx/DEER/blob/c9dd19fbffa27f841cfe47502d015b63811b4d1b/vllm-deer-qwen3.py), MIT.

The default transition string is case-sensitive `Wait`, without a word-boundary
restriction. Its stop string is omitted from the branch prefix, an answer cue is
appended while thinking remains open, and a trial generates until `</think>` or
20 tokens. On rejection `Wait` is restored and main reasoning resumes. HALT
commits the observed transition on the main branch and removes it only from the
isolated trial context, preserving the same visible continuation.

`calculate_average_max_prob_from_logprobs` actually aggregates indices 1 through
the end (the comment about second-to-last is inaccurate), including the terminal
delimiter. `avg2` is geometric; `avg1` is arithmetic. Qwen3 requires the final
trial token to be `</think>`. Confidence must be strictly greater than threshold.
Acceptance discards the trial and generates a separate answer after forced
closure. This is distinct from answer stability or reusing a trial answer.

`DEER.observe` maps the transition to `ProbeAnswer`, computes the pinned span rule,
and chooses `Finalize`. `deer_qwen3_greedy_v1` explicitly uses greedy trials: the
source's trial temperature line is commented out and its unpinned vLLM default
is not treated as greedy. HALT also uses central budgets, a single sequential
trajectory and task validation. This is a common-protocol adaptation, not a
claim to reproduce vLLM timing or DEER-Pro/branch-parallel acceleration.

Fixtures cover strict threshold equality, geometric/arithmetic differences,
missing delimiter, invalid scores, prefix restoration, denied trials and the
separate answer-production decision. Author review: unreviewed.
