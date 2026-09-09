# ThinkBrake integration audit

Minjae Oh, Sangjun Song, Seungkyu Lee, Sungmin Jo, Yohan Jo,
*ThinkBrake: Mitigating Overthinking in Tool Reasoning*,
[arXiv:2510.00546v1](https://arxiv.org/html/2510.00546v1), 2025-10-01.
The [repository commit](https://github.com/holi-lab/ThinkBrake/tree/32f821bfc872ce74720858dd92d8f6fd23541e99)
contains metadata/install files but lacks the advertised decoding modules.

Section 3.1 stops when log(p(top)/p(</think>)) <= threshold. HALT computes the
equivalent difference in natural log probabilities at the exact prefix *after*
the committed newline boundary. Scores used to sample that newline are not used.
The paper v1 argmax covers the vocabulary; it lists no candidate-token exclusion.
HALT requests raw model scores and exact delimiter probability. Missing or invalid
scores cause no stop; they are never replaced by zero.

`ThinkBrake.observe` maps this formula to `ScoreNextToken` and `Finalize`.
The `thinkbrake_v1_newline_raw` common-protocol recipe fixes the paper's
underspecified boundary example to newline segmentation and raw scores, with the
Qwen3 single-token delimiter. Multiple-token delimiters are unsupported. Final
answer generation uses the profile closure once. State is per run.

Fixtures include strict position identity, threshold equality, incorrect sign,
missing/nonfinite scores, and independent runs. This verifies the stated equation,
not a port of absent code or a BFCL reproduction. The repository now uses a later
title and author order; this implementation deliberately targets the older pinned
version. Author review: unreviewed.
