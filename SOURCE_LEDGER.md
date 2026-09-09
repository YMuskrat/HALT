# HALT source ledger

Audit date: 2026-09-09. These are primary sources, not author endorsements.
The M0 ledger was written before research-method implementation. Initial status for
every integration was planned. Current executable and verification status is in
the corresponding machine-readable method card and IMPLEMENTATION_STATUS.md.

| Method | Pinned paper/release | Audited code | Reuse status |
|---|---|---|---|
| HALT-CoT | [2025 workshop paper](https://openreview.net/forum?id=CX5c7C1CZa), PDF mirror within pinned HF release | [yass4/halt-cot at 9ddbe6c81cf2ec073a8c59672c414d2960d31c6b](https://huggingface.co/yass4/halt-cot/tree/9ddbe6c81cf2ec073a8c59672c414d2960d31c6b), core.py and transformers_backend.py | MIT; author code available. Controller independently implemented with attribution. |
| ThinkBrake | [arXiv:2510.00546v1](https://arxiv.org/html/2510.00546v1), 2025-10-01, *Mitigating Overthinking in Tool Reasoning* | [32f821bfc872ce74720858dd92d8f6fd23541e99](https://github.com/holi-lab/ThinkBrake/tree/32f821bfc872ce74720858dd92d8f6fd23541e99): metadata/install files, no referenced generation implementation | Paper equation used; source-code parity unavailable. Later paper titles are different; this card deliberately pins v1. |
| Answer Convergence | [arXiv:2506.02536v1](https://arxiv.org/html/2506.02536v1), 2025-06-03, answer-consistency strategy only | [0ff83811e409bc28cf2e06b5cd51ac4eac553f80](https://github.com/launchnlp/reasoning_earlystop/tree/0ff83811e409bc28cf2e06b5cd51ac4eac553f80), src/early_stopping_via_consistency.py and collect_partial_reasoning_res.py | No root LICENSE found; no upstream code copied. Independent implementation of published rule. |
| REFRAIN | [arXiv:2510.10103v1](https://arxiv.org/html/2510.10103v1), 2025-10-11 | No official implementation verified in primary source/repository search | Paper Algorithms 1/2 and Eqs. 2–9; unresolved source details explicitly recorded. |
| DEER | [arXiv:2504.15895v1](https://arxiv.org/abs/2504.15895v1), 2025-04-22; later Qwen3 code variant pinned separately | [c9dd19fbffa27f841cfe47502d015b63811b4d1b](https://github.com/iie-ycx/DEER/tree/c9dd19fbffa27f841cfe47502d015b63811b4d1b), vllm-deer-qwen3.py | MIT; independently implemented sequential greedy adaptation, not DEER-Pro or branch-parallel. |

The HALT-CoT Hub repository is a method/code release, not pretrained weights.
Its cached local snapshot held only `.gitattributes`; public pinned code was
retrieved to the ignored `.source_audit/` directory for inspection. No author was
contacted. Original method authors and HALT implementation contributors are
separately credited in the method cards.

Backend assumption: single-sequence, decoder-only Transformers with isolated
re-prefill probes. Qwen3 profile/model revisions and dependency verification are
recorded separately in the backend audit and run provenance. Published speed or
accuracy figures are not HALT measurements.

See [method audits](docs/audits/README.md) for source-to-code mappings and the
precise changes in each common-protocol recipe.
