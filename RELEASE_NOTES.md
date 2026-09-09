# 0.1.0rc1 implementation candidate

HALT now provides a dependency-free core with a synchronous runner/stream API,
three baselines, typed signal requests, model/profile capabilities, central budgets,
separate reasoning/probe/answer phases and a reconciled operation ledger. Cancellation,
plugin exceptions, invalid probes, stale prefixes and incomplete answers retain
their actual outcomes and consumed usage.

The reference backend executes pinned Qwen3-0.6B with Transformers on CPU using
isolated prefix recomputation and independent random streams. Scripted and replay
backends support offline development. The initial research controllers implement
HALT-CoT, ThinkBrake, Answer Convergence answer consistency, REFRAIN fixed/adaptive
variants and sequential DEER Qwen3, each with a source audit and named adaptation.

The contributor SDK scaffolds ordinary entry-point plugins; an independent plugin
is installed and exercised. Evaluation includes label-isolated MCQ/numeric JSONL,
resumable run identities, paired bootstrap comparisons, complete cost reports and
separate empirical calibration. Full traces are opt-in; unknown counterfactual replay
operations are rejected.

Release artifacts are prepared locally. This is not a claim that all specification
gates are met: exact upstream experimental parity, author reviews, a larger established
dataset benchmark, additional hardware/model combinations, and executed cross-version
CI remain outstanding. See `IMPLEMENTATION_STATUS.md` for the precise gate evidence.
PyPI publishing, package-name ownership/availability checks, and author outreach
have not been performed.
