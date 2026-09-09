"""Scalar signal primitives with explicit units and probability scope."""
from __future__ import annotations

import math
from collections.abc import Sequence


def softmax(log_scores: Sequence[float]) -> tuple[float, ...]:
    if not log_scores or any(not math.isfinite(x) for x in log_scores):
        raise ValueError("log scores must be nonempty and finite")
    maximum = max(log_scores)
    values = [math.exp(x - maximum) for x in log_scores]
    denominator = math.fsum(values)
    return tuple(x / denominator for x in values)


def entropy(probabilities: Sequence[float], unit: str = "bits") -> float:
    if unit not in {"bits", "nats"}:
        raise ValueError("entropy unit must be bits or nats")
    if not probabilities or any(not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError("probabilities must be finite and nonnegative")
    if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one in their declared scope")
    value = -math.fsum(p * math.log(p) for p in probabilities if p)
    return value / math.log(2) if unit == "bits" else value


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right) or any(not math.isfinite(x) for x in (*left, *right)):
        raise ValueError("vectors must have equal positive dimensions and finite entries")
    norm = math.sqrt(math.fsum(x*x for x in left) * math.fsum(x*x for x in right))
    if not norm:
        raise ValueError("zero vectors have undefined cosine similarity")
    return math.fsum(x*y for x, y in zip(left, right, strict=True)) / norm
