"""Sequential DEER Qwen3 geometric-confidence adaptation, pinned in its audit."""
from __future__ import annotations

import math
from dataclasses import dataclass

from halt.errors import ConfigurationError
from halt.methods._research import ProbeMethod
from halt.types import (
    BoundaryPolicy,
    Finalize,
    MethodSpec,
    Observation,
    ParsedAnswer,
    ProbeAnswer,
    ProbeResult,
    Requirements,
    RunContext,
    WorkLimit,
)


def trial_likelihood(log_probs: tuple[float, ...], aggregation: str = "geometric") -> float:
    """Pinned Qwen3 code's [1:] span, including generated reasoning-end token."""
    if len(log_probs) < 2 or any(not math.isfinite(p) or p > 0 for p in log_probs):
        raise ValueError("DEER trial needs at least two finite nonpositive token log probabilities")
    span = log_probs[1:]
    if aggregation == "geometric":
        return math.exp(math.fsum(span) / len(span))
    if aggregation == "arithmetic":
        return math.fsum(math.exp(p) for p in span) / len(span)
    raise ValueError("aggregation must be geometric or arithmetic")


@dataclass
class DEER(ProbeMethod):
    threshold: float = 0.95
    max_probe_tokens: int = 20

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ConfigurationError("DEER threshold must be finite and in [0, 1]")
        if type(self.max_probe_tokens) is not int or self.max_probe_tokens < 2:
            raise ConfigurationError("DEER max_probe_tokens must be at least two")

    @property
    def spec(self) -> MethodSpec:
        return MethodSpec("deer", variant="deer_qwen3_greedy_v1",
            requirements=Requirements(frozenset({"visible_reasoning", "token_stream",
                "prefix_branch", "exact_prefix_resume", "sequence_scoring", "answer_transition"}),
                profiles=("qwen3_thinking", "scripted")), boundary=BoundaryPolicy("token"))

    def reset(self, context: RunContext) -> None:
        super().reset(context)
        self._tail = ""

    def on_boundary(self, event: Observation) -> None:
        prior = self._tail
        self._tail = (prior + event.payload.text)[-16:]
        # The source uses a case-sensitive stop string, not a reflection vocabulary.
        combined = prior + event.payload.text
        if combined.endswith("Wait") and not prior.endswith("Wait"):
            self.request(ProbeAnswer(request_id=self.request_id(), prefix=event.prefix,
                recipe="deer_qwen3_greedy_v1", temperature=0.0, score_answer=True,
                max_work=WorkLimit(generated_tokens=self.max_probe_tokens)))

    def on_probe(self, result: ProbeResult) -> None:
        value = result.value
        if (not isinstance(value, ParsedAnswer) or not value.valid or not value.complete
                or not value.reasoning_end_found):
            return
        try:
            confidence = trial_likelihood(value.token_log_probs)
        except ValueError:
            return
        if confidence > self.threshold:
            self._decision = Finalize("deer_trial_likelihood", diagnostics={
                "trial_geometric_token_likelihood": confidence, "threshold": self.threshold,
                "score_span": "trial_token_indices_1_through_end_including_delimiter",
                "trial_reasoning_end_found": True, "trial_request_id": result.request_id,
                "trial_answer_reused": False})
