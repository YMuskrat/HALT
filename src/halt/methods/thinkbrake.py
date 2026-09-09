"""ThinkBrake v1 log margin evaluated after the committed newline prefix."""
from __future__ import annotations

import math
from dataclasses import dataclass

from halt.errors import ConfigurationError
from halt.methods._research import ProbeMethod
from halt.types import (
    BoundaryPolicy,
    Finalize,
    MethodSpec,
    NextTokenFeatures,
    Observation,
    ProbeResult,
    Requirements,
    ScoreNextToken,
    WorkLimit,
)


@dataclass
class ThinkBrake(ProbeMethod):
    threshold: float = 0.25

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ConfigurationError("ThinkBrake threshold must be finite and nonnegative (nats)")

    @property
    def spec(self) -> MethodSpec:
        return MethodSpec("thinkbrake", variant="thinkbrake_v1_newline_raw",
            requirements=Requirements(frozenset({"visible_reasoning", "token_stream",
                "step_boundaries", "prefix_branch", "exact_prefix_resume", "answer_transition"}),
                alternatives=(frozenset({"full_vocab_scores", "selected_token_scores"}),),
                profiles=("qwen3_thinking", "scripted")),
            boundary=BoundaryPolicy("newline"))

    def on_boundary(self, event: Observation) -> None:
        self.request(ScoreNextToken(request_id=self.request_id(), prefix=event.prefix,
            processing="raw", max_work=WorkLimit(scored_tokens=1)))

    def on_probe(self, result: ProbeResult) -> None:
        value = result.value
        if (not isinstance(value, NextTokenFeatures) or value.frame.prefix != result.prefix
                or value.frame.processing != "raw"
                or value.frame.vocabulary_coverage not in {"full", "selected_exact"}
                or not math.isfinite(value.top_continuation_log_prob)
                or not math.isfinite(value.reasoning_end_log_prob)):
            return
        margin = value.top_continuation_log_prob - value.reasoning_end_log_prob
        if margin <= self.threshold:
            self._decision = Finalize("thinkbrake_margin", diagnostics={
                "top_minus_end_log_probability": margin, "units": "nats",
                "scored_prefix_position": result.prefix.position,
                "score_processing": "raw", "boundary_side": "after_committed_newline"})
