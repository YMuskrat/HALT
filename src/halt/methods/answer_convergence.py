"""Answer-consistency strategy; streaming common-protocol adaptation, not NLTK parity."""
from __future__ import annotations

from dataclasses import dataclass

from halt.errors import ConfigurationError
from halt.methods._research import ProbeMethod
from halt.types import (
    BoundaryPolicy,
    MethodSpec,
    Observation,
    ParsedAnswer,
    ProbeAnswer,
    ProbeResult,
    Requirements,
    ReturnAnswer,
    RunContext,
    WorkLimit,
)


@dataclass
class AnswerConvergence(ProbeMethod):
    consecutive: int = 3
    max_probe_tokens: int = 32

    def __post_init__(self) -> None:
        if type(self.consecutive) is not int or self.consecutive < 2:
            raise ConfigurationError("answer consistency requires at least two consecutive answers")
        if type(self.max_probe_tokens) is not int or self.max_probe_tokens < 1:
            raise ConfigurationError("max_probe_tokens must be positive")

    @property
    def spec(self) -> MethodSpec:
        return MethodSpec("answer_convergence", variant="answer_consistency_sentence_common_v1",
            requirements=Requirements(frozenset({"visible_reasoning", "token_stream",
                "step_boundaries", "prefix_branch", "exact_prefix_resume", "answer_transition"})),
            boundary=BoundaryPolicy("sentence"), can_return_answer=True)

    def reset(self, context: RunContext) -> None:
        super().reset(context)
        self._previous: str | None = None
        self._streak = 0

    def on_boundary(self, event: Observation) -> None:
        self.request(ProbeAnswer(request_id=self.request_id(), prefix=event.prefix,
            recipe="answer_consistency_sentence_common_v1", temperature=0.0,
            max_work=WorkLimit(generated_tokens=self.max_probe_tokens)))

    def on_probe(self, result: ProbeResult) -> None:
        value = result.value
        if (not isinstance(value, ParsedAnswer) or not value.valid or not value.complete
                or value.answer is None or not value.answer.strip()):
            self.invalid_probe()
            return
        self._streak = self._streak + 1 if value.answer == self._previous else 1
        self._previous = value.answer
        if self._streak >= self.consecutive:
            self._decision = ReturnAnswer(value.answer, result.request_id,
                "answer_consistency", {"consecutive_identical_answers": self._streak,
                "probe_temperature": 0.0})

    def invalid_probe(self) -> None:
        self._previous = None
        self._streak = 0
