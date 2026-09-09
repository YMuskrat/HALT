from dataclasses import dataclass

from halt.errors import ConfigurationError
from halt.methods.base import BaseMethod
from halt.types import (
    Continue,
    Decision,
    EventKind,
    Finalize,
    MethodSpec,
    Observation,
    RunContext,
)


@dataclass
class FullReasoning(BaseMethod):
    spec = MethodSpec("full_reasoning")


@dataclass
class FixedReasoningBudget(BaseMethod):
    tokens: int = 32
    spec = MethodSpec("fixed_reasoning_budget")

    def __post_init__(self) -> None:
        if type(self.tokens) is not int or self.tokens < 0:
            raise ConfigurationError("tokens must be a nonnegative integer")

    def reset(self, context: RunContext) -> None:
        super().reset(context)
        self._tokens = 0

    def observe(self, event: Observation) -> None:
        if event.kind == EventKind.TOKEN_COMMITTED:
            self._tokens += 1

    def decide(self) -> Decision:
        return Finalize("fixed_reasoning_budget") if self._tokens >= self.tokens else Continue()


@dataclass
class ImmediateAnswer(BaseMethod):
    spec = MethodSpec("immediate_answer")

    def decide(self) -> Decision:
        return Finalize("immediate_answer")
