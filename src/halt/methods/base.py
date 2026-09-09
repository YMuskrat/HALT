from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol

from halt.types import Continue, Decision, MethodSpec, Observation, RunContext


class HaltMethod(Protocol):
    @property
    def spec(self) -> MethodSpec: ...
    def reset(self, context: RunContext) -> None: ...
    def observe(self, event: Observation) -> None: ...
    def decide(self) -> Decision: ...


class BaseMethod:
    @property
    def spec(self) -> MethodSpec:
        return MethodSpec("custom_method")

    def reset(self, context: RunContext) -> None:
        self.context = context

    def observe(self, event: Observation) -> None:
        pass

    def decide(self) -> Decision:
        return Continue()

    def configuration(self) -> dict[str, Any]:
        if is_dataclass(self) and not isinstance(self, type):
            return asdict(self)
        return {k: v for k, v in vars(self).items() if not k.startswith("_") and k != "context"}

    def spawn(self) -> BaseMethod:
        return copy.deepcopy(self)
