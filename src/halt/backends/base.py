from __future__ import annotations

from typing import Protocol

from halt.runtime.budgets import UsageLedger
from halt.tasks import Task
from halt.types import (
    BackendInfo,
    MethodSpec,
    Phase,
    PrefixRef,
    ProbeRequest,
    ProbeResult,
    RunContext,
    TokenOutput,
)


class BackendRun(Protocol):
    @property
    def prefix(self) -> PrefixRef: ...
    def next_token(self, phase: Phase) -> TokenOutput: ...
    def transition(self, recipe: str) -> None: ...
    def probe(self, request: ProbeRequest, seed: int) -> ProbeResult: ...
    def close(self) -> None: ...
    def cancel(self) -> None: ...


class Backend(Protocol):
    @property
    def info(self) -> BackendInfo: ...
    def validate(self, task: Task, spec: MethodSpec) -> None: ...
    def open(self, context: RunContext, task: Task, ledger: UsageLedger) -> BackendRun: ...
