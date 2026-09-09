"""Central work admission and an operation ledger (all quantities are observed)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Event

from halt.errors import BudgetExceeded, Cancelled
from halt.types import SCHEMA_VERSION, Budget, Phase


@dataclass
class UsageOperation:
    operation_id: str
    parent_id: str
    kind: str
    phase: str
    input_tokens: int = 0
    generated_tokens: int = 0
    forced_context_tokens: int = 0
    scored_tokens: int = 0
    recomputed_prefix_tokens: int = 0
    forward_calls: int = 0
    sequence_lengths: list[int] = field(default_factory=list)
    embedding_calls: int = 0
    duration_seconds: float = 0.0
    status: str = "started"
    role: str = "accepted"
    cached_input_tokens: int | None = None
    provider_reported_tokens: int | None = None


@dataclass(frozen=True)
class Usage:
    reasoning_tokens: int
    answer_tokens: int
    probe_output_tokens: int
    total_generated_tokens: int
    input_tokens: int
    forced_context_tokens: int
    scored_tokens: int
    recomputed_prefix_tokens: int
    forward_calls: int
    embedding_calls: int
    probe_calls: int
    cancellation_requests: int
    operations: tuple[UsageOperation, ...]
    measurement: str = "observed"
    billed_cost: float | None = None
    unknown_fields: tuple[str, ...] = ("billed_cost", "cached_input_tokens")
    schema_version: str = SCHEMA_VERSION


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class UsageLedger:
    def __init__(self, run_id: str, budget: Budget, cancellation: CancellationToken | None = None) -> None:
        self.run_id = run_id
        self.budget = budget
        self.cancellation = cancellation or CancellationToken()
        self.started = time.monotonic()
        self.operations: list[UsageOperation] = []
        self.probe_calls = 0
        self.cancellation_requests = 0
        self.measurement = "observed"
        self._scope: dict[str, int | None] | None = None
        self._scope_start: tuple[int, int, int] = (0, 0, 0)

    def check(self, *, input_tokens: int = 0, scored_tokens: int = 0,
              generated_tokens: int = 0, context_tokens: int = 0) -> None:
        if self.cancellation.cancelled:
            raise Cancelled("caller cancelled run")
        if self.budget.deadline_seconds is not None and time.monotonic() - self.started >= self.budget.deadline_seconds:
            raise Cancelled("run deadline expired")
        if context_tokens > self.budget.max_context_tokens:
            raise BudgetExceeded("invocation context length exceeds max_context_tokens")
        usage = self.snapshot()
        for name, current, additional, maximum in (
            ("generated", usage.total_generated_tokens, generated_tokens, self.budget.max_total_generated_tokens),
            ("input", usage.input_tokens, input_tokens, self.budget.max_input_tokens),
            ("scored", usage.scored_tokens, scored_tokens, self.budget.max_scored_tokens),
        ):
            if maximum is not None and current + additional > maximum:
                raise BudgetExceeded(f"{name} work budget exhausted")
        if self._scope is not None:
            for i, (name, current, extra) in enumerate((
                ("generated_tokens", usage.total_generated_tokens, generated_tokens),
                ("input_tokens", usage.input_tokens, input_tokens),
                ("scored_tokens", usage.scored_tokens, scored_tokens),
            )):
                limit = self._scope[name]
                if limit is not None and current - self._scope_start[i] + extra > limit:
                    raise BudgetExceeded(f"probe {name} work limit exhausted")

    def start(self, kind: str, phase: Phase | str, *, parent_id: str | None = None,
              input_tokens: int = 0, scored_tokens: int = 0,
              forced_context_tokens: int = 0, recomputed_prefix_tokens: int = 0,
              forward_calls: int = 0, embedding_calls: int = 0,
              context_tokens: int = 0, role: str | None = None) -> UsageOperation:
        self.check(input_tokens=input_tokens, scored_tokens=scored_tokens, context_tokens=context_tokens)
        op = UsageOperation(f"{self.run_id}:op:{len(self.operations)}", parent_id or self.run_id,
                            kind, str(phase), input_tokens=input_tokens, scored_tokens=scored_tokens,
                            forced_context_tokens=forced_context_tokens,
                            recomputed_prefix_tokens=recomputed_prefix_tokens,
                            forward_calls=forward_calls, embedding_calls=embedding_calls,
                            role=role or ("discarded" if phase == Phase.PROBING else "accepted"))
        if context_tokens:
            op.sequence_lengths.append(context_tokens)
        self.operations.append(op)
        return op

    def generated(self, operation: UsageOperation, count: int = 1) -> None:
        # Call check(generated_tokens=1) BEFORE dispatch, then record successful output here.
        operation.generated_tokens += count

    def begin_probe(self, generated_tokens: int, scored_tokens: int | None,
                    input_tokens: int | None) -> None:
        self.check()
        if self.probe_calls >= self.budget.max_probe_calls:
            raise BudgetExceeded("probe call budget exhausted")
        self.probe_calls += 1
        usage = self.snapshot()
        self._scope_start = (usage.total_generated_tokens, usage.input_tokens, usage.scored_tokens)
        self._scope = dict(generated_tokens=generated_tokens, scored_tokens=scored_tokens,
                           input_tokens=input_tokens)

    def end_probe(self) -> None:
        self._scope = None

    def accept_probe(self, operation_ids: tuple[str, ...]) -> None:
        for op in self.operations:
            if op.operation_id in operation_ids:
                op.role = "accepted"

    def snapshot(self) -> Usage:
        def total(key: str, phase: str | None = None) -> int:
            return sum(getattr(op, key) for op in self.operations if phase is None or op.phase == phase)
        return Usage(
            total("generated_tokens", "reasoning"), total("generated_tokens", "answering"),
            total("generated_tokens", "probing"), total("generated_tokens"), total("input_tokens"),
            total("forced_context_tokens"), total("scored_tokens"), total("recomputed_prefix_tokens"),
            total("forward_calls"), total("embedding_calls"), self.probe_calls,
            self.cancellation_requests, tuple(self.operations), measurement=self.measurement,
        )
