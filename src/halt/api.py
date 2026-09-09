"""Small public synchronous API. Runtime owns execution; methods only make decisions."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from halt.backends.base import Backend
from halt.methods.base import HaltMethod
from halt.runtime.budgets import CancellationToken
from halt.runtime.engine import Runtime
from halt.tasks import Task
from halt.types import Budget, EventKind, Observation, RunResult


class HaltRunner:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def inspect(self, task: Task, method: HaltMethod, budget: Budget | None = None) -> dict[str, Any]:
        return Runtime.preflight(self.backend, task, method, budget or Budget())

    def stream(self, task: Task, method: HaltMethod, budget: Budget | None = None, *, seed: int = 0,
               capture: str = "none", trace_path: str | Path | None = None,
               session: Any = None, run_id: str | None = None,
               cancellation: CancellationToken | None = None) -> Iterator[Observation]:
        return Runtime(self.backend).stream(task, method, budget or Budget(), seed=seed,
            capture=capture, trace_path=trace_path, session=session, run_id=run_id,
            cancellation=cancellation)

    def run(self, task: Task, method: HaltMethod, budget: Budget | None = None, *, seed: int = 0,
            capture: str = "none", trace_path: str | Path | None = None,
            session: Any = None, run_id: str | None = None,
            cancellation: CancellationToken | None = None) -> RunResult:
        result = None
        for event in self.stream(task, method, budget, seed=seed, capture=capture,
                trace_path=trace_path, session=session, run_id=run_id, cancellation=cancellation):
            if event.kind == EventKind.RUN_FINISHED:
                result = event.payload
        if not isinstance(result, RunResult):
            raise RuntimeError("runtime did not produce a RunFinished result")
        return result
