"""Shared event plumbing; all published stopping equations remain in their methods."""
from __future__ import annotations

from halt.methods.base import BaseMethod
from halt.types import (
    Continue,
    Decision,
    EventKind,
    Observation,
    Phase,
    ProbeFailure,
    ProbeRequest,
    ProbeResult,
    RequestSignals,
    RunContext,
    StepBoundary,
)


class ProbeMethod(BaseMethod):
    def reset(self, context: RunContext) -> None:
        super().reset(context)
        self._decision: Decision = Continue()
        self._request: ProbeRequest | None = None
        self._request_index = 0

    def request_id(self) -> str:
        self._request_index += 1
        return f"{self.context.run_id}:{self.spec.method_id}:{self._request_index}"

    def request(self, request: ProbeRequest) -> None:
        self._request = request
        self._decision = RequestSignals((request,))

    def observe(self, event: Observation) -> None:
        if event.phase in {Phase.ANSWERING, Phase.FINISHED}:
            self._decision = Continue()
            return
        if event.kind == EventKind.PROBE_SCHEDULED:
            self._decision = Continue()
        elif event.kind in {EventKind.PROBE_DENIED, EventKind.PROBE_FAILED}:
            if (isinstance(event.payload, ProbeFailure) and self._request is not None
                    and event.payload.request_id == self._request.request_id):
                self._request = None
                self._decision = Continue()
                self.invalid_probe()
        elif event.kind == EventKind.PROBE_COMPLETED:
            result = event.payload
            if not isinstance(result, ProbeResult) or self._request is None:
                return
            if result.request_id != self._request.request_id:
                return
            request = self._request
            self._request = None
            self._decision = Continue()
            if not result.valid or result.prefix != request.prefix or event.prefix != request.prefix:
                self.invalid_probe()
                return
            self.on_probe(result)
        elif event.kind == EventKind.STEP_BOUNDARY and isinstance(event.payload, StepBoundary):
            self._decision = Continue()
            self.on_boundary(event)

    def on_boundary(self, event: Observation) -> None:
        pass

    def on_probe(self, result: ProbeResult) -> None:
        pass

    def invalid_probe(self) -> None:
        pass

    def decide(self) -> Decision:
        return self._decision
