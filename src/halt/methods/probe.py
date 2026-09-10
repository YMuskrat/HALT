"""Optional signal-request plumbing for the public method lifecycle.

Use BaseMethod directly when a method needs a different event policy. This helper
does not choose signals, thresholds, measurement frequency, or stopping rules.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from halt.errors import ConfigurationError
from halt.methods.base import BaseMethod
from halt.types import (
    AnswerSequenceScore,
    CandidateScores,
    Continue,
    Decision,
    EmbedSteps,
    EventKind,
    Finalize,
    NextTokenFeatures,
    Observation,
    ParsedAnswer,
    Phase,
    PrefixRef,
    ProbeAnswer,
    ProbeFailure,
    ProbeRequest,
    ProbeResult,
    RequestSignals,
    ReturnAnswer,
    RunContext,
    ScoreAnswerSequence,
    ScoreCandidates,
    ScoreNextToken,
    StepBoundary,
    StepEmbeddings,
    WorkLimit,
)

_Request = TypeVar("_Request", bound=ProbeRequest)
_RESULT_TYPES = {ScoreCandidates: CandidateScores, ScoreNextToken: NextTokenFeatures,
                 ProbeAnswer: ParsedAnswer, EmbedSteps: StepEmbeddings,
                 ScoreAnswerSequence: AnswerSequenceScore}


class ProbeMethod(BaseMethod):
    """Receive valid signals through ``on_probe`` or a batch through ``on_probes``.

    One batch is outstanding at a time. Batches are delivered in request order
    only after every request terminates. A failed, denied, or mismatched result
    invalidates the entire batch and calls ``invalid_probe`` once. Unknown and
    duplicate terminal events are ignored. No callback executes model work.
    """

    def reset(self, context: RunContext) -> None:
        super().reset(context)
        self._decision: Decision = Continue()
        self._pending: dict[str, ProbeRequest] = {}
        self._results: dict[str, ProbeResult | None] = {}
        self._used_ids: set[str] = set()
        self._accepted: dict[str, ProbeResult] = {}
        self._request_index = 0
        self._prefix: PrefixRef | None = None
        self._phase = Phase.PREFILL

    def request_id(self) -> str:
        self._request_index += 1
        return f"{self.context.run_id}:{self.spec.method_id}:{self._request_index}"

    def signal(self, request_type: type[_Request], *, max_work: WorkLimit,
               recipe: str | None = None, **parameters: Any) -> _Request:
        """Build a typed request at the current event prefix, without dispatching it.

        Work limits are explicit upper bounds; the runtime enforces its own
        remaining budgets as well. Use ``request(signal(...), signal(...))`` for
        a batch, or ``request_signal`` for one measurement.
        """
        if self._prefix is None or self._phase not in {Phase.REASONING, Phase.PROBING}:
            raise ConfigurationError("signals require a current reasoning observation")
        if not isinstance(request_type, type) or not issubclass(request_type, ProbeRequest):
            raise ConfigurationError("signal type must be a typed ProbeRequest")
        if {"request_id", "prefix"} & parameters.keys():
            raise ConfigurationError("signal request IDs and prefixes are supplied by ProbeMethod")
        return request_type(request_id=self.request_id(), prefix=self._prefix,
                            max_work=max_work, recipe=recipe or self.spec.finalization_recipe,
                            **parameters)

    def request_signal(self, request_type: type[_Request], *, max_work: WorkLimit,
                       recipe: str | None = None, **parameters: Any) -> str:
        """Queue one typed measurement and return its ID for optional diagnostics."""
        request = self.signal(request_type, max_work=max_work, recipe=recipe, **parameters)
        self.request(request)
        return request.request_id

    def request(self, *requests: ProbeRequest) -> None:
        """Queue one request or a batch; runtime dispatch remains budget controlled."""
        if not requests or self._pending:
            raise ConfigurationError("request a nonempty batch only after the previous batch terminates")
        identifiers = [request.request_id for request in requests]
        if len(set(identifiers)) != len(identifiers) or self._used_ids.intersection(identifiers):
            raise ConfigurationError("probe request IDs must be unique within a run")
        if (self._phase not in {Phase.REASONING, Phase.PROBING}
                or any(request.prefix != self._prefix for request in requests)):
            raise ConfigurationError("probe requests must use the current observation prefix")
        self._pending = {request.request_id: request for request in requests}
        self._results = {}
        self._used_ids.update(identifiers)
        self._decision = RequestSignals(tuple(requests))

    def finalize(self, reason: str = "method_stop", *, recipe: str | None = None,
                 diagnostics: Mapping[str, Any] | None = None) -> None:
        """Ask the runtime to end reasoning and generate the final answer."""
        self._decision = Finalize(reason, recipe, diagnostics or {})

    def return_answer(self, answer: str, source: ProbeResult, reason: str = "method_stop", *,
                      diagnostics: Mapping[str, Any] | None = None) -> None:
        """Return an answer from an accepted probe; runtime checks answer provenance."""
        if self._accepted.get(source.request_id) != source:
            raise ConfigurationError("returned answers require a successfully received probe")
        self._decision = ReturnAnswer(answer, source.request_id, reason, diagnostics or {})

    def observe(self, event: Observation) -> None:
        if event.prefix.run_id != self.context.run_id:
            return
        self._phase = event.phase
        if event.phase in {Phase.ANSWERING, Phase.FINISHED}:
            self._pending.clear()
            self._results.clear()
            self._decision = Continue()
            return
        if event.kind == EventKind.PROBE_SCHEDULED:
            if (isinstance(event.payload, ProbeRequest)
                    and event.payload.request_id in self._pending):
                self._decision = Continue()
            return
        if event.kind in {EventKind.PROBE_COMPLETED, EventKind.PROBE_DENIED, EventKind.PROBE_FAILED}:
            value = event.payload
            if not isinstance(value, ProbeResult | ProbeFailure):
                return
            if value.request_id not in self._pending or value.request_id in self._results:
                return
            request = self._pending[value.request_id]
            expected = _RESULT_TYPES.get(type(request))
            valid = (isinstance(value, ProbeResult) and value.valid
                     and value.prefix == request.prefix and event.prefix == request.prefix
                     and (expected is None or isinstance(value.value, expected)))
            self._decision = Continue()
            self._results[value.request_id] = value if valid and isinstance(value, ProbeResult) else None
            if len(self._results) == len(self._pending):
                results = tuple(self._results[key] for key in self._pending)
                self._pending.clear()
                self._results.clear()
                self._decision = Continue()
                if any(result is None for result in results):
                    self.invalid_probe()
                else:
                    accepted = tuple(result for result in results if result is not None)
                    self._accepted.update((result.request_id, result) for result in accepted)
                    self.on_probes(accepted)
            return
        self._prefix = event.prefix
        if event.kind == EventKind.STEP_BOUNDARY and isinstance(event.payload, StepBoundary):
            if not self._pending:
                self._decision = Continue()
                self.on_boundary(event)
        else:
            self.on_event(event)

    def on_event(self, event: Observation) -> None:
        """Optional hook for token, score, and lifecycle events outside probe plumbing."""

    def on_boundary(self, event: Observation) -> None:
        """Optional hook for the boundary policy declared in MethodSpec."""

    def on_probes(self, results: tuple[ProbeResult, ...]) -> None:
        """Override for joint batch decisions; the default calls on_probe in order."""
        for result in results:
            self.on_probe(result)

    def on_probe(self, result: ProbeResult) -> None:
        """Receive a valid result whose request ID and source prefix were matched."""

    def invalid_probe(self) -> None:
        """Reset method-specific evidence after an invalid, failed, or denied batch."""

    def decide(self) -> Decision:
        return self._decision
