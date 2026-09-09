"""Versioned, dependency-free contracts shared by methods, backends and evaluators."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from halt.errors import ConfigurationError

SCHEMA_VERSION = "1.0"
PLUGIN_API_VERSION = "1"


def to_data(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_data(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): to_data(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [to_data(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted(to_data(v) for v in value)
    if isinstance(value, StrEnum):
        return str(value)
    return value


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(to_data(value), sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


class Phase(StrEnum):
    PREFILL = "prefill"
    REASONING = "reasoning"
    PROBING = "probing"
    ANSWERING = "answering"
    FINISHED = "finished"


class EventKind(StrEnum):
    RUN_STARTED = "RunStarted"
    NEXT_TOKEN_SCORES = "NextTokenScores"
    TOKEN_COMMITTED = "TokenCommitted"
    STEP_BOUNDARY = "StepBoundary"
    PROBE_SCHEDULED = "ProbeScheduled"
    PROBE_COMPLETED = "ProbeCompleted"
    PROBE_FAILED = "ProbeFailed"
    PROBE_DENIED = "ProbeDenied"
    PHASE_CHANGED = "PhaseChanged"
    RUN_FINISHED = "RunFinished"


class RunStatus(StrEnum):
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    ANSWER_INCOMPLETE = "answer_incomplete"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    BACKEND_ERROR = "backend_error"
    METHOD_ERROR = "method_error"


@dataclass(frozen=True)
class Budget:
    max_reasoning_tokens: int = 1024
    max_answer_tokens: int = 128
    max_probe_output_tokens: int = 256
    max_total_generated_tokens: int = 1408
    max_probe_calls: int = 8
    max_probes_per_prefix: int = 8
    max_scored_tokens: int | None = None
    max_input_tokens: int | None = None
    max_context_tokens: int = 32768
    deadline_seconds: float | None = None
    cost_ceiling: float | None = None

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name.startswith("max_") and v is not None:
                if type(v) is not int or v < 0:
                    raise ConfigurationError(f"{f.name} must be a nonnegative integer")
        if self.max_context_tokens == 0:
            raise ConfigurationError("max_context_tokens must be positive")
        if self.max_total_generated_tokens < self.max_answer_tokens:
            raise ConfigurationError("total generated allowance must reserve the answer allowance")
        if self.deadline_seconds is not None and (
            not math.isfinite(self.deadline_seconds) or self.deadline_seconds <= 0
        ):
            raise ConfigurationError("deadline_seconds must be finite and positive")
        if self.cost_ceiling is not None:
            raise ConfigurationError("hard monetary ceilings are unsupported: no billing provider")


@dataclass(frozen=True)
class PrefixRef:
    run_id: str
    trajectory_id: str
    position: int
    identity: str
    model_revision: str


@dataclass(frozen=True)
class Requirements:
    capabilities: frozenset[str] = frozenset({"visible_reasoning", "token_stream", "answer_transition"})
    alternatives: tuple[frozenset[str], ...] = ()
    allow_emulated: bool = True
    profiles: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundaryPolicy:
    name: str = "blank_line"
    version: str = "1"
    chunk_tokens: int = 32


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    variant: str = "baseline"
    requirements: Requirements = field(default_factory=Requirements)
    boundary: BoundaryPolicy = field(default_factory=BoundaryPolicy)
    finalization_recipe: str = "halt_default_v1"
    probe_failure_policy: str = "continue"
    can_return_answer: bool = False
    can_abstain: bool = False
    implementation_status: str = "functional"
    source_relationship: str = "independent_implementation"
    recipe: str = "common_protocol"
    verification: tuple[str, ...] = ("component_fixtures",)
    api_version: str = PLUGIN_API_VERSION
    signal_operations: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendInfo:
    name: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    profile: str
    capabilities: frozenset[str]
    execution_mode: str = "recompute"
    emulated_capabilities: frozenset[str] = frozenset()
    versions: Mapping[str, str] = field(default_factory=dict)
    hardware: Mapping[str, Any] = field(default_factory=dict)
    execution_settings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunContext:
    run_id: str
    method_id: str
    config_hash: str
    task_visible: Mapping[str, Any]
    backend: BackendInfo
    seed: int
    probe_seed: int
    budget: Budget
    session_reference: str | None = None


@dataclass(frozen=True)
class ScoreFrame:
    prefix: PrefixRef
    domain: str = "log_probability"
    processing: str = "raw"
    vocabulary_coverage: str = "selected_exact"
    tokenizer_identity: str = "scripted-v1"


@dataclass(frozen=True)
class SignalValue:
    name: str
    value: Any
    units: str
    version: str = "1"
    scope: str = "prefix"
    valid: bool = True
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkLimit:
    generated_tokens: int = 0
    scored_tokens: int | None = None
    input_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("generated_tokens", "scored_tokens", "input_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ConfigurationError(f"probe {name} must be a nonnegative integer")


@dataclass(frozen=True, kw_only=True)
class ProbeRequest:
    request_id: str
    prefix: PrefixRef
    max_work: WorkLimit = field(default_factory=WorkLimit)
    recipe: str = "halt_default_v1"

    def fingerprint(self) -> str:
        data = to_data(self)
        data.pop("request_id")
        return stable_hash({"operation": type(self).__name__, **data})


@dataclass(frozen=True, kw_only=True)
class ScoreCandidates(ProbeRequest):
    candidates: tuple[str, ...]
    scoring: str = "candidate_next_token"
    suffix: str = "\n"
    length_normalize: bool = False


@dataclass(frozen=True, kw_only=True)
class ProbeAnswer(ProbeRequest):
    temperature: float = 0.0
    induction: str = ""
    score_answer: bool = False


@dataclass(frozen=True, kw_only=True)
class ScoreNextToken(ProbeRequest):
    token_ids: tuple[int, ...] = ()
    include_entropy: bool = False
    include_end_margin: bool = True
    processing: str = "raw"


@dataclass(frozen=True, kw_only=True)
class EmbedSteps(ProbeRequest):
    steps: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class ScoreAnswerSequence(ProbeRequest):
    answer: str
    suffix: str = "\n"
    length_normalize: bool = False


@dataclass(frozen=True)
class CandidateScores:
    candidates: tuple[str, ...]
    log_scores: tuple[float, ...]
    frame: ScoreFrame
    scoring: str = "candidate_next_token"
    length_normalize: bool = False


@dataclass(frozen=True)
class ParsedAnswer:
    raw_text: str
    answer: str | None
    valid: bool
    complete: bool = True
    token_log_probs: tuple[float, ...] = ()
    reasoning_end_found: bool = True


@dataclass(frozen=True)
class NextTokenFeatures:
    frame: ScoreFrame
    top_continuation_log_prob: float
    reasoning_end_log_prob: float
    entropy: float | None = None
    selected_log_probs: Mapping[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StepEmbeddings:
    vectors: tuple[tuple[float, ...], ...]
    encoder: str
    revision: str
    normalized: bool = True


@dataclass(frozen=True)
class AnswerSequenceScore:
    token_log_probs: tuple[float, ...]
    answer: str
    suffix: str = "\n"
    length_normalize: bool = False


@dataclass(frozen=True)
class ProbeResult:
    request_id: str
    prefix: PrefixRef
    value: CandidateScores | ParsedAnswer | NextTokenFeatures | StepEmbeddings | AnswerSequenceScore | None
    valid: bool = True
    operation_ids: tuple[str, ...] = ()
    branch_outcome: str = "discarded"
    error: str | None = None


@dataclass(frozen=True)
class TokenOutput:
    text: str
    token_id: int
    eos: bool = False
    reasoning_end: bool = False


@dataclass(frozen=True)
class StepBoundary:
    text: str
    index: int
    policy: str


@dataclass(frozen=True)
class ProbeFailure:
    request_id: str
    error: str


@dataclass(frozen=True)
class Observation:
    event_id: str
    kind: EventKind
    timestamp: float
    phase: Phase
    prefix: PrefixRef
    payload: Any = None
    parent_event_id: str | None = None
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class Continue:
    pass


@dataclass(frozen=True)
class RequestSignals:
    requests: tuple[ProbeRequest, ...]


@dataclass(frozen=True)
class Finalize:
    reason: str = "method_stop"
    recipe: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReturnAnswer:
    answer: str
    provenance: str
    reason: str = "method_stop"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Abstain:
    reason: str = "method_abstain"


Decision = Continue | RequestSignals | Finalize | ReturnAnswer | Abstain


@dataclass(frozen=True)
class StopInfo:
    reason: str
    decision_event_id: str | None
    reasoning_position: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    finalization_outcome: str = "not_started"


@dataclass(frozen=True)
class RunResult:
    run_id: str
    answer: str | None
    status: RunStatus
    stop: StopInfo
    usage: Any
    durations: Mapping[str, float]
    provenance: Mapping[str, Any]
    partial_answer: str = ""
    trace_reference: str | None = None
    error: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return dict(to_data(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)
