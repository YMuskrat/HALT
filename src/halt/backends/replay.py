"""Replay only recorded operations. Missing counterfactuals raise ReplayUnavailable."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from halt.errors import ReplayUnavailable
from halt.runtime.budgets import UsageLedger
from halt.tasks import Task
from halt.types import (
    AnswerSequenceScore,
    BackendInfo,
    CandidateScores,
    MethodSpec,
    NextTokenFeatures,
    ParsedAnswer,
    Phase,
    PrefixRef,
    ProbeRequest,
    ProbeResult,
    RunContext,
    ScoreFrame,
    StepEmbeddings,
    TokenOutput,
    stable_hash,
    to_data,
)


def _key(request: dict[str, Any], operation: str) -> str:
    request = json.loads(json.dumps(request))
    request.pop("request_id", None)
    request["prefix"].pop("run_id", None)
    return stable_hash({"operation": operation, **request})


def _value(data: dict[str, Any], kind: str, run_id: str) -> Any:
    data = dict(data)
    if "frame" in data:
        frame = dict(data["frame"])
        frame["prefix"] = PrefixRef(**{**frame["prefix"], "run_id": run_id})
        data["frame"] = ScoreFrame(**frame)
    for key in ("candidates", "log_scores", "token_log_probs"):
        if key in data:
            data[key] = tuple(data[key])
    if "vectors" in data:
        data["vectors"] = tuple(tuple(x) for x in data["vectors"])
    classes = {cls.__name__: cls for cls in (CandidateScores, ParsedAnswer, NextTokenFeatures,
                                           StepEmbeddings, AnswerSequenceScore)}
    if kind not in classes:
        raise ReplayUnavailable(f"unsupported recorded result type {kind}")
    return classes[kind](**data)


class ReplayBackend:
    def __init__(self, trace: str | Path) -> None:
        self.records = [json.loads(line) for line in Path(trace).read_text(encoding="utf-8").splitlines() if line]
        started = next((r for r in self.records if r["kind"] == "RunStarted"), None)
        if started is None or started["payload"].get("capture") != "full":
            raise ReplayUnavailable("replay requires a complete capture=full trace")
        if any(r.get("schema_version") != "1.0" for r in self.records):
            raise ReplayUnavailable("unsupported recorded schema version")
        self.started = started
        info = dict(started["payload"]["context"]["backend"])
        for name in ("capabilities", "emulated_capabilities"):
            info[name] = frozenset(info[name])
        info.update(name="replay", execution_mode="recorded_observations")
        self._info = BackendInfo(**info)

    @property
    def info(self) -> BackendInfo:
        return self._info

    def validate(self, task: Task, spec: MethodSpec) -> None:
        if stable_hash(task.visible()) != self.started["payload"]["task_hash"]:
            raise ReplayUnavailable("task differs from the recorded inference task")

    def open(self, context: RunContext, task: Task, ledger: UsageLedger) -> ReplayRun:
        return ReplayRun(self, context, ledger)


class ReplayRun:
    def __init__(self, backend: ReplayBackend, context: RunContext, ledger: UsageLedger) -> None:
        self.backend, self.context, self.ledger = backend, context, ledger
        ledger.measurement = "replay_controller_only; token counts are recorded, source inference work excluded"
        self._prefix = PrefixRef(**{**backend.started["prefix"], "run_id": context.run_id})
        self.tokens = [r for r in backend.records if r["kind"] == "TokenCommitted"]
        self.transitions = [r for r in backend.records if r["kind"] == "PhaseChanged" and r["phase"] == "answering"]
        self.token_index = 0
        self.closed = False
        self.answer_token_log_probs: tuple[float, ...] = ()
        self.probes: dict[str, dict[str, Any]] = {}
        terminal = {r["payload"]["request_id"]: r for r in backend.records
                    if r["kind"] in {"ProbeCompleted", "ProbeFailed"}}
        for row in backend.records:
            if row["kind"] == "ProbeScheduled":
                match = terminal.get(row["payload"]["request_id"])
                if match is not None:
                    self.probes[_key(row["payload"], row["payload_type"])] = match

    @property
    def prefix(self) -> PrefixRef:
        return self._prefix

    def next_token(self, phase: Phase) -> TokenOutput:
        if self.closed:
            raise RuntimeError("replay closed")
        if self.token_index >= len(self.tokens):
            raise ReplayUnavailable("main continuation beyond recorded tokens unavailable")
        row = self.tokens[self.token_index]
        if row["phase"] != str(phase) or row["prefix"]["position"] != self.prefix.position + 1:
            raise ReplayUnavailable("requested phase/prefix has no recorded next token")
        self.ledger.check(generated_tokens=1)
        op = self.ledger.start("replay_token", phase)
        self.ledger.generated(op)
        op.status = "recorded"
        self._prefix = PrefixRef(**{**row["prefix"], "run_id": self.context.run_id})
        self.token_index += 1
        return TokenOutput(**row["payload"])

    def transition(self, recipe: str) -> None:
        row = next((r for r in self.transitions if r["prefix"]["position"] == self.prefix.position
                    and r["payload"].get("recipe") == recipe), None)
        if row is None:
            raise ReplayUnavailable("answer transition at this prefix/recipe was not recorded")
        self._prefix = PrefixRef(**{**row["prefix"], "run_id": self.context.run_id})

    def probe(self, request: ProbeRequest, seed: int) -> ProbeResult:
        row = self.probes.get(_key(to_data(request), type(request).__name__))
        if row is None:
            raise ReplayUnavailable(f"{type(request).__name__} not recorded at prefix {request.prefix.position} with these settings")
        if row["kind"] == "ProbeFailed":
            raise RuntimeError(row["payload"]["error"])
        data = row["payload"]
        value = _value(data["value"], row["value_type"], self.context.run_id)
        # Replay is controller work. Source work is retained in the trace, never billed as new inference.
        op = self.ledger.start("replay_probe", Phase.PROBING, parent_id=request.request_id)
        op.status = "recorded"
        return ProbeResult(request.request_id, self.prefix, value, data["valid"], (op.operation_id,),
                           data["branch_outcome"], data["error"])

    def cancel(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True
