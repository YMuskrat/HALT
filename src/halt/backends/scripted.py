"""Deterministic test backend. Each supplied fragment represents ONE synthetic token."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from halt.errors import CapabilityError
from halt.runtime.budgets import UsageLedger
from halt.tasks import Task
from halt.types import (
    AnswerSequenceScore,
    BackendInfo,
    CandidateScores,
    EmbedSteps,
    MethodSpec,
    NextTokenFeatures,
    ParsedAnswer,
    Phase,
    PrefixRef,
    ProbeAnswer,
    ProbeRequest,
    ProbeResult,
    RunContext,
    ScoreAnswerSequence,
    ScoreCandidates,
    ScoreFrame,
    ScoreNextToken,
    StepEmbeddings,
    TokenOutput,
    stable_hash,
)

SCRIPTED_CAPABILITIES = frozenset({"visible_reasoning", "token_stream", "step_boundaries",
    "full_vocab_scores", "selected_token_scores", "sequence_scoring", "prefix_branch",
    "exact_prefix_resume", "answer_transition", "embeddings", "cancel_request"})


class ScriptedBackend:
    def __init__(self, reasoning: Sequence[str] | None = None, answer: str = "C", *,
                 candidate_scores: Sequence[Sequence[float]] | None = None,
                 probe_answers: Sequence[str | None] | None = None,
                 next_token_features: Sequence[tuple[float, float]] | None = None,
                 embeddings: Sequence[Sequence[float]] | None = None,
                 answer_log_probs: Sequence[float] = (-0.01, -0.02),
                 answer_eos: bool = True, natural_end: bool = True,
                 fail_probe: bool = False, fail_at_token: int | None = None,
                 capabilities: frozenset[str] = SCRIPTED_CAPABILITIES,
                 candidate_token_ids: Mapping[str, int] | None = None) -> None:
        self.reasoning = tuple(reasoning if reasoning is not None else ["12", " -", " 5", " =", " 7", ".", "\n", "\n", "Check", ".", "\n\n"])
        self.answer = answer
        self.candidate_scores = tuple(tuple(x) for x in candidate_scores) if candidate_scores is not None else ()
        self.probe_answers = tuple(probe_answers) if probe_answers is not None else (answer,)
        self.next_token_features = tuple(next_token_features or [(-0.1, -5.0), (-0.2, -0.3)])
        self.embeddings = tuple(tuple(x) for x in embeddings) if embeddings is not None else ()
        self.answer_log_probs = tuple(answer_log_probs)
        self.answer_eos = answer_eos
        self.natural_end = natural_end
        self.fail_probe = fail_probe
        self.fail_at_token = fail_at_token
        self.candidate_token_ids = candidate_token_ids
        self._info = BackendInfo("scripted", "scripted", "scripted-v1", "scripted", "scripted-v1",
            "qwen3_thinking", capabilities, execution_mode="simulated", hardware={"device": "none"})
        self.opened_runs: list[ScriptedRun] = []

    @property
    def info(self) -> BackendInfo:
        return self._info

    def validate(self, task: Task, spec: MethodSpec) -> None:
        if self.candidate_token_ids is not None and task.candidates:
            ids = [self.candidate_token_ids.get(x) for x in task.candidates]
            if None in ids or len(set(ids)) != len(ids):
                raise CapabilityError("candidate labels must map to distinct contextual token IDs")

    def open(self, context: RunContext, task: Task, ledger: UsageLedger) -> ScriptedRun:
        result = ScriptedRun(self, context, task, ledger)
        self.opened_runs.append(result)
        return result


class ScriptedRun:
    def __init__(self, backend: ScriptedBackend, context: RunContext, task: Task, ledger: UsageLedger) -> None:
        self.backend, self.context, self.task, self.ledger = backend, context, task, ledger
        self.committed: list[str] = []
        self.reasoning_index = 0
        self.answer_index = 0
        self.closed = False
        self.transitioned = False
        self.forced_suffix = ""
        self.counts: dict[str, int] = {}
        self.prompt_tokens = len(task.render().split())
        op = ledger.start("prefill", Phase.PREFILL, input_tokens=self.prompt_tokens,
                          context_tokens=self.prompt_tokens, forward_calls=1)
        op.status = "completed"

    @property
    def prefix(self) -> PrefixRef:
        return PrefixRef(self.context.run_id, "main", len(self.committed),
                         stable_hash([self.task.render(), self.committed, self.forced_suffix]),
                         self.backend.info.model_revision)

    @property
    def answer_token_log_probs(self) -> tuple[float, ...]:
        return self.backend.answer_log_probs

    def next_token(self, phase: Phase) -> TokenOutput:
        if self.closed:
            raise RuntimeError("session already closed")
        self.ledger.check(generated_tokens=1)
        if self.backend.fail_at_token == len(self.committed):
            raise RuntimeError("scripted backend failure")
        if phase == Phase.REASONING:
            if self.reasoning_index < len(self.backend.reasoning):
                text = self.backend.reasoning[self.reasoning_index]
                self.reasoning_index += 1
            else:
                text = "</think>" if self.backend.natural_end else "<eos>"
        else:
            if self.answer_index < len(self.backend.answer):
                text = self.backend.answer[self.answer_index]
                self.answer_index += 1
            else:
                text = "<eos>" if self.backend.answer_eos else " "
        op = self.ledger.start("decode", phase, input_tokens=self.prompt_tokens + len(self.committed),
             recomputed_prefix_tokens=self.prompt_tokens + len(self.committed), forward_calls=1,
             context_tokens=self.prompt_tokens + len(self.committed))
        self.ledger.generated(op)
        op.status = "completed"
        self.committed.append(text)
        ended = "</think>" in "".join(self.committed)
        return TokenOutput("" if text == "<eos>" else text, len(self.committed), text == "<eos>", ended)

    def transition(self, recipe: str) -> None:
        if not self.transitioned:
            self.transitioned = True
            if "</think>" not in "".join(self.committed):
                self.forced_suffix = "</think>\n\n"
                op = self.ledger.start("forced_transition", Phase.ANSWERING, forced_context_tokens=1,
                                       context_tokens=self.prompt_tokens + len(self.committed) + 1)
                op.status = "completed"

    def probe(self, request: ProbeRequest, seed: int) -> ProbeResult:
        if request.prefix != self.prefix:
            raise ValueError("stale probe prefix")
        index = self.counts.get(type(request).__name__, 0)
        self.counts[type(request).__name__] = index + 1
        input_count = self.prompt_tokens + len(self.committed)
        op = self.ledger.start(type(request).__name__, Phase.PROBING, parent_id=request.request_id,
                              input_tokens=input_count, recomputed_prefix_tokens=input_count,
                              context_tokens=input_count, forward_calls=1)
        if self.backend.fail_probe:
            op.status = "failed"
            raise RuntimeError("scripted probe failure")
        value: Any
        frame = ScoreFrame(self.prefix)
        if isinstance(request, ScoreCandidates):
            if self.backend.candidate_scores:
                scores = self.backend.candidate_scores[min(index, len(self.backend.candidate_scores)-1)]
            else:
                scores = tuple(-0.01 if c == self.backend.answer else -8.0 for c in request.candidates)
            self.ledger.check(scored_tokens=len(request.candidates))
            op.scored_tokens += len(request.candidates)
            value = CandidateScores(request.candidates, scores, frame, request.scoring, request.length_normalize)
        elif isinstance(request, ScoreNextToken):
            top, end = self.backend.next_token_features[min(index, len(self.backend.next_token_features)-1)]
            self.ledger.check(scored_tokens=1)
            op.scored_tokens += 1
            value = NextTokenFeatures(frame, top, end, 1.0 if request.include_entropy else None)
        elif isinstance(request, ProbeAnswer):
            answer = self.backend.probe_answers[min(index, len(self.backend.probe_answers)-1)] if self.backend.probe_answers else None
            text = answer or ""
            # Completion requires an EOS synthetic token as it does in real generation.
            amount = min(len(text) + 1, request.max_work.generated_tokens)
            self.ledger.check(generated_tokens=amount)
            self.ledger.generated(op, amount)
            complete = amount > len(text)
            text = text[:amount]
            parsed = self.task.normalize(text) if complete else None
            value = ParsedAnswer(text, parsed, parsed is not None, complete,
                                 self.backend.answer_log_probs if request.score_answer else ())
        elif isinstance(request, EmbedSteps):
            op.embedding_calls += 1
            vectors = self.backend.embeddings or tuple((1.0, 0.0) for _ in request.steps)
            value = StepEmbeddings(vectors[:len(request.steps)], "scripted-embedding", "fixture-v1")
        elif isinstance(request, ScoreAnswerSequence):
            probs = self.backend.answer_log_probs if request.answer else ()
            self.ledger.check(scored_tokens=len(probs))
            op.scored_tokens += len(probs)
            value = AnswerSequenceScore(probs, request.answer, request.suffix, request.length_normalize)
        else:
            raise CapabilityError(f"unsupported scripted operation {type(request).__name__}")
        op.status = "completed"
        return ProbeResult(request.request_id, self.prefix, value, operation_ids=(op.operation_id,))

    def close(self) -> None:
        self.closed = True

    def cancel(self) -> None:
        self.closed = True
