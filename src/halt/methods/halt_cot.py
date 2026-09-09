"""HALT-CoT entropy controller; see docs/audits/halt_cot.md for source and changes."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from halt.errors import ConfigurationError
from halt.methods._research import ProbeMethod
from halt.signals import entropy, softmax
from halt.tasks import Task
from halt.types import (
    BoundaryPolicy,
    CandidateScores,
    MethodSpec,
    Observation,
    ProbeResult,
    Requirements,
    ReturnAnswer,
    RunContext,
    ScoreCandidates,
    WorkLimit,
)


@dataclass
class HaltCoTConfig:
    threshold: float = 0.6
    entropy_unit: str = "bits"
    consecutive: int = 2
    scoring: str = "candidate_next_token"

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ConfigurationError("threshold must be finite and nonnegative")
        if self.entropy_unit not in {"bits", "nats"}:
            raise ConfigurationError("entropy_unit must be bits or nats")
        if type(self.consecutive) is not int or self.consecutive < 1:
            raise ConfigurationError("consecutive must be a positive integer")
        if self.scoring not in {"candidate_next_token", "candidate_sequence"}:
            raise ConfigurationError("scoring must be candidate_next_token or candidate_sequence")


@dataclass
class HaltCoT(HaltCoTConfig, ProbeMethod):
    @property
    def spec(self) -> MethodSpec:
        score_capability = ("selected_token_scores" if self.scoring == "candidate_next_token"
                            else "sequence_scoring")
        return MethodSpec(
            "halt_cot", variant="halt_cot_qwen_common_v1" if self.scoring == "candidate_next_token"
            else "halt_cot_sequence_common_v1",
            requirements=Requirements(frozenset({"visible_reasoning", "token_stream",
                "step_boundaries", "prefix_branch", "exact_prefix_resume", score_capability})),
            boundary=BoundaryPolicy("newline"), finalization_recipe="halt_default_v1",
            can_return_answer=True, signal_operations=(self.scoring,),
        )

    @classmethod
    def from_config(cls, config: HaltCoTConfig) -> HaltCoT:
        return cls(**asdict(config))

    def reset(self, context: RunContext) -> None:
        super().reset(context)
        self._streak = 0
        candidates = context.task_visible.get("candidates", ())
        self._candidates = tuple(candidates)
        if len(self._candidates) < 2 or len(set(self._candidates)) != len(self._candidates):
            raise ConfigurationError("HALT-CoT requires at least two distinct inference-visible candidates")

    def validate_task(self, task: Task) -> None:
        if len(task.candidates) < 2 or len(set(task.candidates)) != len(task.candidates):
            raise ConfigurationError("HALT-CoT requires at least two distinct inference-visible candidates")

    def on_boundary(self, event: Observation) -> None:
        self.request(ScoreCandidates(
            request_id=self.request_id(), prefix=event.prefix, candidates=self._candidates,
            scoring=self.scoring, recipe="halt_cot_qwen_common_v1",
            max_work=WorkLimit(scored_tokens=len(self._candidates)
                               if self.scoring == "candidate_next_token" else None),
        ))

    def on_probe(self, result: ProbeResult) -> None:
        scores = result.value
        if not isinstance(scores, CandidateScores):
            self.invalid_probe()
            return
        scored_prefix = scores.frame.prefix
        same_origin = (scored_prefix == result.prefix or (
            scored_prefix.trajectory_id == result.request_id
            and scored_prefix.run_id == result.prefix.run_id
            and scored_prefix.model_revision == result.prefix.model_revision))
        if (not same_origin or scores.candidates != self._candidates or scores.scoring != self.scoring
                or len(scores.log_scores) != len(self._candidates)
                or scores.frame.processing != "raw"):
            self.invalid_probe()
            return
        try:
            probabilities = softmax(scores.log_scores)
            value = entropy(probabilities, self.entropy_unit)
        except ValueError:
            self.invalid_probe()
            return
        self._streak = self._streak + 1 if value < self.threshold else 0
        if self._streak >= self.consecutive:
            selected = max(range(len(probabilities)), key=probabilities.__getitem__)
            self._decision = ReturnAnswer(
                self._candidates[selected], result.request_id, "candidate_entropy",
                {"candidate_entropy": value, "entropy_unit": self.entropy_unit,
                 "streak": self._streak, "candidate_probabilities": probabilities,
                 "scope": "conditional_on_candidate_set"},
            )

    def invalid_probe(self) -> None:
        self._streak = 0
