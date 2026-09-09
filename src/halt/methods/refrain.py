"""REFRAIN's two-stage detector; adaptive threshold selection lives in halt.session."""
from __future__ import annotations

import math
from dataclasses import dataclass

from halt.errors import ConfigurationError
from halt.methods._research import ProbeMethod
from halt.signals import cosine_similarity
from halt.types import (
    BoundaryPolicy,
    EmbedSteps,
    Finalize,
    MethodSpec,
    Observation,
    ProbeResult,
    Requirements,
    RunContext,
    StepEmbeddings,
)

# Appendix A's four reflection categories, excluding the separate new-category ablation.
REFLECTION_CUES = (
    "wait", "let me check", "hold on", "have made a mistake", "let me double check",
    "wait a moment", "is that correct", "let me re-read", "alternatively", "let me try",
    "think of it as", "let me consider", "what if we try", "let’s think from a different angle",
    "an alternative method would be", "instead of doing that", "not sure", "looks like",
    "that seems", "hmm", "perhaps", "maybe i", "i’m not certain", "it seems", "i suspect",
    "my guess is", "earlier we saw", "from before", "so now we have", "recall that",
    "let me go back", "as we established previously", "based on our previous result",
    "remember that we found", "the value from step",
)


@dataclass
class Refrain(ProbeMethod):
    threshold: float = 0.7
    adaptive: bool = False
    answer_cues: tuple[str, ...] = ("answer is", "answer should be")

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ConfigurationError("REFRAIN threshold must be finite and in [0, 1]")
        if type(self.adaptive) is not bool:
            raise ConfigurationError("adaptive must be boolean")
        self.answer_cues = tuple(self.answer_cues)
        if not self.answer_cues or any(not c.strip() for c in self.answer_cues):
            raise ConfigurationError("answer_cues must contain nonempty provisional-answer phrases")

    @property
    def spec(self) -> MethodSpec:
        capabilities = {"visible_reasoning", "token_stream", "step_boundaries", "embeddings",
                        "answer_transition"}
        if self.adaptive:
            capabilities.add("sequence_scoring")
        return MethodSpec("refrain", variant="refrain_adaptive_common_v1" if self.adaptive
            else "refrain_fixed_common_v1", requirements=Requirements(frozenset(capabilities)),
            boundary=BoundaryPolicy("blank_line"), finalization_recipe="refrain_boxed_v1")

    def reset(self, context: RunContext) -> None:
        super().reset(context)
        if self.adaptive and context.session_reference is None:
            raise ConfigurationError("adaptive REFRAIN requires an explicit RefrainSession")
        self._steps: list[str] = []
        self._had_prior_answer = False
        self._expected_steps = 0

    def on_boundary(self, event: Observation) -> None:
        step = event.payload.text
        folded = step.casefold()
        has_reflection = any(cue in folded for cue in REFLECTION_CUES)
        # Read the previous history gate BEFORE admitting this step's answer cue.
        eligible = bool(self._steps) and self._had_prior_answer and has_reflection
        self._steps.append(step)
        self._had_prior_answer |= any(cue.casefold() in folded for cue in self.answer_cues)
        if eligible:
            self._expected_steps = len(self._steps)
            self.request(EmbedSteps(request_id=self.request_id(), prefix=event.prefix,
                                    steps=tuple(self._steps)))

    def on_probe(self, result: ProbeResult) -> None:
        value = result.value
        if not isinstance(value, StepEmbeddings) or len(value.vectors) != self._expected_steps:
            return
        if len(value.vectors) < 2:
            return
        try:
            similarity = max(cosine_similarity(value.vectors[-1], previous)
                             for previous in value.vectors[:-1])
        except ValueError:
            return
        if similarity >= self.threshold:
            self._decision = Finalize("reflective_redundancy", recipe="refrain_boxed_v1",
                diagnostics={"maximum_previous_step_cosine": similarity,
                    "threshold": self.threshold, "prior_steps": len(value.vectors)-1,
                    "encoder": value.encoder, "encoder_revision": value.revision,
                    "adaptive": self.adaptive, "history_excludes_current_step": True})
