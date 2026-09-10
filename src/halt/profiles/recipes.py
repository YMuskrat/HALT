"""Explicit answer-transition behavior, independent of method names."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from halt.errors import ConfigurationError


@dataclass(frozen=True)
class AnswerRecipe:
    suffix: str = "\n\nFinal answer:"
    close_reasoning: bool = True
    answer_prefix: str = ""
    remove_suffix: str = ""
    stop_at_reasoning_end: bool = False
    score_generated_tokens: bool = False

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            expected = str if field.name in {"suffix", "answer_prefix", "remove_suffix"} else bool
            if type(value) is not expected:
                raise ConfigurationError(f"answer recipe {field.name} must be {expected.__name__}")
        if self.stop_at_reasoning_end and self.close_reasoning:
            raise ConfigurationError("a reasoning-end trial must leave reasoning open")

    def reconstruct_answer(self, text: str) -> str:
        if self.answer_prefix.endswith("{"):
            return self.answer_prefix[:-1] + text if text.startswith("{") else self.answer_prefix + text
        return self.answer_prefix + text if self.answer_prefix and text.startswith("{") else text


DEFAULT_RECIPES = {
    "halt_default_v1": AnswerRecipe(),
    "candidate_scoring_v1": AnswerRecipe(suffix="\nTherefore, the final answer is"),
    "boxed_answer_v1": AnswerRecipe(suffix="\n\\boxed", answer_prefix="\\boxed"),
    "boxed_argument_v1": AnswerRecipe(suffix="\n\\boxed{", answer_prefix="\\boxed{"),
    # Retain pinned recipe names as data aliases for recorded experiments.
    "halt_cot_qwen_common_v1": AnswerRecipe(suffix="\nTherefore, the final answer is"),
    "answer_convergence_qwen_common_v1": AnswerRecipe(suffix="\n\\boxed", answer_prefix="\\boxed"),
    "answer_convergence_common_v1": AnswerRecipe(suffix="\n\\boxed", answer_prefix="\\boxed"),
    "answer_consistency_sentence_common_v1": AnswerRecipe(suffix="\n\\boxed{", answer_prefix="\\boxed{"),
    "refrain_boxed_v1": AnswerRecipe(suffix="\n\nFinal Answer: \\boxed{", answer_prefix="\\boxed{"),
    "deer_qwen3_greedy_v1": AnswerRecipe(suffix="\n**Final Answer**\n\\boxed", close_reasoning=False,
        answer_prefix="\\boxed", remove_suffix="Wait", stop_at_reasoning_end=True, score_generated_tokens=True),
}


def recipe_catalog(custom: Mapping[str, Any] | None = None) -> dict[str, AnswerRecipe]:
    result = dict(DEFAULT_RECIPES)
    for name, value in (custom or {}).items():
        if not isinstance(name, str) or not name:
            raise ConfigurationError("answer recipe names must be nonempty strings")
        if name in result:
            raise ConfigurationError(f"answer recipe {name!r} already exists; use a new versioned name")
        try:
            result[name] = value if isinstance(value, AnswerRecipe) else AnswerRecipe(**value)
        except TypeError as exc:
            raise ConfigurationError(f"invalid answer recipe {name!r}: {exc}") from exc
    return result
