"""Inference-only task contracts. Gold labels live exclusively in evaluation."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from halt.errors import ConfigurationError


class Task(Protocol):
    def render(self) -> str: ...
    def normalize(self, text: str) -> str | None: ...
    def visible(self) -> Mapping[str, Any]: ...
    @property
    def candidates(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class MultipleChoiceTask:
    question: str
    choices: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.question or not self.choices:
            raise ConfigurationError("question and choices must be nonempty")
        if any(not isinstance(k, str) or not k.strip() for k in self.choices):
            raise ConfigurationError("choice labels must be nonempty strings")

    @property
    def candidates(self) -> tuple[str, ...]:
        return tuple(self.choices)

    def render(self) -> str:
        options = "\n".join(f"{k}. {v}" for k, v in self.choices.items())
        return f"{self.question}\n{options}\nGive your final answer as one choice label."

    def normalize(self, text: str) -> str | None:
        value = text.strip()
        if value in self.choices:
            return value
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", value)
        if boxed:
            return boxed[-1].strip() if boxed[-1].strip() in self.choices else None
        matches = re.findall(r"(?:final answer|answer)\s*(?:is|:)\s*([\w.-]+)", value, re.I)
        if matches:
            label = matches[-1].rstrip(".")
            return label if label in self.choices else None
        # Accept only a standalone label with formatting, not a choice mentioned in reasoning.
        label = value.strip(" *().\n")
        if label in self.choices:
            return label
        # A model may repeat the exact inference-visible option, e.g. '**B. 4**'.
        # Require the complete supplied option text, so a wrong/ambiguous body fails.
        formatted = value.strip(" *\n")
        for key, option in self.choices.items():
            if formatted in {f"{key}. {option}", f"{key}) {option}", f"{key}: {option}"}:
                return key
        return None

    def visible(self) -> Mapping[str, Any]:
        return {"adapter": "mcq_v1", "question": self.question, "choices": dict(self.choices),
                "candidates": self.candidates}


@dataclass(frozen=True)
class NumericTask:
    question: str

    @property
    def candidates(self) -> tuple[str, ...]:
        return ()

    def render(self) -> str:
        return f"{self.question}\nGive your final numeric answer in \\boxed{{}}."

    def normalize(self, text: str) -> str | None:
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
        value = (boxed[-1] if boxed else text).strip()
        # Thousands separators are validated; arbitrary punctuation is not discarded.
        if not re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?(?:[eE][+-]?\d+)?", value):
            return None
        try:
            number = Decimal(value.replace(",", ""))
            return format(number.normalize(), "f") if number.is_finite() else None
        except InvalidOperation:
            return None

    def visible(self) -> Mapping[str, Any]:
        return {"adapter": "numeric_v1", "question": self.question, "candidates": ()}
