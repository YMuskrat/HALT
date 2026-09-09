"""Qwen3 thinking profile, audited against a pinned tokenizer and model card."""

from dataclasses import dataclass
from typing import Any

QWEN3_MODEL_ID = "Qwen/Qwen3-0.6B"
QWEN3_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FINALIZATION_SUFFIXES = {
    "halt_default_v1": "\n\nFinal answer:",
    "halt_cot_qwen_common_v1": "\nTherefore, the final answer is",
    "answer_convergence_qwen_common_v1": "\n\\boxed",
    "answer_convergence_common_v1": "\n\\boxed",
    "answer_consistency_sentence_common_v1": "\n\\boxed{",
    "refrain_boxed_v1": "\n\nFinal Answer: \\boxed{",
}


@dataclass(frozen=True)
class Qwen3ThinkingProfile:
    """Use the native chat template and explicitly close thinking once.

    Delimiter IDs are validated through the tokenizer; they are never guessed
    by the generation runtime. Other Qwen3 sizes require an explicit revision.
    """

    name: str = "qwen3_thinking"
    version: str = "1"
    reasoning_start: str = "<think>"
    reasoning_end: str = "</think>"
    end_token_id: int = 151668

    def validate(self, tokenizer: Any, model_config: Any) -> None:
        if getattr(model_config, "model_type", None) != "qwen3":
            raise ValueError("qwen3_thinking requires a decoder-only Qwen3 model")
        if getattr(model_config, "is_encoder_decoder", False):
            raise ValueError("HALT currently supports decoder-only models")
        if tokenizer.encode(self.reasoning_end, add_special_tokens=False) != [
            self.end_token_id
        ]:
            raise ValueError("Qwen3 tokenizer must encode </think> as token 151668")
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("Qwen3 requires its documented native chat template")

    def render(self, tokenizer: Any, prompt: str) -> list[int]:
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )

    def transition(self, tokenizer: Any, ids: list[int], recipe: str) -> list[int]:
        """Return appended input tokens, which are charged as inserted input.

        Natural closure needs no injected delimiter. HALT's shared recipe
        preserves the original reasoning prefix and adds a final-answer cue.
        Paper-specific cues are explicit recipes, not implicit fallbacks.
        """
        if recipe not in FINALIZATION_SUFFIXES:
            raise ValueError(f"Unsupported Qwen3 finalization recipe: {recipe!r}")
        closure = [] if self.end_token_id in ids else [self.end_token_id]
        return closure + list(
            tokenizer.encode(FINALIZATION_SUFFIXES[recipe], add_special_tokens=False)
        )

    def candidate_tokens(self, tokenizer: Any, candidate: str) -> list[int]:
        """Validate a continuation after an explicit stable answer cue.

        Appending a candidate must leave the cue's encoding unchanged. The
        leading space belongs to the candidate, avoiding a trailing-space
        token that would be replaced by contextual BPE tokenization.
        """
        cue = FINALIZATION_SUFFIXES["halt_cot_qwen_common_v1"]
        before = list(tokenizer.encode(cue, add_special_tokens=False))
        after = list(tokenizer.encode(cue + " " + candidate, add_special_tokens=False))
        if after[: len(before)] != before:
            raise ValueError(f"Candidate {candidate!r} changes the answer prefix tokenization")
        return after[len(before) :]
