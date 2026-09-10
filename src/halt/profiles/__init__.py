"""Lightweight, versioned model-family behavior (no model imports)."""

from .qwen3 import QWEN3_MODEL_ID, QWEN3_REVISION, Qwen3ThinkingProfile
from .recipes import AnswerRecipe

__all__ = ["AnswerRecipe", "QWEN3_MODEL_ID", "QWEN3_REVISION", "Qwen3ThinkingProfile"]
