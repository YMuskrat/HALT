"""Stable method identifiers shared by command-line selection and evaluation."""
from __future__ import annotations

from typing import Any


def method_id(entry: dict[str, Any]) -> str:
    if "id" in entry:
        return str(entry["id"])
    name = str(entry["name"])
    if name == "fixed_reasoning_budget":
        from halt.methods.baselines import FixedReasoningBudget
        tokens = entry.get("parameters", {}).get("tokens", FixedReasoningBudget().tokens)
        return f"{name}_{tokens}"
    return name
