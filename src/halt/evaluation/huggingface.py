"""Optional ARC loader. References remain in EvaluationItem, outside inference tasks."""
from __future__ import annotations

import re
from typing import Any

from halt.errors import ConfigurationError
from halt.evaluation.datasets import Dataset, EvaluationItem
from halt.tasks import MultipleChoiceTask
from halt.types import stable_hash

ARC_DATASET_ID = "allenai/ai2_arc"
ARC_AUDITED_REVISION = "210d026faf9955653af8916fad021475a3f00453"


def arc_rows(rows: Any, *, revision: str, subset: str, split: str,
             limit: int | None = None) -> Dataset:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ConfigurationError("ARC dataset revision must be an immutable commit")
    if subset not in {"ARC-Easy", "ARC-Challenge"} or split not in {"train", "validation", "test"}:
        raise ConfigurationError("ARC subset/split must identify a canonical dataset partition")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ConfigurationError("limit must be a positive integer")
    items: list[EvaluationItem] = []
    identities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        if len(labels) != len(texts) or len(set(labels)) != len(labels):
            raise ConfigurationError("ARC choice labels must be distinct and aligned with option text")
        # Candidates come exclusively from published visible labels, never answerKey.
        task = MultipleChoiceTask(row["question"], dict(zip(labels, texts, strict=True)))
        reference = task.normalize(row["answerKey"])
        item_id = str(row["id"])
        if reference is None or item_id in seen:
            raise ConfigurationError("invalid ARC reference or duplicate item ID")
        items.append(EvaluationItem(item_id, task, reference))
        identities.append({"id": item_id, "task": task.visible(), "reference": reference})
        seen.add(item_id)
    if not items:
        raise ConfigurationError("ARC selection must contain at least one item")
    return Dataset(tuple(items), "arc_hf_v1", split,
                   f"{ARC_DATASET_ID}@{revision}/{subset}", stable_hash(identities))


def load_arc(*, revision: str = ARC_AUDITED_REVISION, subset: str = "ARC-Easy",
             split: str = "validation", limit: int | None = None,
             cache_dir: str | None = None) -> Dataset:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ConfigurationError("ARC dataset revision must be an immutable commit")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install halt-reasoning[evaluation] for the optional ARC loader") from exc
    rows = load_dataset(ARC_DATASET_ID, subset, split=split, revision=revision,
                        cache_dir=cache_dir)
    return arc_rows(rows, revision=revision, subset=subset, split=split, limit=limit)
