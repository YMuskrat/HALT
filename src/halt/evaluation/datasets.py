from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from halt.errors import ConfigurationError
from halt.tasks import MultipleChoiceTask, NumericTask, Task


@dataclass(frozen=True)
class EvaluationItem:
    item_id: str
    task: Task
    reference: str = field(repr=False)

    def correct(self, answer: str | None) -> bool:
        return answer is not None and self.task.normalize(answer) == self.reference


@dataclass(frozen=True)
class Dataset:
    items: tuple[EvaluationItem, ...]
    adapter: str
    split: str
    revision: str
    content_sha256: str

    def identity(self) -> dict[str, Any]:
        return {"adapter": self.adapter, "split": self.split, "revision": self.revision,
                "content_sha256": self.content_sha256, "ordered_item_ids": [x.item_id for x in self.items]}


def load_dataset(path: str | Path, *, adapter: str = "mcq_jsonl", split: str = "development",
                 revision: str = "local", limit: int | None = None) -> Dataset:
    if adapter not in {"mcq_jsonl", "numeric_jsonl"}:
        raise ConfigurationError("supported adapters are mcq_jsonl and numeric_jsonl")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ConfigurationError("dataset limit must be a positive integer")
    raw = Path(path).read_bytes()
    items = []
    ids: set[str] = set()
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            allowed = {"id", "question", "answer", "choices"} if adapter == "mcq_jsonl" else {"id", "question", "answer"}
            if not isinstance(row, dict) or set(row) - allowed:
                raise ValueError("unknown fields or non-object row")
            item_id = str(row["id"])
            if not item_id or item_id in ids:
                raise ValueError("item IDs must be nonempty and unique")
            if not isinstance(row["question"], str) or not row["question"].strip():
                raise ValueError("question must be nonempty text")
            task: Task = MultipleChoiceTask(row["question"], row["choices"]) if adapter == "mcq_jsonl" else NumericTask(row["question"])
            reference = task.normalize(str(row["answer"]))
            if reference is None:
                raise ValueError("reference does not satisfy the task's answer contract")
        except (KeyError, ValueError, TypeError) as exc:
            raise ConfigurationError(f"invalid dataset row {number}: {exc}") from exc
        ids.add(item_id)
        items.append(EvaluationItem(item_id, task, reference))
    if not items:
        raise ConfigurationError("dataset must contain at least one item")
    return Dataset(tuple(items[:limit]), adapter, split, revision, hashlib.sha256(raw).hexdigest())
