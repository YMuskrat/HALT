"""Local dataset import and validation without loading models or exposing gold labels."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
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
    import_metadata: Mapping[str, Any] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        identity: dict[str, Any] = {"adapter": self.adapter, "split": self.split, "revision": self.revision,
                    "content_sha256": self.content_sha256,
                    "ordered_item_ids": [x.item_id for x in self.items]}
        if self.import_metadata:
            identity["import"] = dict(self.import_metadata)
        return identity


@dataclass(frozen=True)
class DatasetIssue:
    """A physical source line (CSV records may span lines), or a file-level error."""

    row: int | None
    message: str


@dataclass(frozen=True)
class DatasetValidationReport:
    path: str
    format: str | None
    adapter: str | None
    checked_rows: int
    valid_rows: int
    columns: Mapping[str, Any]
    ignored_columns: tuple[str, ...]
    errors: tuple[DatasetIssue, ...]
    dataset: Dataset | None = field(default=None, repr=False)

    @property
    def valid(self) -> bool:
        return self.dataset is not None and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "path": self.path, "format": self.format,
                "adapter": self.adapter, "checked_rows": self.checked_rows,
                "valid_rows": self.valid_rows,
                "selected_rows": len(self.dataset.items) if self.dataset is not None else 0,
                "columns": dict(self.columns), "ignored_columns": list(self.ignored_columns),
                "errors": [{"row": error.row, "message": error.message} for error in self.errors]}


_ALIASES = {
    "question": {"question", "prompt", "query", "problem"},
    "answer": {"answer", "answer_key", "answerkey", "correct_answer", "label", "target", "gold"},
    "id": {"id", "question_id", "item_id", "example_id", "uid"},
    "choices": {"choices", "options"},
}
_LEGACY = {"mcq_jsonl": "mcq", "numeric_jsonl": "numeric"}


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _source_rows(text: str, file_format: str, errors: list[DatasetIssue]) -> tuple[list[tuple[int, dict[str, Any]]], list[str], int]:
    rows: list[tuple[int, dict[str, Any]]] = []
    fields: dict[str, None] = {}
    checked = 0
    if file_format == "jsonl":
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            checked += 1
            try:
                row = json.loads(line, object_pairs_hook=_json_object, parse_float=Decimal)
                if not isinstance(row, dict):
                    raise ValueError("expected one JSON object per line")
                fields.update(dict.fromkeys(row))
                rows.append((number, row))
            except ValueError as exc:
                errors.append(DatasetIssue(number, f"invalid JSONL: {exc}"))
    else:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        try:
            header = next(reader, [])
            if not header or any(not name.strip() for name in header):
                raise ValueError("CSV needs a header with nonempty column names")
            if len(set(header)) != len(header):
                raise ValueError("CSV header contains duplicate column names")
            fields.update(dict.fromkeys(header))
            while True:
                number = reader.line_num + 1
                values = next(reader, None)
                if values is None:
                    break
                if not values:
                    continue
                checked += 1
                if len(values) != len(header):
                    errors.append(DatasetIssue(number, f"CSV has {len(values)} fields; expected {len(header)}. Quote commas inside values."))
                    continue
                rows.append((number, dict(zip(header, values, strict=True))))
        except (csv.Error, ValueError) as exc:
            errors.append(DatasetIssue(max(reader.line_num, 1), f"invalid CSV: {exc}"))
    return rows, list(fields), checked


def _column(fields: list[str], name: str, explicit: str | None, *, required: bool) -> str | None:
    if explicit is not None:
        if not isinstance(explicit, str) or explicit not in fields:
            raise ValueError(f"{name} column {explicit!r} does not exist")
        return explicit
    matches = [key for key in fields if re.sub(r"[ -]+", "_", key.strip().casefold()) in _ALIASES[name]]
    if len(matches) > 1:
        raise ValueError(f"ambiguous {name} columns {matches}; set {name}_column explicitly")
    if not matches and required:
        raise ValueError(f"no {name} column found; set {name}_column explicitly")
    return matches[0] if matches else None


def _columns(fields: list[str], adapter: str, question: str | None, answer: str | None,
             item_id: str | None, choices: str | None, options: Mapping[str, str] | None) -> dict[str, Any]:
    if adapter in _LEGACY:
        if any(value is not None for value in (question, answer, item_id, choices, options)):
            raise ValueError("column mappings require adapter auto, mcq, or numeric; legacy JSONL adapters use fixed fields")
        return {"question": "question", "answer": "answer", "id": "id",
                "choices": "choices" if adapter == "mcq_jsonl" else None, "choice_columns": {}}
    if choices is not None and options is not None:
        raise ValueError("use choices_column or choice_columns, not both")
    resolved: dict[str, Any] = {
        "question": _column(fields, "question", question, required=True),
        "answer": _column(fields, "answer", answer, required=True),
        "id": _column(fields, "id", item_id, required=False),
        "choices": None, "choice_columns": {},
    }
    if adapter == "numeric":
        if choices is not None or options is not None:
            raise ValueError("numeric datasets do not use choices mappings")
        return resolved
    if options is not None:
        if not isinstance(options, Mapping) or not options:
            raise ValueError("choice_columns must map nonempty labels to column names")
        for label, source in options.items():
            if not isinstance(label, str) or not label.strip() or source not in fields:
                raise ValueError(f"invalid choice column mapping {label!r}={source!r}")
        if len(set(options.values())) != len(options):
            raise ValueError("each choice label must map to a different column")
        resolved["choice_columns"] = dict(options)
    else:
        resolved["choices"] = _column(fields, "choices", choices, required=False)
        if choices is None:
            for key in fields:
                if key in (resolved["question"], resolved["answer"], resolved["id"]):
                    continue
                match = re.fullmatch(r"(?:(?:choice|option)[ _-])?([a-z])", key.strip(), re.I)
                if match:
                    label = match[1].upper()
                    if label in resolved["choice_columns"]:
                        raise ValueError(f"ambiguous option columns for label {label}; set choice_columns explicitly")
                    resolved["choice_columns"][label] = key
            if resolved["choices"] and resolved["choice_columns"]:
                raise ValueError("both a choices column and option columns were found; set choices_column or choice_columns explicitly")
            resolved["choice_columns"] = dict(sorted(resolved["choice_columns"].items()))
    return resolved


def _required(row: Mapping[str, Any], column: str, name: str) -> Any:
    if column not in row or row[column] is None or (isinstance(row[column], str) and not row[column].strip()):
        raise ValueError(f"missing {name} in column {column!r}")
    return row[column]


def _choices(row: Mapping[str, Any], columns: Mapping[str, Any], *, legacy: bool) -> dict[str, str]:
    if columns["choice_columns"]:
        values = {label: _required(row, column, f"choice {label}")
                  for label, column in columns["choice_columns"].items()}
    elif columns["choices"]:
        values = _required(row, columns["choices"], "choices")
        if isinstance(values, str) and not legacy:
            try:
                values = json.loads(values, object_pairs_hook=_json_object)
            except ValueError as exc:
                raise ValueError("choices must contain a valid JSON object or list") from exc
        if isinstance(values, list) and not legacy:
            if len(values) > 26:
                raise ValueError("choice lists support up to 26 entries; use a JSON object for explicit labels")
            values = {chr(65 + i): value for i, value in enumerate(values)}
    else:
        raise ValueError("MCQ rows need a choices JSON object/list or option columns; set choices_column or choice_columns")
    if not isinstance(values, dict) or len(values) < (1 if legacy else 2):
        raise ValueError("choices must be an object with " + ("at least one choice" if legacy else "at least two choices"))
    if any(not isinstance(label, str) or not label.strip() for label in values):
        raise ValueError("choice labels must be nonempty strings")
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        raise ValueError("choice values must be nonempty text")
    return values


def validate_dataset(path: str | Path, *, adapter: str = "auto", split: str = "development",
                     revision: str = "local", limit: int | None = None,
                     question_column: str | None = None, answer_column: str | None = None,
                     id_column: str | None = None, choices_column: str | None = None,
                     choice_columns: Mapping[str, str] | None = None) -> DatasetValidationReport:
    """Check every source row before inference; limit selects only after validation.

    ``auto``, ``mcq``, and ``numeric`` recognize common columns in CSV/JSONL and
    generate deterministic row IDs if no ID column exists. Legacy ``*_jsonl``
    adapters retain their fixed schema. Errors never return a partial dataset.
    """
    errors: list[DatasetIssue] = []
    columns: dict[str, Any] = {}
    ignored: tuple[str, ...] = ()
    items: list[EvaluationItem] = []
    file_format: str | None = None
    resolved_adapter: str | None = None
    checked = 0
    dataset = None
    try:
        if adapter not in {*_LEGACY, "auto", "mcq", "numeric"}:
            raise ValueError("supported adapters are auto, mcq, numeric, mcq_jsonl, and numeric_jsonl")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("dataset limit must be a positive integer")
        suffix = Path(path).suffix.casefold()
        file_format = "jsonl" if adapter in _LEGACY or suffix in {".jsonl", ".ndjson"} else "csv" if suffix == ".csv" else None
        if file_format is None:
            raise ValueError("dataset file must end in .csv, .jsonl, or .ndjson")
        raw = Path(path).read_bytes()
        rows, fields, checked = _source_rows(raw.decode("utf-8-sig"), file_format, errors)
        if not rows:
            raise ValueError("dataset must contain at least one valid data row")
        columns = _columns(fields, adapter, question_column, answer_column, id_column, choices_column, choice_columns)
        used = {value for key, value in columns.items() if key != "choice_columns" and value is not None}
        sources = [value for key, value in columns.items() if key != "choice_columns" and value is not None]
        sources.extend(columns["choice_columns"].values())
        if len(set(sources)) != len(sources):
            raise ValueError("each dataset field must use a distinct source column; answer columns cannot also be inference inputs")
        used.update(columns["choice_columns"].values())
        ignored = tuple(key for key in fields if key not in used)
        ids: set[str] = set()
        kind = _LEGACY.get(adapter, adapter)
        if kind == "auto":
            kind = "mcq" if columns["choices"] or columns["choice_columns"] else "numeric"
        resolved_adapter = f"{kind}_{file_format}"
        for number, row in rows:
            try:
                if adapter in _LEGACY and set(row) - used:
                    raise ValueError("unknown fields in legacy JSONL schema; use adapter auto for column detection")
                raw_id = _required(row, columns["id"], "id") if columns["id"] else f"row-{number}"
                if isinstance(raw_id, bool) or not isinstance(raw_id, str | int):
                    raise ValueError("id must be nonempty text or an integer")
                item_id = str(raw_id).strip()
                if item_id in ids:
                    raise ValueError(f"duplicate item id {item_id!r}; IDs must be unique")
                ids.add(item_id)
                question = _required(row, columns["question"], "question")
                if not isinstance(question, str):
                    raise ValueError("question must be nonempty text")
                answer = _required(row, columns["answer"], "answer")
                if isinstance(answer, bool) or not isinstance(answer, str | int | float | Decimal):
                    raise ValueError("answer must be a choice label or a finite numeric value")
                task: Task = MultipleChoiceTask(question, _choices(row, columns, legacy=adapter in _LEGACY)) if kind == "mcq" else NumericTask(question)
                reference = task.normalize(str(answer))
                if reference is None:
                    if kind == "numeric" and adapter == "auto":
                        raise ValueError("cannot infer task: no choices were found and answer is not numeric; map choices for MCQ or use a numeric answer (free-text grading is unsupported)")
                    raise ValueError("answer does not match a choice label" if kind == "mcq" else "answer is not a valid finite numeric value")
                items.append(EvaluationItem(item_id, task, reference))
            except (KeyError, ValueError, TypeError, ConfigurationError) as exc:
                errors.append(DatasetIssue(number, str(exc)))
        if not errors:
            metadata = {} if adapter in _LEGACY else {"version": "local_import_v1", "requested_adapter": adapter,
                "format": file_format, "columns": columns, "ignored_columns": list(ignored),
                "generated_ids": columns["id"] is None}
            dataset = Dataset(tuple(items[:limit]), resolved_adapter, split, revision,
                              hashlib.sha256(raw).hexdigest(), metadata)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        errors.append(DatasetIssue(None, str(exc)))
    errors.sort(key=lambda error: (error.row is None, error.row or 0))
    return DatasetValidationReport(str(path), file_format, resolved_adapter, checked,
                                   len(items), columns, ignored, tuple(errors), dataset)


def load_dataset(path: str | Path, *, adapter: str = "mcq_jsonl", split: str = "development",
                 revision: str = "local", limit: int | None = None,
                 question_column: str | None = None, answer_column: str | None = None,
                 id_column: str | None = None, choices_column: str | None = None,
                 choice_columns: Mapping[str, str] | None = None) -> Dataset:
    report = validate_dataset(path, adapter=adapter, split=split, revision=revision, limit=limit,
                              question_column=question_column, answer_column=answer_column,
                              id_column=id_column, choices_column=choices_column,
                              choice_columns=choice_columns)
    if report.dataset is None:
        details = "; ".join((f"row {error.row}: " if error.row is not None else "") + error.message
                            for error in report.errors[:10])
        remaining = len(report.errors) - 10
        if remaining > 0:
            details += f"; and {remaining} more errors (use dataset validation for the full list)"
        raise ConfigurationError(f"invalid dataset: {details}")
    return report.dataset
