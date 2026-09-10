import csv
import hashlib
import json

import pytest

from halt.errors import ConfigurationError
from halt.evaluation.datasets import load_dataset, validate_dataset
from halt.tasks import MultipleChoiceTask, NumericTask


def write_jsonl(tmp_path, rows, *, name="questions.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_csv_detection_order_hash_and_gold_separation(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text("Question ID,Prompt,option_b,option_a,AnswerKey,category\nq2,Pick a number,4,3,B,math\nq1,Pick again,6,5,A,math\n", encoding="utf-8-sig")
    report = validate_dataset(path, limit=1, split="test", revision="sample-v1")
    assert report.valid, report.to_dict()
    dataset = report.dataset
    assert dataset.adapter == "mcq_csv"
    assert dataset.split == "test" and dataset.revision == "sample-v1"
    assert dataset.content_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report.checked_rows == report.valid_rows == 2
    assert report.to_dict()["selected_rows"] == 1
    assert dataset.identity()["ordered_item_ids"] == ["q2"]
    assert report.ignored_columns == ("category",)
    item = dataset.items[0]
    assert item.task.candidates == ("A", "B")
    assert item.correct("B") and not item.correct("A")
    assert item.task.visible() == {"adapter": "mcq_v1", "question": "Pick a number",
                                   "choices": {"A": "3", "B": "4"}, "candidates": ("A", "B")}
    assert "reference" not in vars(item.task)
    assert "category" in dataset.identity()["import"]["ignored_columns"]


def test_jsonl_options_list_and_generated_ids(tmp_path):
    path = write_jsonl(tmp_path, [{"query": "Choose", "options": ["red", "blue"], "label": "B"}])
    report = validate_dataset(path)
    assert report.valid, report.to_dict()
    item = report.dataset.items[0]
    assert item.item_id == "row-1"
    assert item.task.candidates == ("A", "B")
    assert item.task.choices == {"A": "red", "B": "blue"}
    assert load_dataset(path, adapter="auto").identity() == report.dataset.identity()


def test_csv_json_choices_and_quoted_multiline_question(tmp_path):
    path = tmp_path / "choices.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question", "choices", "answer"])
        writer.writerow(["Line one, with comma\nLine two", json.dumps({"yes": "Yes", "no": "No"}), "yes"])
    dataset = load_dataset(path, adapter="mcq")
    assert dataset.items[0].item_id == "row-2"
    assert "\nLine two" in dataset.items[0].task.question
    assert dataset.items[0].correct("yes")


def test_explicit_mappings_with_nonstandard_choices(tmp_path):
    path = tmp_path / "mapped.csv"
    path.write_text("key,body,expected,left,right,notes\n8,Pick,left,First,Second,unused\n", encoding="utf-8")
    kwargs = {"adapter": "mcq", "question_column": "body", "answer_column": "expected",
              "id_column": "key", "choice_columns": {"left": "left", "right": "right"}}
    report = validate_dataset(path, **kwargs)
    assert report.valid, report.to_dict()
    assert report.dataset.items[0].item_id == "8"
    assert report.dataset.items[0].task.candidates == ("left", "right")
    assert report.ignored_columns == ("notes",)
    assert report.dataset.identity()["import"]["columns"]["answer"] == "expected"


def test_explicit_choices_column_resolves_auto_detection_ambiguity(tmp_path):
    path = write_jsonl(tmp_path, [{"question": "Pick", "choices": ["one", "two"],
                                  "A": "unused", "B": "unused", "answer": "B"}])
    assert not validate_dataset(path).valid
    report = validate_dataset(path, choices_column="choices")
    assert report.valid, report.to_dict()
    assert report.ignored_columns == ("A", "B")


def test_numeric_auto_preserves_zero_negative_decimal_and_bom(tmp_path):
    path = tmp_path / "numbers.csv"
    path.write_text("problem,target\nZero?,0\nNegative?,-1.50\nThousands?,\"1,000\"\n", encoding="utf-8-sig")
    dataset = load_dataset(path, adapter="auto")
    assert dataset.adapter == "numeric_csv"
    assert all(isinstance(item.task, NumericTask) for item in dataset.items)
    assert [item.reference for item in dataset.items] == ["0", "-1.5", "1000"]
    assert all(item.task.candidates == () for item in dataset.items)


def test_json_numeric_reference_preserves_decimal_precision(tmp_path):
    path = tmp_path / "precision.jsonl"
    path.write_text('{"question":"Value?","answer":0.123456789123456789}\n', encoding="utf-8")
    item = load_dataset(path, adapter="auto").items[0]
    assert item.correct("0.123456789123456789")
    assert not item.correct("0.12345678912345678")


def test_legacy_adapters_remain_strict(tmp_path):
    row = {"id": "q", "question": "Pick", "choices": {"A": "yes", "B": "no"}, "answer": "A"}
    path = write_jsonl(tmp_path, [row])
    legacy = load_dataset(path)
    assert isinstance(legacy.items[0].task, MultipleChoiceTask)
    assert "import" not in legacy.identity()
    path = write_jsonl(tmp_path, [{**row, "category": "example"}])
    with pytest.raises(ConfigurationError, match="unknown fields"):
        load_dataset(path)
    assert load_dataset(path, adapter="auto").items[0].correct("A")
    with pytest.raises(ConfigurationError, match="column mappings require"):
        load_dataset(path, question_column="question")
    numeric = write_jsonl(tmp_path, [{"id": "n", "question": "Count?", "answer": 0}])
    assert load_dataset(numeric, adapter="numeric_jsonl").items[0].correct("0.00")


def test_collects_row_errors_and_never_returns_partial_dataset(tmp_path):
    path = write_jsonl(tmp_path, [
        {"id": "ok", "question": "Pick", "choices": ["one", "two"], "answer": "A"},
        {"id": "bad-answer", "question": "Pick", "choices": ["one", "two"], "answer": "C"},
        {"id": "missing-answer", "question": "Pick", "choices": ["one", "two"]},
        {"id": "ok", "question": "Pick", "choices": ["one", "two"], "answer": "A"},
        {"id": "", "question": "Pick", "choices": ["one", "two"], "answer": "A"},
    ])
    report = validate_dataset(path, limit=1)
    assert not report.valid and report.dataset is None
    assert report.checked_rows == 5 and report.valid_rows == 1
    assert [error.row for error in report.errors] == [2, 3, 4, 5]
    assert "choice label" in report.errors[0].message
    assert "missing answer" in report.errors[1].message
    assert "duplicate item id" in report.errors[2].message
    assert "missing id" in report.errors[3].message
    with pytest.raises(ConfigurationError, match="row 2"):
        load_dataset(path, adapter="auto", limit=1)


@pytest.mark.parametrize("content,match", [
    ('question,answer\n"unterminated,4\n', "invalid CSV"),
    ("question,answer\nCount?,4,extra\n", "expected 2"),
    ("question,answer\nCount?\n", "expected 2"),
    ("question,answer,answer\nCount?,4,4\n", "duplicate column"),
    ("question,,answer\nCount?,x,4\n", "nonempty column"),
    ("", "CSV needs a header"),
])
def test_malformed_csv_is_reported(tmp_path, content, match):
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    report = validate_dataset(path)
    assert not report.valid
    assert any(match in error.message for error in report.errors)


def test_jsonl_malformed_rows_and_duplicate_keys_are_collected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question":"Count?","answer":4}\n{bad}\n[]\n{"question":"Count?","answer":4,"answer":5}\n', encoding="utf-8")
    report = validate_dataset(path)
    assert not report.valid and report.dataset is None
    assert report.checked_rows == 4 and report.valid_rows == 1
    assert [error.row for error in report.errors] == [2, 3, 4]
    assert "duplicate JSON key" in report.errors[-1].message


@pytest.mark.parametrize("row,match", [
    ({"question": "Q", "answer": "not numeric"}, "cannot infer task"),
    ({"question": "Q", "prompt": "also Q", "answer": 1}, "ambiguous question"),
    ({"question": "Q", "answer": 1, "target": 1}, "ambiguous answer"),
    ({"question": "Q", "answer": "A", "A": "yes", "option_a": "yes", "B": "no"}, "ambiguous option"),
    ({"question": "Q", "answer": "A", "choices": ["one"]}, "at least two choices"),
    ({"question": "Q", "answer": "A", "choices": ["one", 2]}, "nonempty text"),
    ({"question": "Q", "answer": "A", "choices": {"": "one", "A": "two"}}, "nonempty strings"),
    ({"question": "Q", "answer": "A", "choices": "one,two"}, "valid JSON"),
    ({"question": "Q", "answer": None}, "missing answer"),
    ({"question": "Q", "answer": True}, "answer must be"),
    ({"question": "Q", "answer": float("nan")}, "cannot infer task"),
    ({"question": " ", "answer": 1}, "missing question"),
])
def test_unusable_or_ambiguous_rows_need_explicit_fix(tmp_path, row, match):
    report = validate_dataset(write_jsonl(tmp_path, [row]))
    assert not report.valid
    assert any(match in error.message for error in report.errors), report.to_dict()


def test_mapping_can_resolve_ambiguity_but_cannot_feed_gold_into_prompt(tmp_path):
    path = write_jsonl(tmp_path, [{"question": "Q", "prompt": "Prompt", "answer": 1}])
    report = validate_dataset(path, question_column="prompt")
    assert report.valid and report.dataset.items[0].task.question == "Prompt"
    bad = validate_dataset(path, question_column="answer", answer_column="answer")
    assert not bad.valid and "distinct source column" in bad.errors[0].message


@pytest.mark.parametrize("limit", [0, -1, True, 1.2])
def test_invalid_limit_reports_before_reading(tmp_path, limit):
    report = validate_dataset(tmp_path / "missing.csv", limit=limit)
    assert not report.valid and "positive integer" in report.errors[0].message


def test_file_errors_are_structured(tmp_path):
    assert not validate_dataset(tmp_path / "missing.csv").valid
    path = tmp_path / "not_utf8.csv"
    path.write_bytes(b"question,answer\n\xff,4")
    assert "decode" in validate_dataset(path).errors[0].message
    empty = write_jsonl(tmp_path, [])
    assert "at least one" in validate_dataset(empty).errors[0].message


def test_id_absence_is_different_from_partial_missing_ids(tmp_path):
    path = write_jsonl(tmp_path, [{"id": "one", "question": "Q", "answer": 1},
                                  {"question": "Q", "answer": 2}])
    report = validate_dataset(path)
    assert not report.valid and "missing id" in report.errors[0].message


def test_explicit_option_columns_take_priority_and_affect_identity(tmp_path):
    path = write_jsonl(tmp_path, [{"question": "Q", "answer": "A", "A": "first",
                                  "B": "second", "choices": ["other", "unused"]}])
    first = load_dataset(path, adapter="auto", choice_columns={"A": "A", "B": "B"})
    second = load_dataset(path, adapter="auto", choices_column="choices")
    assert first.content_sha256 == second.content_sha256
    assert first.identity() != second.identity()


def test_mixed_task_rows_are_not_silently_reclassified(tmp_path):
    path = write_jsonl(tmp_path, [{"question": "Q", "answer": "A", "choices": ["one", "two"]},
                                  {"question": "Number?", "answer": 2}])
    report = validate_dataset(path)
    assert not report.valid and report.errors[0].row == 2
    assert "missing choices" in report.errors[0].message
