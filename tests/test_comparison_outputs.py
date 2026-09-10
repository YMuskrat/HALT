import copy
import csv
import json
from dataclasses import replace
from html.parser import HTMLParser

import pytest

from halt.backends import ScriptedBackend
from halt.config import load_config
from halt.errors import ConfigurationError
from halt.evaluation import compare, evaluate, grouped_compare, load_results, render_comparison
from halt.evaluation.analysis import RESULTS_SCHEMA_VERSION, measurement_row
from halt.evaluation.reporting import write_reports


def record(method="full_reasoning", question="q1", *, correct=True, generated=10, seed=1,
           adaptive=False, evidence="real", dataset="fixture", status="completed"):
    return {
        "schema_version": "1.0", "identity": f"{method}-{question}-{seed}",
        "item_id": question, "method_id": method, "correct": correct,
        "evidence_kind": evidence,
        "session_before": {"session_id": "ordered-session"} if adaptive else None,
        "evaluation": {"question": "What is 2 + 2?", "choices": {"A": "4"}, "reference": "A"},
        "manifest": {"backend": {"name": "test-backend", "model_id": "test-model",
            "model_revision": "pinned", "tokenizer_revision": "pinned", "hardware": {"device": "cpu"}},
            "dataset": {"content_sha256": dataset, "split": "test", "ordered_item_ids": ["q1", "q2"]},
            "runtime": {"seed": seed, "capture": "none"},
            "method_spec": {"method_id": method}, "method_configuration": {"name": method},
            "environment": {"halt_source_sha256": "test-code"}, "order": 0},
        "result": {"answer": "A" if correct else None, "status": status,
            "error": None if status == "completed" else "failed fixture",
            "stop": {"reason": "method_stop", "reasoning_position": generated - 3, "diagnostics": {}},
            "usage": {"reasoning_tokens": generated - 3, "answer_tokens": 1,
                "probe_output_tokens": 2, "total_generated_tokens": generated,
                "input_tokens": 100, "recomputed_prefix_tokens": 80, "forced_context_tokens": 4,
                "scored_tokens": 7, "probe_calls": 2, "forward_calls": 12, "embedding_calls": 1,
                "measurement": "observed", "operations": [{"kind": "probe", "duration_seconds": 2}]},
            "durations": {"reasoning": 1.0, "probes": 2.0, "finalization": 1.0, "end_to_end": 4.2}}}


def paired_records():
    return [record(question="q1", generated=10), record(question="q2", generated=30),
        record("custom", "q1", generated=5),
        record("custom", "q2", generated=15, correct=False, status="backend_error")]


def test_exports_roundtrip_preserves_failed_runs_and_full_work(tmp_path):
    records = paired_records()
    summary = write_reports(records, tmp_path, baseline="full_reasoning", bootstrap_samples=100)
    for filename in ("results.jsonl", "results.csv", "per_example.jsonl"):
        rows = load_results(tmp_path / filename)
        assert len(rows) == 4
        assert rows[-1]["status"] == "backend_error"
        assert rows[-1]["correct"] is False
        assert rows[-1]["answer"] is None
        assert rows[-1]["reference"] == "A"
        assert rows[-1]["elapsed_seconds"] == 4.2  # Do not double-count component durations.
        assert rows[-1]["probe_output_tokens"] == 2
        assert rows[-1]["input_tokens"] == 100
        assert rows[-1]["scored_tokens"] == 7
        assert rows[-1]["results_schema_version"] == RESULTS_SCHEMA_VERSION
        assert compare(rows, bootstrap_samples=100) == summary
    detailed = json.loads((tmp_path / "results.jsonl").read_text().splitlines()[-1])
    assert detailed["details"]["result"]["usage"]["operations"] == records[-1]["result"]["usage"]["operations"]
    method = summary["methods"][1]
    assert method["accuracy"] == 0.5 and method["failure_count"] == 1
    assert method["aggregate_generated_token_reduction"] == 0.5
    assert method["generated_reduction_baseline_denominator"] == 40
    assert method["mean_total_generated_tokens"] == 10
    assert "Not completed" in render_comparison(summary)
    assert "inference" in summary["interpretation"] or "Measured runs" in summary["interpretation"]


def test_aggregation_pairs_questions_and_seeds_and_retains_unmatched():
    rows = [record(seed=1, correct=True), record(seed=2, correct=False),
            record("custom", seed=1, correct=False), record("custom", seed=2, correct=True),
            record("custom", "unmatched", seed=1)]
    summary = compare(rows, bootstrap_samples=20)
    custom = summary["methods"][1]
    assert custom["paired_sample_count"] == 2
    assert custom["unpaired_sample_count"] == 1
    assert custom["paired_accuracy_difference"] == 0
    assert custom["correct_to_wrong"] == 1 and custom["wrong_to_correct"] == 1
    assert custom["paired_accuracy_ci95"] is None
    assert "Repeated questions" in custom["uncertainty_note"]


def test_comparisons_reject_incompatible_experiments_duplicates_and_configs():
    original = paired_records()
    changed = [record(dataset="different")]
    with pytest.raises(ConfigurationError, match="incompatible"):
        compare(original + changed)
    assert len(grouped_compare(original + changed, bootstrap_samples=20)) == 2
    with pytest.raises(ConfigurationError, match="duplicate"):
        compare(original + [original[0]])
    changed = copy.deepcopy(original)
    changed[-1]["manifest"]["method_configuration"]["parameters"] = {"threshold": 0.9}
    with pytest.raises(ConfigurationError, match="different configurations"):
        compare(changed)


@pytest.mark.parametrize("adaptive_baseline", [True, False])
def test_adaptive_comparison_never_uses_iid_item_interval(adaptive_baseline):
    rows = [record(adaptive=adaptive_baseline), record("custom", adaptive=not adaptive_baseline)]
    custom = compare(rows, bootstrap_samples=20)["methods"][1]
    assert custom["paired_accuracy_ci95"] is None
    assert custom["paired_accuracy_ci95_low"] is None
    assert "Ordered dependent" in custom["uncertainty_note"]


def test_absent_timing_stays_unknown_and_no_pairs_is_reportable(tmp_path):
    baseline = record()
    baseline["result"]["durations"] = {}
    custom = record("custom", "different-question")
    result = write_reports([baseline, custom], tmp_path, baseline="full_reasoning")
    assert result["methods"][0]["mean_latency_seconds"] is None
    assert result["methods"][0]["timing_sample_count"] == 0
    assert result["methods"][1]["paired_accuracy_difference"] is None
    assert "n/a" in render_comparison(result)
    assert (tmp_path / "report.html").is_file()


@pytest.mark.parametrize("evidence,scope,label", [
    ("simulated", "simulated_runtime", "Simulated demonstration"),
    ("replay", "controller_only", "Replay demonstration"),
])
def test_demo_evidence_never_claims_model_timing(evidence, scope, label):
    result = compare([record(evidence=evidence)])
    assert result["methods"][0]["timing_scope"] == scope
    assert label in render_comparison(result)


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_html_escapes_dataset_method_diagnostics_and_has_no_external_assets(tmp_path):
    payload = '</td><script src="https://malicious.invalid/x"></script><img src=x onerror=alert(1)>'
    records = [record(), record(payload)]
    records[0]["evaluation"]["question"] = payload
    records[0]["result"]["stop"]["diagnostics"] = {"custom": payload}
    write_reports(records, tmp_path, baseline="full_reasoning", bootstrap_samples=20)
    source = (tmp_path / "report.html").read_text(encoding="utf-8")
    parsed = Tags()
    parsed.feed(source)
    assert len([tag for tag, _ in parsed.tags if tag == "script"]) == 1
    assert not any(tag == "img" for tag, _ in parsed.tags)
    assert not any(attrs.get("src") for _, attrs in parsed.tags)
    assert "&lt;script" in source
    assert "data-column" in source and 'id="search"' in source
    assert "Work per question" in source


def test_loader_rejects_wrong_schema_and_inconsistent_token_totals(tmp_path):
    row = measurement_row(record())
    row["results_schema_version"] = "99"
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="line 1.*unsupported"):
        load_results(path)
    row["results_schema_version"] = RESULTS_SCHEMA_VERSION
    row["total_generated_tokens"] += 1
    with pytest.raises(ConfigurationError, match="must include"):
        compare([row])


def test_summary_csv_has_stable_scalar_columns_for_adaptive_and_partial_timing(tmp_path):
    ordinary, adaptive, mixed = tmp_path / "ordinary", tmp_path / "adaptive", tmp_path / "mixed"
    write_reports([record()], ordinary, baseline="full_reasoning", bootstrap_samples=20)
    write_reports([record(adaptive=True)], adaptive, baseline="full_reasoning", bootstrap_samples=20)
    records = [record(question=f"q{i}") for i in range(20)]
    records.append(record("custom", "q1"))
    write_reports(records, mixed, baseline="full_reasoning", bootstrap_samples=20)
    contents = []
    for path in (ordinary, adaptive, mixed):
        with (path / "summary.csv").open(newline="", encoding="utf-8") as handle:
            contents.append(list(csv.DictReader(handle)))
    assert list(contents[0][0]) == list(contents[1][0]) == list(contents[2][0])
    assert "paired_accuracy_ci95" not in contents[0][0]
    assert contents[0][0]["paired_accuracy_ci95_low"] == "0.0"
    assert contents[1][0]["paired_accuracy_ci95_low"] == ""
    assert contents[2][0]["p95_latency_seconds"] == "4.2"
    assert contents[2][1]["p95_latency_seconds"] == ""
    lines = (mixed / "report.md").read_text().splitlines()
    table_lines = [line for line in lines if line.startswith("|")]
    assert len({line.count("|") for line in table_lines}) == 1


def test_evaluation_progress_includes_resumed_records_and_labels_stay_separate(tmp_path):
    config = load_config("configs/benchmark_mcq.yaml")
    config = replace(config, evaluation={**config.evaluation, "limit": 1,
        "methods": [{"name": "full_reasoning"}, {"name": "immediate_answer"}]})
    backend = ScriptedBackend()
    updates = []
    evaluate(config, tmp_path, backend=backend,
        progress=lambda complete, total, row: updates.append((complete, total, row)))
    assert [(complete, total) for complete, total, _ in updates] == [(1, 2), (2, 2)]
    assert updates[0][2]["evaluation"]["reference"]
    assert "reference" not in backend.opened_runs[0].context.task_visible
    calls = len(backend.opened_runs)
    updates.clear()
    evaluate(config, tmp_path, backend=backend,
        progress=lambda complete, total, row: updates.append((complete, total, row)))
    assert len(backend.opened_runs) == calls
    assert len(updates) == 2


def test_task_rejection_precedes_backend_loading(tmp_path, monkeypatch):
    data = tmp_path / "numeric.jsonl"
    data.write_text(json.dumps({"id": "1", "question": "2+2?", "answer": "4"}), encoding="utf-8")
    config = load_config("configs/benchmark_mcq.yaml")
    config = replace(config, evaluation={**config.evaluation, "dataset": str(data),
        "adapter": "numeric_jsonl", "methods": [{"name": "full_reasoning"}, {"name": "halt_cot"}]})
    def unexpected_load(config):
        pytest.fail("backend must not be loaded before task validation")
    monkeypatch.setattr("halt.evaluation.evaluator.make_backend", unexpected_load)
    with pytest.raises(ConfigurationError, match="candidate"):
        evaluate(config, tmp_path / "results")


def test_calibration_preserves_column_mapping_in_artifact(tmp_path):
    from halt.evaluation.calibration import calibrate
    from halt.evaluation.datasets import load_dataset

    data = tmp_path / "mapped.csv"
    data.write_text("key,prompt,gold,left,right\nq1,Which?,A,yes,no\n", encoding="utf-8")
    config = load_config("configs/calibrate_halt_cot.yaml")
    mappings = {"question_column": "prompt", "answer_column": "gold", "id_column": "key",
                "choice_columns": {"A": "left", "B": "right"}}
    config = replace(config, evaluation={**config.evaluation, "dataset": str(data), "adapter": "auto",
        "bootstrap_samples": 20, **mappings},
        calibration={**config.calibration, "parameter_grid": {"threshold": [0.1]}})
    artifact = calibrate(config, tmp_path / "calibration.json")
    expected = load_dataset(data, adapter="auto", split="calibration",
                            revision=config.evaluation["data_revision"], **mappings)
    assert artifact["dataset"] == expected.identity()


def test_calibration_invalid_data_precedes_backend_loading(tmp_path, monkeypatch):
    from halt.evaluation.calibration import calibrate

    path = tmp_path / "invalid.csv"
    path.write_text("question,answer\n,4\n", encoding="utf-8")
    config = load_config("configs/calibrate_halt_cot.yaml")
    config = replace(config, evaluation={**config.evaluation, "dataset": str(path), "adapter": "auto"})
    def unexpected_load(config):
        pytest.fail("invalid calibration data must fail before backend loading")
    monkeypatch.setattr("halt.evaluation.calibration.make_backend", unexpected_load)
    with pytest.raises(ConfigurationError):
        calibrate(config, tmp_path / "calibration.json")
