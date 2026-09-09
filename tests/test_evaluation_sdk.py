import json
from dataclasses import replace

import pytest

from halt import Budget, HaltRunner, MultipleChoiceTask
from halt.backends import ScriptedBackend
from halt.config import load_config, validate_config
from halt.errors import ConfigurationError
from halt.evaluation import evaluate
from halt.evaluation.calibration import calibrate
from halt.evaluation.datasets import load_dataset
from halt.evaluation.reporting import paired_bootstrap
from halt.methods import Refrain
from halt.registry import MethodRegistry
from halt.sdk import scaffold_method
from halt.session import RefrainSession


def test_config_rejects_unknown_and_numeric_choices():
    with pytest.raises(ConfigurationError, match="unknown"):
        validate_config({"budget": {"typo": 1}})
    with pytest.raises(ConfigurationError, match="invalid parameters"):
        validate_config({"method": {"name": "halt_cot", "parameters": {"threshol": 1}}})
    with pytest.raises(ConfigurationError, match="choices"):
        validate_config({"task": {"type": "numeric", "question": "2+2", "choices": {"A": "4"}}})


def test_labels_never_enter_inference_objects(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"id": "1", "question": "pick", "choices": {"A": "yes", "B": "no"}, "answer": "A"}))
    dataset = load_dataset(data)
    task = dataset.items[0].task
    assert "reference" not in vars(task) and "answer" not in task.visible()
    assert task.candidates == ("A", "B")
    assert dataset.items[0].correct("A")


def test_resume_identity_and_reports(tmp_path):
    config = load_config("configs/benchmark_mcq.yaml")
    backend = ScriptedBackend()
    summary = evaluate(config, tmp_path, backend=backend)
    calls = len(backend.opened_runs)
    assert calls > 0
    assert summary["methods"][0]["evidence_kind"] == ["simulated"]
    evaluate(config, tmp_path, backend=backend)
    assert len(backend.opened_runs) == calls
    changed = replace(config, runtime={**config.runtime, "seed": 47})
    evaluate(changed, tmp_path, backend=backend)
    assert len(backend.opened_runs) == calls * 2
    for name in ("summary.json", "summary.csv", "report.md", "per_example.jsonl", "manifest.json"):
        assert (tmp_path / name).is_file()
    row = summary["methods"][0]
    assert row["aggregate_generated_token_reduction"] == 0
    assert row["generated_reduction_baseline_denominator"] > 0


def test_calibration_requires_split_and_freeze(tmp_path):
    config = load_config("configs/calibrate_halt_cot.yaml")
    with pytest.raises(ConfigurationError, match="split"):
        calibrate(replace(config, evaluation={**config.evaluation, "split": "test"}), tmp_path / "bad.json")
    result = calibrate(config, tmp_path / "calibration.json")
    assert result["statistical_guarantee"] is False
    assert result["chosen_parameters"] in result["tested_parameters"]


def test_bootstrap_uses_pairs_and_explicit_rng():
    assert paired_bootstrap([1, 1], seed=5) == (1, 1)
    assert paired_bootstrap([-1, 0, 1], seed=5) == paired_bootstrap([-1, 0, 1], seed=5)


def test_registry_collision_and_plugin_api():
    class Entry:
        name = "halt_cot"
    with pytest.raises(ConfigurationError, match="collision"):
        MethodRegistry(entry_points=[Entry()])


def test_scaffold_preserves_existing_work(tmp_path):
    path = scaffold_method("my_stopper_fixture", tmp_path / "plugin")
    assert (path / "pyproject.toml").is_file()
    assert (path / "method_card.json").is_file()
    with pytest.raises(ConfigurationError, match="preserved"):
        scaffold_method("my_stopper_fixture", path)


def test_runtime_adaptive_session_updates_once():
    session = RefrainSession((0.6, 0.8))
    backend = ScriptedBackend(reasoning=["The answer is C.\n\n", "Wait, check again.\n\n"])
    runner = HaltRunner(backend)
    task = MultipleChoiceTask("Which?", {"A": "no", "C": "yes"})
    first = runner.run(task, Refrain(), Budget(), session=session, run_id="first")
    assert first.status == "completed", first.error
    assert first.stop.reason == "reflective_redundancy"
    assert first.provenance["method"]["variant"] == "refrain_adaptive_common_v1"
    assert first.provenance["configuration"]["threshold"] == 0.6
    assert first.provenance["session_update"]["updated"] is True
    runner.run(task, Refrain(), Budget(), session=session, run_id="first")
    assert len(session.to_dict()["completed"]) == 1


def test_mcq_whole_option_normalization():
    task = MultipleChoiceTask("2+2?", {"A": "3", "B": "4"})
    assert task.normalize("**B. 4**") == "B"
    assert task.normalize("B. 5") is None
    assert task.normalize("A or B") is None


def test_established_arc_adapter_preserves_visible_candidates():
    from halt.evaluation.huggingface import ARC_AUDITED_REVISION, arc_rows
    rows = [{"id": "fixture", "question": "Which?", "choices": {
        "label": ["1", "2"], "text": ["one", "two"]}, "answerKey": "2"}]
    dataset = arc_rows(rows, revision=ARC_AUDITED_REVISION, subset="ARC-Easy", split="validation")
    assert dataset.items[0].task.candidates == ("1", "2")
    assert dataset.items[0].reference == "2"
    assert "answerKey" not in dataset.items[0].task.visible()
    with pytest.raises(ConfigurationError, match="immutable"):
        arc_rows(rows, revision="main", subset="ARC-Easy", split="validation")
