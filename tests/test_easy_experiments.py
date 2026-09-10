import argparse
import json
from types import SimpleNamespace

import pytest

from halt.cli import main
from halt.errors import CapabilityError, ConfigurationError
from halt.experiments import trial_config
from halt.profiles.qwen3 import QWEN3_MODEL_ID, QWEN3_REVISION
from halt.profiles.selection import resolve_model_settings


def questions(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text('question,A,B,C,answer\nWhat is 2+2?,3,5,4,C\n', encoding="utf-8")
    return path


def test_dataset_check_and_simple_benchmark_save_results(tmp_path, capsys):
    path = questions(tmp_path)
    output = tmp_path / "report"
    assert main(["dataset", "check", str(path)]) == 0
    assert "Valid dataset" in capsys.readouterr().out
    assert main(["benchmark", "--dataset", str(path), "--backend", "scripted",
                 "--methods", "halt_cot", "--param", "halt_cot.threshold=0.9",
                 "--output-dir", str(output), "--quiet"]) == 0
    printed = capsys.readouterr().out
    assert "halt_cot" in printed and "full_reasoning" in printed
    assert "Results saved" in printed
    for name in ("results.csv", "results.jsonl", "summary.csv", "manifest.json", "report.html"):
        assert (output / name).is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    methods = manifest["resolved_configuration"]["evaluation"]["methods"]
    assert methods == [{"name": "full_reasoning"}, {"name": "halt_cot", "parameters": {"threshold": 0.9}}]


def test_bad_dataset_and_incompatible_method_fail_before_network(tmp_path, monkeypatch, capsys):
    def forbidden():
        pytest.fail("invalid input must not load optional model dependencies or contact the hub")
    monkeypatch.setattr("halt.profiles.selection._hub", forbidden)
    path = tmp_path / "bad.csv"
    path.write_text("question,answer\n2+2?,\n", encoding="utf-8")
    assert main(["dataset", "check", str(path)]) == 1
    assert "Row" in capsys.readouterr().err
    assert main(["benchmark", "--dataset", str(path)]) == 2
    path.write_text("question,answer\n2+2?,4\n", encoding="utf-8")
    assert main(["benchmark", "--dataset", str(path), "--methods", "halt_cot"]) == 2
    assert "candidate" in capsys.readouterr().err


def test_numeric_default_is_compatible_and_parameter_errors_are_early(tmp_path):
    path = tmp_path / "numeric.csv"
    path.write_text("question,answer\n2+2?,4\n", encoding="utf-8")
    config = trial_config(argparse.Namespace(dataset=str(path), backend="scripted"))
    assert [m["name"] for m in config.evaluation["methods"]] == ["full_reasoning", "answer_convergence"]
    with pytest.raises(ConfigurationError, match="unselected"):
        trial_config(argparse.Namespace(dataset=str(path), backend="scripted", param=["missing.threshold=1"]))
    with pytest.raises(ConfigurationError, match="invalid parameters"):
        trial_config(argparse.Namespace(dataset=str(questions(tmp_path)), backend="scripted", param=["halt_cot.typo=1"]))


def test_model_change_drops_inherited_checkpoint_pin(tmp_path):
    config = trial_config(argparse.Namespace(dataset=str(questions(tmp_path)), model="example/qwen-finetune"))
    assert config.model["name"] == "example/qwen-finetune"
    assert config.model.get("revision") is None
    assert config.model.get("tokenizer_revision") is None
    assert config.backend["name"] == "transformers"


def fake_hub(tmp_path, monkeypatch, model_type="qwen3"):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    calls = []

    def info(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return SimpleNamespace(sha="a" * 40)

    def download(model_id, filename, **kwargs):
        assert filename == "config.json"
        assert len(kwargs["revision"]) == 40
        return str(path)

    hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=info), hf_hub_download=download)
    monkeypatch.setattr("halt.profiles.selection._hub", lambda: hub)
    return calls


def test_named_model_revision_resolves_once_and_default_stays_pinned(tmp_path, monkeypatch):
    calls = fake_hub(tmp_path, monkeypatch)
    result = resolve_model_settings({"name": "example/qwen", "revision": "release", "tokenizer_revision": "release"})
    assert result["revision"] == result["tokenizer_revision"] == "a" * 40
    assert len(calls) == 1
    result = resolve_model_settings({"name": QWEN3_MODEL_ID})
    assert result["revision"] == result["tokenizer_revision"] == QWEN3_REVISION
    assert len(calls) == 1


def test_unsupported_family_and_offline_unpinned_checkpoint_fail_early(tmp_path, monkeypatch):
    calls = fake_hub(tmp_path, monkeypatch, model_type="llama")
    with pytest.raises(CapabilityError, match="model_type='llama'"):
        resolve_model_settings({"name": "example/llama", "revision": "b" * 40})
    with pytest.raises(ConfigurationError, match="Offline"):
        resolve_model_settings({"name": "example/qwen", "local_files_only": True})
    assert not calls


def test_static_model_doctor_does_not_contact_hub(monkeypatch, capsys):
    monkeypatch.setattr("halt.profiles.selection._hub", lambda: pytest.fail("static doctor must stay offline"))
    assert main(["doctor", "--model", "example/qwen"]) == 0
    assert json.loads(capsys.readouterr().out)["model_loaded"] is False


def experiment_file(tmp_path, **sections):
    path = tmp_path / "experiment.json"
    data = {"backend": {"name": "scripted", "parameters": {"answer": "B"}},
            "evaluation": {"dataset": str(questions(tmp_path)), "adapter": "auto",
                           "methods": [{"name": "full_reasoning"},
                                       {"name": "halt_cot", "parameters": {"threshold": 0.3, "consecutive": 1}}]}}
    data.update(sections)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_selecting_existing_backend_and_methods_preserves_configuration(tmp_path):
    path = experiment_file(tmp_path)
    config = trial_config(argparse.Namespace(config=str(path), backend="scripted",
                                            methods=["halt_cot"], param=["halt_cot.threshold=0.7"]))
    assert config.backend["parameters"] == {"answer": "B"}
    assert config.evaluation["methods"] == [{"name": "full_reasoning"},
        {"name": "halt_cot", "parameters": {"threshold": 0.7, "consecutive": 1}}]


def test_same_transformers_backend_keeps_custom_backend_parameters(tmp_path):
    path = experiment_file(tmp_path, backend={"name": "transformers", "parameters": {"answer_recipes": {}}})
    config = trial_config(argparse.Namespace(config=str(path), backend="transformers", model="example/qwen"))
    assert config.backend["parameters"] == {"answer_recipes": {}}


def test_fixed_budget_defaults_and_baseline_use_evaluator_identifiers(tmp_path, capsys):
    path = questions(tmp_path)
    output = tmp_path / "fixed-report"
    assert main(["benchmark", "--dataset", str(path), "--backend", "scripted",
                 "--methods", "fixed_reasoning_budget", "immediate_answer",
                 "--baseline", "fixed_reasoning_budget", "--output-dir", str(output), "--quiet"]) == 0
    assert "fixed_reasoning_budget_32" in capsys.readouterr().out
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resolved_configuration"]["evaluation"]["baseline"] == "fixed_reasoning_budget_32"


def test_configured_fixed_budget_ids_select_variants_and_parameter_overrides(tmp_path):
    path = experiment_file(tmp_path, evaluation={"dataset": str(questions(tmp_path)), "adapter": "auto",
        "baseline": "fixed_reasoning_budget_4", "methods": [
            {"name": "fixed_reasoning_budget", "parameters": {"tokens": 4}},
            {"name": "fixed_reasoning_budget", "parameters": {"tokens": 8}},
            {"name": "halt_cot", "parameters": {"threshold": 0.2}}]})
    config = trial_config(argparse.Namespace(config=str(path), methods=["halt_cot", "fixed_reasoning_budget_8"],
                                            param=["fixed_reasoning_budget_8.tokens=10"]))
    assert config.evaluation["baseline"] == "fixed_reasoning_budget_4"
    assert config.evaluation["methods"] == [
        {"name": "fixed_reasoning_budget", "parameters": {"tokens": 4}},
        {"name": "halt_cot", "parameters": {"threshold": 0.2}},
        {"name": "fixed_reasoning_budget", "parameters": {"tokens": 10}}]
    with pytest.raises(ConfigurationError, match="multiple configured variants"):
        trial_config(argparse.Namespace(config=str(path), methods=["fixed_reasoning_budget"]))


def test_invalid_method_ids_and_seed_fail_before_model_resolution(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("halt.profiles.selection._hub", lambda: pytest.fail("invalid experiment must fail before network"))
    path = questions(tmp_path)
    assert main(["benchmark", "--dataset", str(path), "--methods", "missing_stopper"]) == 2
    assert "unknown method" in capsys.readouterr().err
    assert main(["benchmark", "--dataset", str(path), "--seed", "-1"]) == 2
    assert "seed" in capsys.readouterr().err


def test_duplicate_configured_method_ids_fail_before_model_resolution(tmp_path, monkeypatch, capsys):
    path = experiment_file(tmp_path, evaluation={"dataset": str(questions(tmp_path)), "adapter": "auto",
        "methods": [{"name": "full_reasoning", "id": "same"}, {"name": "immediate_answer", "id": "same"}]})
    monkeypatch.setattr("halt.profiles.selection._hub", lambda: pytest.fail("duplicate IDs must fail before network"))
    assert main(["benchmark", "--config", str(path), "--model", "example/qwen"]) == 2
    assert "duplicate evaluation method IDs" in capsys.readouterr().err


def test_probe_call_override_preserves_explicit_total_token_cap(tmp_path):
    path = experiment_file(tmp_path, budget={"max_reasoning_tokens": 100, "max_answer_tokens": 10,
        "max_probe_output_tokens": 40, "max_total_generated_tokens": 70})
    config = trial_config(argparse.Namespace(config=str(path), max_probe_calls=3, max_reasoning_tokens=200))
    assert config.budget.max_total_generated_tokens == 70
    assert config.budget.max_probe_calls == 3
    assert config.budget.max_reasoning_tokens == 200


@pytest.mark.parametrize("settings", [
    {"name": ""}, {"revision": 12}, {"tokenizer_revision": ""}, {"dtype": "invalid"},
    {"dtype": []}, {"device": None}, {"local_files_only": "false"}, {"temperature": -1},
    {"temperature": float("nan")}, {"temperature": "hot"}, {"top_p": 0}, {"top_k": 1.5},
])
def test_invalid_model_settings_fail_before_hub_access(settings, monkeypatch):
    monkeypatch.setattr("halt.profiles.selection._hub", lambda: pytest.fail("invalid settings must not contact hub"))
    with pytest.raises(ConfigurationError):
        resolve_model_settings(settings)


def test_nonobject_model_metadata_is_a_clear_configuration_error(tmp_path, monkeypatch):
    fake_hub(tmp_path, monkeypatch)
    (tmp_path / "config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Could not read model configuration"):
        resolve_model_settings({"name": QWEN3_MODEL_ID})
