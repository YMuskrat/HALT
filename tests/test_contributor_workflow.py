"""Public signal plumbing and complete scaffold workflows, without model downloads."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from halt import Budget, HaltRunner, MultipleChoiceTask
from halt.backends import ScriptedBackend
from halt.errors import ConfigurationError
from halt.methods import ProbeMethod
from halt.sdk import scaffold_method
from halt.types import (
    Continue,
    EmbedSteps,
    EventKind,
    Finalize,
    MethodSpec,
    NextTokenFeatures,
    Observation,
    Phase,
    PrefixRef,
    ProbeFailure,
    ProbeResult,
    RunContext,
    ScoreFrame,
    ScoreNextToken,
    StepBoundary,
    StepEmbeddings,
    WorkLimit,
)


class BatchMethod(ProbeMethod):
    spec = MethodSpec("batch_fixture")

    def reset(self, context):
        super().reset(context)
        self._received = []
        self._invalid = 0

    def on_boundary(self, event):
        self.request(
            self.signal(ScoreNextToken, include_entropy=True, max_work=WorkLimit(scored_tokens=1)),
            self.signal(EmbedSteps, steps=(event.payload.text,), max_work=WorkLimit()),
        )

    def on_probes(self, results):
        assert isinstance(results[0].value, NextTokenFeatures)
        assert results[0].value.frame.vocabulary_coverage == "full"
        self._received.append(results)
        self.finalize("both_received", diagnostics={"measurements": len(results)})

    def invalid_probe(self):
        self._invalid += 1


def fixture_batch():
    method = BatchMethod()
    context = RunContext("fixture", "batch_fixture", "config", {}, ScriptedBackend().info,
                         0, 1, Budget())
    prefix = PrefixRef("fixture", "main", 1, "prefix-1", "scripted-v1")
    method.reset(context)
    method.observe(Observation("boundary", EventKind.STEP_BOUNDARY, 0, Phase.REASONING,
                               prefix, StepBoundary("One.\n\n", 1, "blank_line")))
    requests = method.decide().requests
    for request in requests:
        method.observe(Observation("scheduled", EventKind.PROBE_SCHEDULED, 0, Phase.PROBING,
                                   prefix, request))
    return method, prefix, requests


def complete(method, prefix, request, *, result_prefix=None):
    value = (StepEmbeddings(((1.0, 0.0),), "fixture", "1") if isinstance(request, EmbedSteps)
             else NextTokenFeatures(ScoreFrame(prefix, vocabulary_coverage="full"), -1, -3, 0.5))
    result = ProbeResult(request.request_id, result_prefix or prefix, value)
    method.observe(Observation("completed", EventKind.PROBE_COMPLETED, 0, Phase.PROBING,
                               prefix, result))


def test_batch_waits_for_all_results_preserves_request_order_and_ignores_duplicates():
    method, prefix, requests = fixture_batch()
    complete(method, prefix, requests[1])
    assert isinstance(method.decide(), Continue)
    assert method._received == []
    complete(method, prefix, requests[1])
    complete(method, prefix, replace(requests[0], request_id="unrelated"))
    assert method._received == []
    complete(method, prefix, requests[0])
    assert isinstance(method.decide(), Finalize)
    assert [result.request_id for result in method._received[0]] == [r.request_id for r in requests]
    decision = method.decide()
    complete(method, prefix, requests[0])
    assert method.decide() == decision
    assert len(method._received) == 1


@pytest.mark.parametrize("failure", ["stale", "denied", "failed"])
def test_invalid_batch_never_delivers_partial_evidence(failure):
    method, prefix, requests = fixture_batch()
    complete(method, prefix, requests[0])
    if failure == "stale":
        complete(method, prefix, requests[1], result_prefix=replace(prefix, identity="wrong"))
    else:
        kind = EventKind.PROBE_DENIED if failure == "denied" else EventKind.PROBE_FAILED
        method.observe(Observation("failure", kind, 0, Phase.PROBING, prefix,
                                   ProbeFailure(requests[1].request_id, failure)))
    assert isinstance(method.decide(), Continue)
    assert method._invalid == 1
    assert method._received == []


def test_signal_request_prefix_bounds_and_ids_are_managed():
    method, prefix, requests = fixture_batch()
    assert requests[0].prefix == prefix
    assert requests[0].max_work.scored_tokens == 1
    assert requests[0].request_id != requests[1].request_id
    with pytest.raises(ConfigurationError, match="previous batch"):
        method.request(requests[0])
    for request in requests:
        complete(method, prefix, request)
    with pytest.raises(ConfigurationError, match="unique"):
        method.request(requests[0])
    with pytest.raises(ConfigurationError, match="current observation"):
        method.request(replace(requests[0], request_id="fresh", prefix=replace(prefix, position=0)))


def test_public_batch_helper_runs_with_accounted_scripted_signals():
    result = HaltRunner(ScriptedBackend()).run(
        MultipleChoiceTask("Which?", {"A": "no", "C": "yes"}), BatchMethod(), Budget())
    assert result.status == "completed", result.error
    assert result.stop.reason == "both_received"
    assert result.stop.diagnostics["measurements"] == 2
    assert result.usage.probe_calls == 2
    assert result.usage.embedding_calls == 1


def run_python(checkout, *arguments, extra_paths=()):
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(checkout / "src"), *(str(path) for path in extra_paths)]),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    process = subprocess.run([sys.executable, *arguments], cwd=checkout, env=environment,
                             text=True, capture_output=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    return process.stdout


@pytest.fixture
def checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "HALT"
    shutil.copytree(root / "src/halt", destination / "src/halt", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(root / "pyproject.toml", destination / "pyproject.toml")
    return destination


@pytest.mark.parametrize("template", ["boundary", "moving_average"])
def test_contribution_is_discovered_checked_and_run_without_registry_edits(checkout, template):
    method_id = f"fixture_{template}"
    registry_before = (checkout / "src/halt/registry.py").read_bytes()
    scaffold_method(method_id, checkout, contribute=True, template=template)
    assert (checkout / "src/halt/registry.py").read_bytes() == registry_before
    run_python(checkout, "-m", "pytest", f"tests/methods/test_{method_id}.py", "-q", "-p", "no:cacheprovider")
    inspection = json.loads(run_python(checkout, "-m", "halt", "methods", "inspect", method_id))
    assert inspection["method_card"]["source_relationship"] == "original_method"
    assert inspection["method_card"]["citation"] is None
    check = json.loads(run_python(checkout, "-m", "halt", "methods", "check", method_id))
    assert check["passed"]
    run_python(checkout, "-m", "halt", "run", "--config", f"configs/methods/{method_id}.json",
               "--output", "run.json")
    result = json.loads((checkout / "run.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["stop"]["reason"].startswith(method_id)
    with pytest.raises(ConfigurationError, match="preserved"):
        scaffold_method(method_id, checkout, contribute=True, template=template)


def test_plugin_card_is_packaged_and_discoverable(checkout):
    method_id = "fixture_packaged_card"
    plugin = scaffold_method(method_id, checkout / "plugin", template="moving_average")
    # Install-equivalent metadata avoids network access/build tooling in core tests.
    distribution = plugin / "src/halt_plugin_fixture_packaged_card-0.1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Metadata-Version: 2.1\nName: halt-plugin-fixture-packaged-card\nVersion: 0.1.0\n")
    (distribution / "entry_points.txt").write_text(
        "[halt.methods]\nfixture_packaged_card = halt_plugin_fixture_packaged_card:FixturePackagedCard\n")
    run_python(checkout, "-m", "pytest", str(plugin / "tests"), "-q", "-p", "no:cacheprovider",
               extra_paths=(plugin / "src",))
    inspection = json.loads(run_python(checkout, "-m", "halt", "methods", "inspect", method_id,
                                       extra_paths=(plugin / "src",)))
    assert inspection["method_card"]["method_id"] == method_id
    assert "method_card.json" in (plugin / "pyproject.toml").read_text()


def test_contribution_checks_all_paths_before_writing(checkout):
    conflicting = checkout / "tests/methods/test_fixture_conflict.py"
    conflicting.parent.mkdir(parents=True)
    conflicting.write_text("# existing author work\n")
    with pytest.raises(ConfigurationError, match="preserved"):
        scaffold_method("fixture_conflict", checkout, contribute=True)
    assert not (checkout / "src/halt/methods/fixture_conflict.py").exists()
    assert conflicting.read_text() == "# existing author work\n"
