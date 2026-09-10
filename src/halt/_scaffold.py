"""Generate working method packages or repository contributions without model dependencies."""
from __future__ import annotations

import json
from pathlib import Path

from halt.errors import ConfigurationError
from halt.registry import MethodRegistry, validate_method_id

_BOUNDARY = '''"""Original boundary-count example; replace the rule with your method."""
from dataclasses import dataclass

from halt.errors import ConfigurationError
from halt.methods import BaseMethod
from halt.types import Continue, Decision, EventKind, Finalize, MethodSpec, Observation, RunContext


@dataclass
class {class_name}(BaseMethod):
    boundaries: int = 1
    spec = MethodSpec("{method_id}", variant="original_boundary_example",
                      source_relationship="original_method", verification=("scripted_fixture",))

    def __post_init__(self) -> None:
        if type(self.boundaries) is not int or self.boundaries < 1:
            raise ConfigurationError("boundaries must be a positive integer")

    def reset(self, context: RunContext) -> None:
        super().reset(context)
        self._boundaries = 0

    def observe(self, event: Observation) -> None:
        if event.kind == EventKind.STEP_BOUNDARY:
            self._boundaries += 1

    def decide(self) -> Decision:
        if self._boundaries >= self.boundaries:
            return Finalize("{method_id}_boundary")
        return Continue()
'''

_MOVING_AVERAGE = '''"""Original example: stop when a full window's mean token entropy is low."""
import math
from collections import deque
from dataclasses import dataclass

from halt.errors import ConfigurationError
from halt.methods import ProbeMethod
from halt.types import (
    MethodSpec,
    NextTokenFeatures,
    Observation,
    ProbeResult,
    Requirements,
    RunContext,
    ScoreNextToken,
    WorkLimit,
)


@dataclass
class {class_name}(ProbeMethod):
    window: int = 3
    threshold: float = 1.1  # Nats; a demonstration setting, not a calibrated recommendation.
    spec = MethodSpec("{method_id}", variant="original_moving_average_example",
        requirements=Requirements(frozenset({{
            "visible_reasoning", "token_stream", "step_boundaries", "answer_transition",
            "prefix_branch", "exact_prefix_resume", "full_vocab_scores",
        }})), source_relationship="original_method", verification=("scripted_fixture",))

    def __post_init__(self) -> None:
        if type(self.window) is not int or self.window < 1:
            raise ConfigurationError("window must be a positive integer")
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ConfigurationError("threshold must be finite and nonnegative (nats)")

    def reset(self, context: RunContext) -> None:
        super().reset(context)
        self._history: deque[float] = deque(maxlen=self.window)

    def on_boundary(self, event: Observation) -> None:
        self.request_signal(ScoreNextToken, include_entropy=True, include_end_margin=False,
                            max_work=WorkLimit(scored_tokens=1))

    def on_probe(self, result: ProbeResult) -> None:
        value = result.value
        if (not isinstance(value, NextTokenFeatures) or value.frame.prefix != result.prefix
                or value.frame.processing != "raw" or value.frame.vocabulary_coverage != "full"
                or value.entropy is None or not math.isfinite(value.entropy) or value.entropy < 0):
            self.invalid_probe()
            return
        self._history.append(value.entropy)
        if len(self._history) == self.window:
            mean = sum(self._history) / self.window
            if mean < self.threshold:
                self.finalize("{method_id}_mean_entropy", diagnostics={{
                    "moving_average_entropy": mean, "units": "nats", "window": self.window,
                    "scope": "raw_full_vocabulary_at_reasoning_boundaries",
                }})

    def invalid_probe(self) -> None:
        self._history.clear()  # Missing evidence must not fill the window.
'''

_BOUNDARY_TEST = '''{external_import}from halt import Budget, HaltRunner, MultipleChoiceTask
from halt.backends import ScriptedBackend
{local_import}

def test_stop_boundary_and_fresh_question_state():
    task = MultipleChoiceTask("Which label is scripted?", {{"A": "no", "C": "yes"}})
    runner = HaltRunner(ScriptedBackend(reasoning=["One", ".", "\\n\\n", "Two", ".", "\\n\\n"]))
    method = {class_name}(boundaries=2)
    results = [runner.run(task, method, Budget()) for _ in range(2)]
    assert all(result.status == "completed" and result.answer == "C" for result in results)
    assert [result.stop.reasoning_position for result in results] == [6, 6]
    assert all(result.stop.reason == "{method_id}_boundary" for result in results)
'''

_MOVING_TEST = '''{external_import}from halt import Budget
from halt.backends import ScriptedBackend
{local_import}from halt.types import (
    Continue,
    EventKind,
    Finalize,
    NextTokenFeatures,
    Observation,
    Phase,
    PrefixRef,
    ProbeFailure,
    ProbeResult,
    RequestSignals,
    RunContext,
    ScoreFrame,
    StepBoundary,
)


def measure(method, entropy, position):
    prefix = PrefixRef("fixture", "main", position, str(position), "scripted-v1")
    method.observe(Observation(str(position), EventKind.STEP_BOUNDARY, 0, Phase.REASONING,
                               prefix, StepBoundary("Step.\\n\\n", position, "blank_line")))
    decision = method.decide()
    assert isinstance(decision, RequestSignals)
    request = decision.requests[0]
    method.observe(Observation("scheduled", EventKind.PROBE_SCHEDULED, 0, Phase.PROBING,
                               prefix, request))
    if entropy is None:
        method.observe(Observation("denied", EventKind.PROBE_DENIED, 0, Phase.PROBING,
                                   prefix, ProbeFailure(request.request_id, "fixture budget")))
    else:
        value = NextTokenFeatures(ScoreFrame(prefix, vocabulary_coverage="full"), -1, -3, entropy)
        method.observe(Observation("result", EventKind.PROBE_COMPLETED, 0, Phase.PROBING,
                                   prefix, ProbeResult(request.request_id, prefix, value)))
    return method.decide()


def test_mean_strict_threshold_missing_measurement_and_reset():
    method = {class_name}(window=2, threshold=0.5)
    context = RunContext("fixture", "{method_id}", "config", {{"candidates": ("A", "B")}},
                         ScriptedBackend().info, 0, 1, Budget())
    method.reset(context)
    assert isinstance(measure(method, 0.8, 1), Continue)
    assert isinstance(measure(method, 0.2, 2), Continue)  # Mean equals threshold.
    assert isinstance(measure(method, None, 3), Continue)  # Clear the window.
    assert isinstance(measure(method, 0.1, 4), Continue)
    decision = measure(method, 0.3, 5)
    assert isinstance(decision, Finalize)
    assert decision.diagnostics["moving_average_entropy"] == 0.2
    assert method.decide() == decision
    method.reset(context)
    assert isinstance(measure(method, 0.1, 1), Continue)
'''


def scaffold_method(method_id: str, output: str | Path | None = None, *,
                    contribute: bool = False, template: str = "boundary") -> Path:
    """Generate an external plugin, or add a method to an existing HALT checkout.

    In contribution mode, output identifies the checkout root (default: cwd).
    No shared registry file is edited. The packaged method card explicitly names
    its implementation and is discovered by MethodRegistry.
    """
    validate_method_id(method_id)
    if method_id in MethodRegistry().entries:
        raise ConfigurationError(f"method ID {method_id!r} is already registered")
    if template not in {"boundary", "moving_average"}:
        raise ConfigurationError("template must be boundary or moving_average")
    directory = Path(output or (Path.cwd() if contribute else f"examples/{method_id}")).resolve()
    if contribute:
        if not (directory / "src/halt/registry.py").is_file() or not (directory / "pyproject.toml").is_file():
            raise ConfigurationError("--contribute requires a HALT checkout root as --output (or cwd)")
    elif directory.exists() and any(directory.iterdir()):
        raise ConfigurationError("plugin output directory must be empty; existing author work is preserved")
    module = f"halt.methods.{method_id}" if contribute else f"halt_plugin_{method_id}"
    class_name = "".join(part.capitalize() for part in method_id.split("_"))
    implementation_import = f"from {module} import {class_name}\n"
    fields = {"method_id": method_id, "module": module, "class_name": class_name,
              "local_import": implementation_import if contribute else "",
              "external_import": "" if contribute else implementation_import + "\n"}
    moving = template == "moving_average"
    source = (_MOVING_AVERAGE if moving else _BOUNDARY).format(**fields)
    test = (_MOVING_TEST if moving else _BOUNDARY_TEST).format(**fields)
    parameters = {"window": 3, "threshold": 1.1} if moving else {"boundaries": 1}
    card = {"schema_version": "1.0", "method_id": method_id, "plugin_api_version": "1",
            "title": "Original moving-average entropy example" if moving else "Original boundary-count example",
            "authors": [], "citation": None, "source_revision": None,
            "source_relationship": "original_method", "implementation_status": "functional_example",
            "verification": ["scripted_fixture"],
            "limitations": ["Example algorithm and settings; no accuracy or efficiency benefit established."],
            "attribution_guidance": "Original methods need authors and a description. Cite and pin source material only when adapting or reusing it."}
    config = {"backend": {"name": "scripted", "parameters": {
        "reasoning": ["First check.\n\n", "Second check.\n\n", "Third check.\n\n", "Fourth check.\n\n"]}},
        "method": {"name": method_id, "parameters": parameters}}
    encode = lambda value: json.dumps(value, indent=2) + "\n"  # noqa: E731
    if contribute:
        card["implementation"] = f"{module}:{class_name}"
        generated = {
            f"src/halt/methods/{method_id}.py": source,
            f"tests/methods/test_{method_id}.py": test,
            f"src/halt/resources/method_cards/{method_id}.json": encode(card),
            f"configs/methods/{method_id}.json": encode(config),
        }
    else:
        generated = {
            "pyproject.toml": f'''[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "halt-plugin-{method_id.replace('_', '-')}"
version = "0.1.0"
description = "Original HALT stopping-method example"
requires-python = ">=3.11"
dependencies = ["halt-reasoning>=0.1.0rc1,<0.2"]

[project.entry-points."halt.methods"]
{method_id} = "{module}:{class_name}"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
{module} = ["method_card.json"]
''',
            f"src/{module}/__init__.py": source,
            "tests/test_method.py": test,
            "method_card.json": encode(card),
            f"src/{module}/method_card.json": encode(card),
            "config.json": encode(config),
            "README.md": f'''# {method_id}

An original `{template}` example. Edit the method and its test, then fill the packaged
`src/{module}/method_card.json` with your description and authors. The root card is an
initial copy for browsing; the packaged card is authoritative. Original ideas need no
paper. Cite and pin sources when adapting or reusing other work.

```console
python -m pip install -e .
python -m pytest tests
halt methods check {method_id}
halt methods inspect {method_id}
halt run --config config.json --output run.json
halt benchmark --dataset questions.csv --methods full_reasoning,{method_id}
```

The example run is scripted. The benchmark uses real inference unless `--backend scripted`
is selected explicitly. Install HALT's model extra before real inference. See HALT's
author guide and signal reference for the shared lifecycle and measurement contracts.
''',
        }
    # Validate all destinations before creating anything; never overwrite author files
    # or follow a generated parent symlink outside the named output directory.
    for relative in generated:
        destination = directory / relative
        if not destination.resolve().is_relative_to(directory):
            raise ConfigurationError(f"generated path escapes output directory: {relative}")
        if destination.exists():
            raise ConfigurationError(f"existing author work is preserved: {relative}")
    for relative, content in generated.items():
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return directory
