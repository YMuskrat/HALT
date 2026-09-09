"""Author scaffolding and a small runtime conformance check."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from halt.errors import ConfigurationError
from halt.registry import MethodRegistry, validate_method_id
from halt.types import to_data


def scaffold_method(method_id: str, output: str | Path) -> Path:
    validate_method_id(method_id)
    if method_id in MethodRegistry().entries:
        raise ConfigurationError(f"method ID {method_id!r} is already registered")
    directory = Path(output).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ConfigurationError("plugin output directory must be empty; existing author work is preserved")
    module = f"halt_plugin_{method_id}"
    class_name = "".join(part.capitalize() for part in method_id.split("_"))
    files = {
        "pyproject.toml": f'''[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "halt-plugin-{method_id.replace('_', '-')}"
version = "0.1.0"
description = "Example HALT method plugin; replace with your source-specific implementation"
requires-python = ">=3.11"
dependencies = ["halt-reasoning>=0.1.0rc1,<0.2"]

[project.entry-points."halt.methods"]
{method_id} = "{module}:{class_name}"

[tool.setuptools.packages.find]
where = ["src"]
''',
        f"src/{module}/__init__.py": f'''"""Original demonstration plugin; no published-method fidelity is claimed."""
from dataclasses import dataclass

from halt.errors import ConfigurationError
from halt.methods.base import BaseMethod
from halt.types import Continue, EventKind, Finalize, MethodSpec


@dataclass
class {class_name}(BaseMethod):
    """Stop after a configured number of observed boundaries."""

    boundaries: int = 1
    spec = MethodSpec("{method_id}", variant="author_example", api_version="1",
                      source_relationship="original_demo", verification=("plugin_fixture",))

    def __post_init__(self):
        if type(self.boundaries) is not int or self.boundaries < 1:
            raise ConfigurationError("boundaries must be a positive integer")

    def reset(self, context):
        super().reset(context)
        self._boundaries = 0

    def observe(self, event):
        if event.kind == EventKind.STEP_BOUNDARY:
            self._boundaries += 1

    def decide(self):
        return Finalize("{method_id}_boundary") if self._boundaries >= self.boundaries else Continue()
''',
        "tests/test_method.py": f'''from halt import Budget, HaltRunner, MultipleChoiceTask
from halt.backends import ScriptedBackend
from halt.registry import get_method


def test_boundary_stop_and_fresh_run_state():
    method = get_method("{method_id}", boundaries=1)
    runner = HaltRunner(ScriptedBackend(reasoning=["One", ".", "\\n\\n", "Two", ".", "\\n\\n"]))
    task = MultipleChoiceTask("Which label is scripted?", {{"A": "no", "C": "yes"}})
    results = [runner.run(task, method, Budget()) for _ in range(2)]
    assert all(str(r.status) == "completed" and r.answer == "C" for r in results)
    assert [r.stop.reasoning_position for r in results] == [3, 3]
    assert all(r.stop.reason == "{method_id}_boundary" for r in results)
''',
        "method_card.json": json.dumps({"schema_version": "1.0", "method_id": method_id,
            "plugin_api_version": "1", "title": "Original boundary-stop example",
            "authors": [], "citation": None, "source_revision": None,
            "source_relationship": "original_demo", "implementation_status": "functional_demo",
            "verification": ["scripted_fixture"], "limitations": ["Replace the demo algorithm and fill source attribution before claiming research fidelity."]}, indent=2) + "\n",
        "config.json": json.dumps({"backend": {"name": "scripted"},
                                  "method": {"name": method_id, "parameters": {"boundaries": 1}}}, indent=2) + "\n",
        "fixtures/boundaries.json": json.dumps({"schema_version": "1.0", "reasoning": ["One", ".", "\n\n", "Two", ".", "\n\n"],
                                               "expected_reasoning_position": 3}, indent=2) + "\n",
        "README.md": f'''# {method_id}

This independently packaged example stops at a text boundary. It is original demonstration code with no paper attribution.

Install HALT, then run from this directory:

```console
python -m pip install -e .
python -m pytest tests
halt methods inspect {method_id}
halt doctor --config config.json
halt run --config config.json --output run.json
```

Replace the decision rule, supply source citation and immutable revision, and add source-derived fixtures. Method code receives inference-only observations and requests typed probes; it must not call a model, manage caches, or access reference labels. API version 1 requires reset, observe, decide and a MethodSpec. Keep decide deterministic for unchanged state. No author review is implied by this scaffold.
''',
    }
    for relative, content in files.items():
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return directory


def check_conformance(method_id: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    from halt.api import HaltRunner
    from halt.backends.scripted import ScriptedBackend
    from halt.tasks import MultipleChoiceTask
    from halt.types import Budget

    method = MethodRegistry().create(method_id, parameters)
    backend = ScriptedBackend()
    runner = HaltRunner(backend)
    task = MultipleChoiceTask("12 minus 5?", {"A": "5", "B": "6", "C": "7", "D": "8"})
    inspection = runner.inspect(task, method, Budget())
    first = runner.run(task, method, Budget(), seed=0)
    second = runner.run(task, method, Budget(), seed=0)
    equal = (first.answer, first.status, first.stop.reason, first.stop.reasoning_position) == (
        second.answer, second.status, second.stop.reason, second.stop.reasoning_position)
    passed = equal and all(str(x.status) not in {"method_error", "backend_error"} for x in (first, second))
    return {"schema_version": "1.0", "method_id": method_id, "passed": passed,
            "evidence": "scripted_conformance_only", "repeated_configuration_isolated": equal,
            "inspection": to_data(inspection), "results": [first.to_dict(), second.to_dict()]}
