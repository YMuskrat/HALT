"""Author scaffolding and a small runtime conformance check."""
from __future__ import annotations

from typing import Any

from halt._scaffold import scaffold_method
from halt.registry import MethodRegistry
from halt.types import to_data

__all__ = ["check_conformance", "scaffold_method"]


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
