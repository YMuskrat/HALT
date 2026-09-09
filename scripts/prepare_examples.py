"""Build public synthetic fixtures and the independently packaged contributor example."""
from pathlib import Path

from halt import Budget, HaltRunner, MultipleChoiceTask
from halt.backends import ScriptedBackend
from halt.methods import HaltCoT
from halt.sdk import scaffold_method

target = Path("examples/external_plugin")
if not target.exists():
    scaffold_method("example_stopper", target)
result = HaltRunner(ScriptedBackend()).run(
    MultipleChoiceTask("12 minus 5?", {"A": "5", "B": "6", "C": "7", "D": "8"}),
    HaltCoT(threshold=0.6, consecutive=2), Budget(), seed=0,
    capture="full", trace_path="tests/fixtures/example_trace.jsonl", run_id="public-scripted-fixture-v1",
)
assert result.answer == "C" and result.status == "completed", result.error
print("Prepared public scripted trace and external plugin.")
