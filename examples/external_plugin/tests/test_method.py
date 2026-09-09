from halt import Budget, HaltRunner, MultipleChoiceTask
from halt.backends import ScriptedBackend
from halt.registry import get_method


def test_boundary_stop_and_fresh_run_state():
    method = get_method("example_stopper", boundaries=1)
    runner = HaltRunner(ScriptedBackend(reasoning=["One", ".", "\n\n", "Two", ".", "\n\n"]))
    task = MultipleChoiceTask("Which label is scripted?", {"A": "no", "C": "yes"})
    results = [runner.run(task, method, Budget()) for _ in range(2)]
    assert all(str(r.status) == "completed" and r.answer == "C" for r in results)
    assert [r.stop.reasoning_position for r in results] == [3, 3]
    assert all(r.stop.reason == "example_stopper_boundary" for r in results)
