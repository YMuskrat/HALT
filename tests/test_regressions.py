from halt import Budget, CancellationToken, HaltRunner, MultipleChoiceTask
from halt.backends import ScriptedBackend
from halt.methods import HaltCoT
from halt.methods.baselines import FullReasoning
from halt.types import EventKind


def test_cancelled_scheduled_probe_has_terminal_event_and_no_dispatch():
    token = CancellationToken()
    backend = ScriptedBackend()
    stream = HaltRunner(backend).stream(
        MultipleChoiceTask("12-5?", {"A": "5", "C": "7"}), HaltCoT(), Budget(), cancellation=token)
    events = []
    for event in stream:
        events.append(event)
        if event.kind == EventKind.PROBE_SCHEDULED:
            token.cancel()
    assert events[-1].payload.status == "cancelled"
    assert sum(e.kind == EventKind.PROBE_SCHEDULED for e in events) == 1
    assert sum(e.kind == EventKind.PROBE_FAILED for e in events) == 1
    assert events[-1].payload.usage.probe_calls == 0
    assert backend.opened_runs[-1].closed


def test_completed_prefix_survives_cleanup():
    events = list(HaltRunner(ScriptedBackend()).stream(
        MultipleChoiceTask("12-5?", {"A": "5", "C": "7"}), FullReasoning()))
    last_commit = [e for e in events if e.kind == EventKind.TOKEN_COMMITTED][-1]
    assert events[-1].prefix == last_commit.prefix


def test_probe_seeds_depend_on_explicit_seed_and_work_not_run_uuid():
    from dataclasses import replace

    from halt.runtime.engine import probe_random_seed
    from halt.types import PrefixRef, ProbeAnswer, WorkLimit
    left = ProbeAnswer(request_id="run-one:probe:1",
        prefix=PrefixRef("run-one", "main", 5, "same-token-prefix", "model-v1"),
        temperature=0.6, max_work=WorkLimit(generated_tokens=8))
    right = replace(left, request_id="run-two:probe:1", prefix=replace(left.prefix, run_id="run-two"))
    assert probe_random_seed(19, left, 1) == probe_random_seed(19, right, 1)
    assert probe_random_seed(19, left, 1) != probe_random_seed(20, right, 1)
    assert probe_random_seed(19, left, 1) != probe_random_seed(19, right, 2)
