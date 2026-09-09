import json

import pytest

from halt import Budget, CancellationToken, HaltRunner, MultipleChoiceTask, NumericTask
from halt.backends import ScriptedBackend
from halt.backends.replay import ReplayBackend
from halt.errors import CapabilityError, ConfigurationError, ReplayUnavailable
from halt.methods.base import BaseMethod
from halt.methods.baselines import FixedReasoningBudget, FullReasoning, ImmediateAnswer
from halt.runtime.boundaries import Segmenter
from halt.signals import entropy, softmax
from halt.types import (
    Abstain,
    BoundaryPolicy,
    Continue,
    EventKind,
    MethodSpec,
    ProbeAnswer,
    RequestSignals,
    ReturnAnswer,
    RunStatus,
    WorkLimit,
)

TASK = MultipleChoiceTask("12-5?", {"A": "5", "B": "6", "C": "7"})


def run(backend=None, method=None, **kwargs):
    return HaltRunner(backend or ScriptedBackend()).run(TASK, method or FullReasoning(), **kwargs)


def test_lightweight_import():
    import subprocess
    import sys
    code = "import halt,sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_natural_answer_and_ledger():
    backend = ScriptedBackend()
    result = run(backend)
    assert result.answer == "C" and result.status == RunStatus.COMPLETED
    assert result.stop.reason == "natural_reasoning_end"
    assert result.usage.reasoning_tokens == 12
    assert result.usage.answer_tokens == 2
    assert result.usage.total_generated_tokens == 14
    assert result.usage.total_generated_tokens == sum(op.generated_tokens for op in result.usage.operations)
    assert result.usage.forced_context_tokens == 0
    assert backend.opened_runs[-1].closed
    assert json.loads(result.to_json())["schema_version"] == "1.0"


@pytest.mark.parametrize("limit", [0, 1, 4])
def test_reasoning_budget_reserves_final_answer(limit):
    result = run(budget=Budget(max_reasoning_tokens=limit, max_answer_tokens=2,
                              max_total_generated_tokens=limit+2))
    assert result.answer == "C"
    assert result.stop.reason == "reasoning_budget"
    assert result.usage.reasoning_tokens == limit
    assert result.usage.forced_context_tokens == 1


def test_answer_limit_does_not_report_success():
    result = run(ScriptedBackend(answer_eos=False), ImmediateAnswer(), budget=Budget(max_answer_tokens=2))
    assert result.status == "answer_incomplete"
    assert result.partial_answer == "C "
    assert result.stop.reason == "immediate_answer"


def test_eos_is_not_reasoning_end():
    result = run(ScriptedBackend(reasoning=[], natural_end=False))
    assert result.status == "answer_incomplete"
    assert result.stop.reason == "eos_before_answer"
    assert result.usage.answer_tokens == 0


def test_split_reasoning_end():
    result = run(ScriptedBackend(reasoning=["x", "</thi", "nk>"]))
    assert result.stop.reason == "natural_reasoning_end"
    assert result.usage.reasoning_tokens == 3
    assert result.usage.forced_context_tokens == 0


class Prober(BaseMethod):
    spec = MethodSpec("prober", can_return_answer=True)

    def reset(self, context):
        super().reset(context)
        self.decision = Continue()
        self.count = 0

    def observe(self, event):
        if event.kind == EventKind.STEP_BOUNDARY and not self.count:
            self.count += 1
            self.decision = RequestSignals((ProbeAnswer(request_id="p", prefix=event.prefix,
                                                        max_work=WorkLimit(generated_tokens=4)),))
        if event.kind == EventKind.PROBE_COMPLETED:
            self.decision = ReturnAnswer(event.payload.value.answer, "p")
        if event.kind in {EventKind.PROBE_FAILED, EventKind.PROBE_DENIED}:
            self.decision = Continue()

    def decide(self):
        return self.decision


def test_reused_answer_is_counted_once():
    result = run(method=Prober())
    assert result.status == "completed" and result.answer == "C"
    assert result.usage.probe_output_tokens == 2
    assert result.usage.answer_tokens == 0
    assert result.usage.total_generated_tokens == 10
    assert result.usage.probe_calls == 1
    assert [op.role for op in result.usage.operations if op.phase == "probing"] == ["accepted"]


def test_denied_probe_does_not_consume_answer_reservation():
    result = run(method=Prober(), budget=Budget(max_total_generated_tokens=12, max_answer_tokens=4))
    assert result.answer == "C"
    assert result.usage.probe_calls == 0
    assert result.usage.probe_output_tokens == 0


def test_failed_probe_consumed_work_and_continues():
    result = run(ScriptedBackend(fail_probe=True), Prober())
    assert result.answer == "C" and result.stop.reason == "natural_reasoning_end"
    assert result.usage.probe_calls == 1
    assert any(op.status == "failed" for op in result.usage.operations)


def test_repeated_probe_cannot_loop():
    class Repeater(Prober):
        def observe(self, event):
            if event.kind == EventKind.PROBE_COMPLETED:
                return
            super().observe(event)
    result = run(method=Repeater())
    assert result.status == "method_error"
    assert result.usage.probe_calls == 1


def test_callback_exception_keeps_usage_and_closes():
    class Broken(BaseMethod):
        def observe(self, event):
            if event.kind == EventKind.TOKEN_COMMITTED:
                raise RuntimeError("plugin fixture")
    backend = ScriptedBackend()
    result = run(backend, Broken())
    assert result.status == "method_error"
    assert result.usage.reasoning_tokens == 1
    assert backend.opened_runs[-1].closed


def test_backend_error_keeps_partial_usage():
    backend = ScriptedBackend(fail_at_token=2)
    result = run(backend)
    assert result.status == "backend_error" and result.usage.reasoning_tokens == 2
    assert backend.opened_runs[-1].closed


def test_cancellation_and_stream_close():
    backend = ScriptedBackend()
    token = CancellationToken()
    stream = HaltRunner(backend).stream(TASK, FullReasoning(), cancellation=token)
    next(stream)
    token.cancel()
    result = list(stream)[-1].payload
    assert result.status == "cancelled" and backend.opened_runs[-1].closed
    stream = HaltRunner(backend).stream(TASK, FullReasoning())
    next(stream)
    stream.close()
    assert backend.opened_runs[-1].closed


def test_interleaved_runs_and_config_not_mutated():
    backend = ScriptedBackend()
    method = FixedReasoningBudget(2)
    runner = HaltRunner(backend)
    left, right = runner.stream(TASK, method), runner.stream(TASK, method)
    next(left)
    next(right)
    a, b = list(left)[-1].payload, list(right)[-1].payload
    assert a.run_id != b.run_id
    assert a.usage.reasoning_tokens == b.usage.reasoning_tokens == 2
    assert not hasattr(method, "_tokens")


def test_abstain():
    class Abstainer(BaseMethod):
        spec = MethodSpec("abstainer", can_abstain=True)
        def decide(self):
            return Abstain("uncertain")
    result = run(method=Abstainer(), budget=Budget(max_answer_tokens=0))
    assert result.status == "abstained"


def test_capability_and_work_preflight():
    backend = ScriptedBackend(capabilities=frozenset())
    with pytest.raises(CapabilityError, match="answer_transition"):
        run(backend)
    assert not backend.opened_runs
    with pytest.raises(ConfigurationError):
        run(budget=Budget(max_answer_tokens=0))
    result = run(budget=Budget(max_input_tokens=1))
    assert result.status == "budget_exhausted"


@pytest.mark.parametrize("kwargs", [{"max_reasoning_tokens": -1}, {"max_probe_calls": True},
                                  {"deadline_seconds": float("nan")},
                                  {"max_total_generated_tokens": 1}, {"cost_ceiling": 1}])
def test_invalid_budgets(kwargs):
    with pytest.raises(ConfigurationError):
        Budget(**kwargs)


def test_boundary_fragments():
    segmenter = Segmenter(BoundaryPolicy("blank_line"))
    assert segmenter.push("hello\n") == []
    assert segmenter.push("\nworld")[0].text == "hello\n\n"
    assert segmenter.push("") == []
    assert segmenter.push("\n\n")[0].text == "world\n\n"


def test_entropy_math():
    assert entropy([0.5, 0.5]) == 1
    assert entropy([1, 0]) == 0
    assert entropy([0.5, 0.5], "nats") == pytest.approx(0.6931471805599453)
    assert softmax([1000, 1000]) == (0.5, 0.5)
    with pytest.raises(ValueError):
        softmax([float("nan"), 1])
    with pytest.raises(ValueError):
        entropy([0.1, 0.1])


def test_numeric_values_are_not_conflated():
    task = NumericTask("number?")
    assert task.normalize("10") != task.normalize("1")
    assert task.normalize("1,000") == "1000"
    assert task.normalize("1,00") is None
    assert task.normalize("error 1") is None


def test_replay_and_missing_counterfactual(tmp_path):
    trace = tmp_path / "trace.jsonl"
    real = run(method=Prober(), capture="full", trace_path=trace)
    result = run(ReplayBackend(trace), Prober())
    assert result.answer == real.answer
    assert result.stop.reason == real.stop.reason
    assert result.provenance["timing_kind"] == "controller_only"
    with pytest.raises(ReplayUnavailable):
        run(ReplayBackend(trace), ImmediateAnswer())


def test_metadata_capture_redacts_text(tmp_path):
    trace = tmp_path / "metadata.jsonl"
    run(capture="metadata", trace_path=trace)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert rows[0]["payload"] == {"redacted": True}
    assert rows[-1]["payload"]["answer"] is None
    with pytest.raises(ReplayUnavailable):
        ReplayBackend(trace)
