"""Hand-computed decision fixtures linked to the five primary-source audits."""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from halt.backends.scripted import ScriptedBackend
from halt.errors import ConfigurationError
from halt.methods import DEER, AnswerConvergence, HaltCoT, Refrain, ThinkBrake
from halt.methods.deer import trial_likelihood
from halt.types import (
    Budget,
    CandidateScores,
    Continue,
    EventKind,
    Finalize,
    NextTokenFeatures,
    Observation,
    ParsedAnswer,
    Phase,
    PrefixRef,
    ProbeFailure,
    ProbeResult,
    RequestSignals,
    ReturnAnswer,
    RunContext,
    ScoreFrame,
    StepBoundary,
    StepEmbeddings,
)


def context(run_id="fixture"):
    return RunContext(run_id, "test", "config", {"candidates": ("A", "B")},
                      ScriptedBackend().info, 0, 1, Budget())


def prefix(position=1, run_id="fixture"):
    return PrefixRef(run_id, "main", position, f"prefix-{position}", "scripted-v1")


def boundary(method, text="Done.\n", position=1):
    event = Observation(f"step-{position}", EventKind.STEP_BOUNDARY, 0.0, Phase.REASONING,
                        prefix(position), StepBoundary(text, position, method.spec.boundary.name))
    method.observe(event)
    return method.decide()


def deliver(method, value, position=1, valid=True, source_prefix=None):
    decision = method.decide()
    assert isinstance(decision, RequestSignals)
    request = decision.requests[0]
    method.observe(Observation("scheduled", EventKind.PROBE_SCHEDULED, 0.0, Phase.PROBING,
                               prefix(position), request))
    result = ProbeResult(request.request_id, source_prefix or prefix(position), value, valid)
    method.observe(Observation("completed", EventKind.PROBE_COMPLETED, 0.0, Phase.PROBING,
                               prefix(position), result))
    return method.decide()


def halt_scores(method, scores, position):
    boundary(method, position=position)
    return deliver(method, CandidateScores(("A", "B"), tuple(scores),
                   ScoreFrame(prefix(position))), position)


def test_halt_strict_equality_streak_reset_and_current_answer():
    # Author EntropyHaltingController: H < theta, not <= theta.
    method = HaltCoT(threshold=1.0, consecutive=2)
    method.reset(context())
    assert isinstance(halt_scores(method, (0.0, -20.0), 1), Continue)
    assert isinstance(halt_scores(method, (0.0, 0.0), 2), Continue)  # H = 1 bit resets.
    assert isinstance(halt_scores(method, (-20.0, 0.0), 3), Continue)
    decision = halt_scores(method, (-20.0, 0.0), 4)
    assert isinstance(decision, ReturnAnswer)
    assert decision.answer == "B"
    assert decision.diagnostics["streak"] == 2
    assert method.decide() == decision  # No decision-time mutation.


def test_halt_tie_uses_candidate_order_and_nats():
    method = HaltCoT(threshold=0.7, entropy_unit="nats", consecutive=1)
    method.reset(context())
    decision = halt_scores(method, (0.0, 0.0), 1)
    assert isinstance(decision, ReturnAnswer)
    assert decision.answer == "A"
    assert decision.diagnostics["candidate_entropy"] == pytest.approx(math.log(2))


def test_halt_invalid_scores_do_not_continue_a_streak():
    method = HaltCoT(threshold=0.6, consecutive=2)
    method.reset(context())
    halt_scores(method, (0.0, -20.0), 1)
    assert isinstance(halt_scores(method, (float("nan"), 0.0), 2), Continue)
    assert isinstance(halt_scores(method, (0.0, -20.0), 3), Continue)


def test_halt_accepts_declared_answer_branch_but_not_unrelated_frame():
    method = HaltCoT(threshold=0.6, consecutive=1)
    method.reset(context())
    request = boundary(method).requests[0]
    branch_prefix = replace(prefix(), trajectory_id=request.request_id, identity="forced-answer")
    assert isinstance(deliver(method, CandidateScores(("A", "B"), (0.0, -20.0),
                                                       ScoreFrame(branch_prefix))), ReturnAnswer)
    boundary(method, position=2)
    unrelated = replace(prefix(2), run_id="other-run")
    assert isinstance(deliver(method, CandidateScores(("A", "B"), (0.0, -20.0),
                   ScoreFrame(unrelated)), 2), Continue)


@pytest.mark.parametrize("kwargs", [
    {"threshold": float("nan")}, {"threshold": float("inf")}, {"consecutive": 0},
    {"consecutive": True}, {"entropy_unit": "unknown"}, {"scoring": "first_token_approx"},
])
def test_halt_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ConfigurationError):
        HaltCoT(**kwargs)


def test_halt_single_candidate_rejected():
    with pytest.raises(ConfigurationError, match="at least two"):
        HaltCoT().reset(replace(context(), task_visible={"candidates": ("A",)}))


def test_thinkbrake_exact_next_prefix_not_punctuation_sampling_scores():
    method = ThinkBrake(threshold=0.25)
    method.reset(context())
    boundary(method, position=2)
    # A confident score frame before the newline must be rejected.
    old_frame = NextTokenFeatures(ScoreFrame(prefix(1)), -0.1, -0.1)
    assert isinstance(deliver(method, old_frame, 2), Continue)
    boundary(method, position=3)
    current_frame = NextTokenFeatures(ScoreFrame(prefix(3)), -0.5, -0.75)
    decision = deliver(method, current_frame, 3)
    assert isinstance(decision, Finalize)  # Equality is a stop for ThinkBrake.
    assert decision.diagnostics["scored_prefix_position"] == 3


@pytest.mark.parametrize("top,end", [(-0.1, -10.0), (-0.1, float("nan")),
                                    (float("nan"), -0.1)])
def test_thinkbrake_margin_sign_and_invalid_scores(top, end):
    method = ThinkBrake()
    method.reset(context())
    boundary(method)
    assert isinstance(deliver(method, NextTokenFeatures(ScoreFrame(prefix()), top, end)), Continue)


def test_answer_consistency_invalid_history_and_disagreement():
    method = AnswerConvergence(consecutive=2)
    method.reset(context())
    observations = [("A", True), (None, False), ("A", True), ("B", True), ("B", True)]
    for position, (answer, valid) in enumerate(observations, 1):
        boundary(method, position=position)
        decision = deliver(method, ParsedAnswer(answer or "", answer, valid), position)
        assert isinstance(decision, ReturnAnswer if position == 5 else Continue)
    assert decision.answer == "B"


def test_probe_denial_resets_answer_consistency_without_repeat_request():
    method = AnswerConvergence(consecutive=2)
    method.reset(context())
    boundary(method)
    deliver(method, ParsedAnswer("A", "A", True))
    request = boundary(method, position=2).requests[0]
    method.observe(Observation("denied", EventKind.PROBE_DENIED, 0.0, Phase.PROBING,
        prefix(2), ProbeFailure(request.request_id, "budget reservation")))
    assert isinstance(method.decide(), Continue)
    boundary(method, position=3)
    assert isinstance(deliver(method, ParsedAnswer("A", "A", True), 3), Continue)


def test_refrain_prior_history_gate_and_no_self_similarity():
    method = Refrain(threshold=0.9)
    method.reset(context())
    # Reflective current step's provisional answer cannot satisfy its own gate.
    assert isinstance(boundary(method, "Wait, the answer is A.\n\n"), Continue)
    assert isinstance(boundary(method, "Wait, let me check.\n\n", 2), RequestSignals)
    vectors = StepEmbeddings(((1.0, 0.0), (0.0, 1.0)), "fixture", "1")
    assert isinstance(deliver(method, vectors, 2), Continue)  # Would wrongly stop if self included.
    boundary(method, "Recall that the answer is A.\n\n", 3)
    vectors = StepEmbeddings(((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)), "fixture", "1")
    decision = deliver(method, vectors, 3)
    assert isinstance(decision, Finalize)
    assert decision.diagnostics["maximum_previous_step_cosine"] == 1.0


def test_refrain_nonreflective_step_never_requests_semantic_stop():
    method = Refrain()
    method.reset(context())
    boundary(method, "The answer is A.\n\n")
    assert isinstance(boundary(method, "Computing a new value.\n\n", 2), Continue)


def test_refrain_adaptive_requires_session():
    with pytest.raises(ConfigurationError, match="explicit"):
        Refrain(adaptive=True).reset(context())


def test_deer_split_transition_distinct_trial_rule_and_separate_finalization():
    method = DEER(threshold=0.8)
    method.reset(context())
    assert isinstance(boundary(method, "Wa"), Continue)
    decision = boundary(method, "it", 2)
    assert isinstance(decision, RequestSignals)
    assert decision.requests[0].recipe == "deer_qwen3_greedy_v1"
    # First token excluded per actual source code; subsequent geometric mean = 0.9.
    trial = ParsedAnswer("A</think>", "A", True, True,
                         (math.log(0.01), math.log(0.9), math.log(0.9)), True)
    decision = deliver(method, trial, 2)
    assert isinstance(decision, Finalize)
    assert decision.diagnostics["trial_answer_reused"] is False


@pytest.mark.parametrize("valid,end,probs", [
    (True, False, (-0.01, -0.01)), (False, True, (-0.01, -0.01)),
    (True, True, (-0.01, float("nan"))), (True, True, ()),
])
def test_deer_missing_end_or_invalid_trial_does_not_stop(valid, end, probs):
    method = DEER(threshold=0.8)
    method.reset(context())
    boundary(method, "Wait")
    assert isinstance(deliver(method, ParsedAnswer("A", "A", valid, True, probs, end)), Continue)


def test_deer_geometric_arithmetic_and_strict_threshold():
    logs = (0.0, math.log(0.25), 0.0)
    assert trial_likelihood(logs) == pytest.approx(0.5)
    assert trial_likelihood(logs, "arithmetic") == pytest.approx(0.625)
    method = DEER(threshold=0.5)
    method.reset(context())
    boundary(method, "Wait")
    assert isinstance(deliver(method, ParsedAnswer("A", "A", True, True, logs, True)), Continue)


def test_interleaved_methods_keep_answer_histories_separate():
    one, two = AnswerConvergence(consecutive=2), AnswerConvergence(consecutive=2)
    one.reset(context())
    two.reset(context())
    boundary(one)
    deliver(one, ParsedAnswer("A", "A", True))
    boundary(two)
    assert isinstance(deliver(two, ParsedAnswer("B", "B", True)), Continue)
    boundary(one, position=2)
    assert isinstance(deliver(one, ParsedAnswer("A", "A", True), 2), ReturnAnswer)
    boundary(two, position=2)
    assert isinstance(deliver(two, ParsedAnswer("A", "A", True), 2), Continue)
