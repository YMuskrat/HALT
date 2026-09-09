"""REFRAIN Algorithm 2 fixtures: per-arm windows are deliberately not global windows."""
import math

import pytest

from halt.errors import ConfigurationError
from halt.session import RefrainSession


def complete(session, run_id, length=100, likelihood=0.9):
    arm = session.acquire(run_id)
    result = session.update(run_id, answer_log_probs=(math.log(likelihood),),
                            output_tokens=length, completed=True)
    return arm, result


def test_initialization_reward_and_prior_running_length_mean():
    session = RefrainSession((0.6, 0.8), length_penalty=0.1)
    arm, first = complete(session, "one", length=100)
    assert arm == 0.6
    assert first["reward"] == pytest.approx(0.89)  # 0.9 - 0.0001*100
    arm, second = complete(session, "two", length=50)
    assert arm == 0.8
    assert second["prior_mean_output_tokens"] == 100
    assert second["reward"] == pytest.approx(0.85)  # 0.9 - 0.1*(50/100)


def test_per_arm_window_and_ucb_tie_order():
    session = RefrainSession((0.6, 0.8), window_size=1, exploration=0.1, length_penalty=0.01)
    complete(session, "one", likelihood=0.99)
    complete(session, "two", likelihood=0.2)
    complete(session, "three", likelihood=0.99)
    state = session.to_dict()
    assert len(state["buffers"][0]) == 1
    assert len(state["buffers"][1]) == 1  # Global latest-1 would evict this arm.
    assert state["completed"]["three"]["threshold"] == 0.6
    assert state["window_scope"] == "per_arm"


def test_save_load_preserves_next_selection_and_exactly_once_update(tmp_path):
    session = RefrainSession((0.6, 0.8), seed=7)
    complete(session, "one")
    complete(session, "two")
    target = tmp_path / "session.json"
    session.save(target)
    restored = RefrainSession.load(target)
    assert restored.to_dict() == session.to_dict()
    assert restored.acquire("three") == session.acquire("three")
    for item in (restored, session):
        item.abort("three")
    before = restored.to_dict()
    restored.acquire("one")
    repeated = restored.update("one", answer_log_probs=(0.0,), output_tokens=500, completed=True)
    assert not repeated["updated"]
    assert restored.to_dict()["completed"] == before["completed"]
    assert restored.to_dict()["buffers"] == before["buffers"]


def test_concurrency_abort_and_invalid_reward_are_explicit():
    session = RefrainSession()
    session.acquire("active")
    with pytest.raises(ConfigurationError, match="in use"):
        session.acquire("second")
    with pytest.raises(ConfigurationError, match="completed request boundaries"):
        session.to_dict()
    session.abort("wrong")
    with pytest.raises(ConfigurationError, match="in use"):
        session.acquire("second")
    session.abort("active")
    session.acquire("invalid")
    assert not session.update("invalid", answer_log_probs=(), output_tokens=1, completed=True)["updated"]
    assert session.to_dict()["completed"] == {}
    assert session.acquire("valid") == 0.6
    session.abort("valid")


def test_failed_run_does_not_update_or_poison_cold_start():
    session = RefrainSession()
    session.acquire("failed")
    record = session.update("failed", answer_log_probs=(-0.1,), output_tokens=50, completed=False)
    assert record == {"updated": False, "reason": "run_not_completed"}
    assert session.acquire("retry") == 0.6
    session.abort("retry")


def test_session_rejects_corrupted_window_definition():
    state = RefrainSession().to_dict()
    state["window_scope"] = "global"
    with pytest.raises(ConfigurationError):
        RefrainSession.from_dict(state)
