"""Profile checks are offline; model checks require explicit opt-in and cached weights."""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest

from halt.errors import CapabilityError, ConfigurationError
from halt.profiles import Qwen3ThinkingProfile
from halt.types import (
    Budget,
    CandidateScores,
    NextTokenFeatures,
    Phase,
    ProbeAnswer,
    RunContext,
    ScoreCandidates,
    ScoreNextToken,
    WorkLimit,
)


class ProfileTokenizer:
    chat_template = "native"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [151668] if text == "</think>" else [ord(c) for c in text]

    def apply_chat_template(self, messages: object, **kwargs: object) -> list[int]:
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": True}
        return [1, 2, 3]


def test_profile_native_template_and_single_closure() -> None:
    profile, tokenizer = Qwen3ThinkingProfile(), ProfileTokenizer()
    profile.validate(tokenizer, SimpleNamespace(model_type="qwen3"))
    assert profile.render(tokenizer, "Question") == [1, 2, 3]
    forced = profile.transition(tokenizer, [1], "halt_default_v1")
    assert forced.count(151668) == 1
    assert 151668 not in profile.transition(tokenizer, [1, 151668], "halt_default_v1")
    with pytest.raises(ValueError, match="Unsupported"):
        profile.transition(tokenizer, [1], "invented_recipe")


def test_profile_rejects_foreign_model_and_delimiter() -> None:
    profile, tokenizer = Qwen3ThinkingProfile(), ProfileTokenizer()
    with pytest.raises(ValueError, match="Qwen3"):
        profile.validate(tokenizer, SimpleNamespace(model_type="llama"))
    tokenizer.encode = lambda *args, **kwargs: [1, 2]  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="151668"):
        profile.validate(tokenizer, SimpleNamespace(model_type="qwen3"))


def test_revision_validation_is_before_optional_imports() -> None:
    from halt.backends.transformers import TransformersBackend

    with pytest.raises(ConfigurationError, match="immutable"):
        TransformersBackend.from_pretrained(revision="main")
    with pytest.raises(CapabilityError, match="profile"):
        TransformersBackend.from_pretrained(model_profile="unknown")


def test_user_quoted_delimiter_cannot_suppress_forced_closure() -> None:
    from halt.backends.transformers import TransformersRun
    from halt.runtime.budgets import UsageLedger
    run = TransformersRun.__new__(TransformersRun)
    run.backend = SimpleNamespace(profile=Qwen3ThinkingProfile(), tokenizer=ProfileTokenizer())
    run.context = SimpleNamespace(run_id="fixture")
    run.ledger = UsageLedger("fixture", Budget())
    run.closed = run.transitioned = False
    run.ids = [151668, 4, 5]
    run.prompt_length = 3
    run.transition("halt_default_v1")
    assert run.ids.count(151668) == 2  # One in user context, one newly forced in assistant context.


@pytest.fixture(scope="module")
def model_backend():  # type: ignore[no-untyped-def]
    if os.environ.get("HALT_RUN_MODEL_TESTS") != "1":
        pytest.skip("Set HALT_RUN_MODEL_TESTS=1 with pinned Qwen3 weights cached locally")
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from halt.backends.transformers import TransformersBackend

    torch.set_num_threads(4)
    return TransformersBackend.from_pretrained(
        cache_dir=os.environ.get("HALT_MODEL_CACHE", ".model-cache"),
        local_files_only=True,
    )


@pytest.mark.model
def test_real_probe_isolation_scores_and_forced_transition(model_backend):  # type: ignore[no-untyped-def]
    from halt.runtime.budgets import UsageLedger
    from halt.tasks import MultipleChoiceTask

    backend = model_backend
    task = MultipleChoiceTask("What is 2 + 2?", {"A": "3", "B": "4"})
    budget = Budget(max_reasoning_tokens=12, max_answer_tokens=32,
                    max_probe_output_tokens=32, max_total_generated_tokens=76)

    def open_run(name: str):  # type: ignore[no-untyped-def]
        context = RunContext(name, "test", "test", task.visible(), backend.info, 19, 23, budget)
        ledger = UsageLedger(name, budget)
        return backend.open(context, task, ledger), ledger

    main, ledger = open_run("probed")
    control, _ = open_run("control")
    try:
        before = main.prefix
        score = main.probe(ScoreCandidates(request_id="candidate", prefix=before,
            candidates=task.candidates, recipe="halt_cot_qwen_common_v1"), seed=33)
        assert isinstance(score.value, CandidateScores)
        assert all(math.isfinite(x) for x in score.value.log_scores)
        assert score.value.frame.prefix.identity != before.identity
        features = main.probe(ScoreNextToken(request_id="features", prefix=before,
                                            include_entropy=True), seed=1)
        assert isinstance(features.value, NextTokenFeatures)
        assert features.value.frame.prefix == before
        assert features.value.entropy is not None and features.value.entropy >= 0
        main.probe(ProbeAnswer(request_id="trial", prefix=before,
            max_work=WorkLimit(generated_tokens=8), temperature=0.6), seed=100)
        assert main.prefix == before
        assert [main.next_token(Phase.REASONING).token_id for _ in range(4)] == [
            control.next_token(Phase.REASONING).token_id for _ in range(4)
        ]
        count = main.ids.count(backend.profile.end_token_id)
        main.transition("halt_default_v1")
        assert main.ids.count(backend.profile.end_token_id) == count + 1
        main.transition("halt_default_v1")
        assert main.ids.count(backend.profile.end_token_id) == count + 1
        for _ in range(32):
            if main.next_token(Phase.ANSWERING).eos:
                break
        usage = ledger.snapshot()
        assert usage.reasoning_tokens == 4
        assert 1 <= usage.probe_output_tokens <= 8
        assert usage.forward_calls > 1
        assert usage.recomputed_prefix_tokens > 0
        assert usage.input_tokens == sum(op.input_tokens for op in usage.operations)
    finally:
        main.close()
        control.close()
    assert main.closed and not main.ids


@pytest.mark.model
def test_real_multitoken_candidates_fail_preflight(model_backend):  # type: ignore[no-untyped-def]
    with pytest.raises(CapabilityError, match="one token"):
        model_backend.candidate_ids(("alpha beta", "gamma delta"))
