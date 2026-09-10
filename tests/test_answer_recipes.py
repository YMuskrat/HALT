"""Recipe behavior is driven by explicit settings, independent of controller names."""
from types import SimpleNamespace

import pytest

from halt.backends.transformers import TransformersBackend, TransformersRun
from halt.errors import CapabilityError, ConfigurationError
from halt.profiles import Qwen3ThinkingProfile
from halt.profiles.recipes import AnswerRecipe, recipe_catalog
from halt.runtime.budgets import UsageLedger
from halt.types import Budget, MethodSpec


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(token) for token in ids)


def branch_fixture(recipes):
    run = TransformersRun.__new__(TransformersRun)
    run.backend = SimpleNamespace(profile=Qwen3ThinkingProfile(recipes=recipe_catalog(recipes)),
                                  tokenizer=Tokenizer())
    run.context = SimpleNamespace(run_id="fixture")
    run.ledger = UsageLedger("fixture", Budget())
    run.ids = [1, 2, *map(ord, "Work. Pause")]
    run.prompt_length = 2
    return run


def test_custom_branch_rolls_back_only_exact_assistant_suffix_and_accounts_insertion():
    run = branch_fixture({"original_trial_v1": {"suffix": " Answer:", "close_reasoning": False,
                                              "remove_suffix": "Pause", "stop_at_reasoning_end": True}})
    before = list(run.ids)
    branch = run._branch("original_trial_v1", "probe")
    assert branch == [1, 2, *map(ord, "Work.  Answer:")]
    assert run.ids == before
    assert run.ledger.snapshot().forced_context_tokens == len(" Answer:")
    run.ids.append(ord("!"))
    with pytest.raises(ValueError, match="exact terminal"):
        run._branch("original_trial_v1", "probe")
    run.ids = list(map(ord, "Pause"))
    run.prompt_length = len(run.ids)
    with pytest.raises(ValueError, match="exact terminal"):
        run._branch("original_trial_v1", "probe")


def test_trial_refuses_text_suffix_that_is_not_an_exact_token_suffix():
    run = branch_fixture({"trial_v1": {"remove_suffix": "Pause"}})
    run.backend.tokenizer.encode = lambda text, **kwargs: [999]
    with pytest.raises(CapabilityError, match="exact removable token"):
        run._branch("trial_v1", "probe")


def test_custom_answer_recipe_closes_once_and_can_score_its_own_cue():
    profile = Qwen3ThinkingProfile(recipes=recipe_catalog({"original_answer_v1": {"suffix": " Result:"}}))
    tokenizer = Tokenizer()
    assert profile.transition(tokenizer, [], "original_answer_v1") == [151668, *map(ord, " Result:")]
    assert 151668 not in profile.transition(tokenizer, [151668], "original_answer_v1")
    assert profile.candidate_tokens(tokenizer, "A", "original_answer_v1") == list(map(ord, " A"))
    backend = TransformersBackend.__new__(TransformersBackend)
    backend.profile = profile
    backend.validate(None, MethodSpec("original", finalization_recipe="original_answer_v1"))


def test_main_finalization_refuses_trial_only_recipe():
    backend = TransformersBackend.__new__(TransformersBackend)
    backend.profile = Qwen3ThinkingProfile()
    with pytest.raises(CapabilityError, match="Main answer"):
        backend.validate(None, MethodSpec("original", finalization_recipe="deer_qwen3_greedy_v1"))


def test_aliases_preserve_recorded_recipe_behavior():
    recipes = recipe_catalog()
    assert recipes["candidate_scoring_v1"] == recipes["halt_cot_qwen_common_v1"]
    assert recipes["boxed_answer_v1"] == recipes["answer_convergence_qwen_common_v1"]
    trial = recipes["deer_qwen3_greedy_v1"]
    assert trial.suffix == "\n**Final Answer**\n\\boxed"
    assert trial.remove_suffix == "Wait" and trial.stop_at_reasoning_end
    assert trial.score_generated_tokens and not trial.close_reasoning
    assert trial.reconstruct_answer("{4}") == "\\boxed{4}"
    assert recipes["boxed_argument_v1"].reconstruct_answer("4}") == "\\boxed{4}"


@pytest.mark.parametrize("recipes", [
    {"halt_default_v1": {}}, {"new_v1": {"typo": True}},
    {"new_v1": {"close_reasoning": "false"}},
    {"new_v1": {"stop_at_reasoning_end": True}},
])
def test_invalid_recipe_fails_before_optional_model_imports(recipes):
    with pytest.raises(ConfigurationError):
        TransformersBackend.from_pretrained(answer_recipes=recipes)


def test_recipe_catalog_does_not_mutate_defaults():
    recipes = recipe_catalog({"original_v1": AnswerRecipe(suffix=" Answer:")})
    assert "original_v1" in recipes and "original_v1" not in recipe_catalog()
