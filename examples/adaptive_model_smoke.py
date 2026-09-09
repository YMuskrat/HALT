"""Small REFRAIN session + DEER trial mechanics, not a research benchmark."""
from __future__ import annotations

import json
from pathlib import Path

from halt import Budget, HaltRunner, MultipleChoiceTask
from halt.backends import TransformersBackend
from halt.methods import Refrain
from halt.runtime.budgets import UsageLedger
from halt.session import RefrainSession
from halt.signals.embeddings import SentenceTransformerEmbeddings
from halt.types import Phase, ProbeAnswer, RunContext, WorkLimit, to_data


def main() -> None:
    import torch

    torch.set_num_threads(4)
    encoder = SentenceTransformerEmbeddings(cache_folder=".model-cache", local_files_only=True)
    embedding_result = encoder.embed(("The answer is four.", "The answer is four.", "An unrelated blue bird."))
    backend = TransformersBackend.from_pretrained(cache_dir=".model-cache", local_files_only=True,
                                                  embeddings=encoder)
    task = MultipleChoiceTask("What is 2 + 2?", {"A": "3", "B": "4", "C": "5"})
    session = RefrainSession((0.6, 0.8), seed=7)
    budget = Budget(max_reasoning_tokens=8, max_answer_tokens=32, max_probe_output_tokens=20,
                    max_total_generated_tokens=60)
    result = HaltRunner(backend).run(task, Refrain(), budget, session=session, seed=19)
    print("REFRAIN", result.status, result.answer, result.provenance["session_update"], flush=True)
    # This is an induced backend branch fixture: Wait is explicitly inserted as input.
    # It does not claim the model naturally generated a DEER transition on this task.
    context = RunContext("deer-real-fixture", "deer_fixture", "fixture", task.visible(),
                         backend.info, 19, 23, budget)
    ledger = UsageLedger(context.run_id, budget)
    handle = backend.open(context, task, ledger)
    try:
        extra = backend.tokenizer.encode("<think>Wait", add_special_tokens=False)
        handle.ids = handle._insert(handle.ids, extra, Phase.REASONING, context.run_id)
        before = handle.prefix
        request = ProbeAnswer(request_id="induced-deer-trial", prefix=before,
            recipe="deer_qwen3_greedy_v1", score_answer=True,
            max_work=WorkLimit(generated_tokens=20))
        ledger.begin_probe(20, None, None)
        trial = handle.probe(request, seed=23)
        ledger.end_probe()
        assert before == handle.prefix
        print("DEER trial", to_data(trial.value), flush=True)
    finally:
        handle.close()
    report = {"schema_version": "1.0", "kind": "real_model_component_smoke",
              "backend": to_data(backend.info), "embedding_fixture": to_data(embedding_result),
              "refrain_run": result.to_dict(), "session": session.to_dict(),
              "deer_induced_prefix_trial": to_data(trial), "deer_usage": to_data(ledger.snapshot()),
              "limitations": ["One easy item; REFRAIN need not trigger before reasoning budget.",
                              "DEER prefix explicitly induced; this is a backend branch fixture.",
                              "No efficiency or paper-reproduction conclusion."]}
    destination = Path("runs/adaptive_model_smoke.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
