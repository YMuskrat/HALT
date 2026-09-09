"""Real Qwen3 mechanics comparison with immutable revisions and full ledgers.

This is a one-question mechanics check, not an accuracy/savings benchmark.
All methods use the same model, task, seed, sampling and resource budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from halt import Budget, HaltRunner, MultipleChoiceTask
from halt.backends import TransformersBackend
from halt.methods import AnswerConvergence, HaltCoT, ThinkBrake
from halt.methods.baselines import FixedReasoningBudget, FullReasoning, ImmediateAnswer
from halt.types import to_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=".model-cache")
    parser.add_argument("--output", default="runs/backend_smoke.json")
    parser.add_argument("--methods", nargs="*", default=["immediate_answer", "fixed_reasoning_budget",
                        "full_reasoning", "halt_cot", "thinkbrake", "answer_convergence"])
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    backend = TransformersBackend.from_pretrained(cache_dir=args.cache_dir, local_files_only=True)
    task = MultipleChoiceTask("What is 2 + 2? Keep reasoning brief.",
                              {"A": "3", "B": "4", "C": "5", "D": "6"})
    budget = Budget(max_reasoning_tokens=128, max_answer_tokens=48,
                    max_probe_output_tokens=128, max_total_generated_tokens=304,
                    max_probe_calls=8)
    methods = {
        "immediate_answer": ImmediateAnswer(),
        "fixed_reasoning_budget": FixedReasoningBudget(16),
        "full_reasoning": FullReasoning(),
        "halt_cot": HaltCoT(threshold=2.1, consecutive=1),
        "thinkbrake": ThinkBrake(threshold=100.0),
        "answer_convergence": AnswerConvergence(consecutive=2, max_probe_tokens=16),
    }
    runner, results = HaltRunner(backend), []
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name in args.methods:
        result = runner.run(task, methods[name], budget, seed=19)
        results.append(result.to_dict())
        destination.write_text(json.dumps({
            "schema_version": "1.0", "kind": "real_model_mechanics",
            "label_visibility": "Evaluation checks B after each run; no labels enter method context",
            "expected_answer": "B", "task": task.visible(), "backend": to_data(backend.info),
            "results": results,
        }, indent=2), encoding="utf-8")
        print(json.dumps({"method": name, "status": result.status, "answer": result.answer,
            "stop": result.stop.reason, "reasoning_tokens": result.usage.reasoning_tokens,
            "answer_tokens": result.usage.answer_tokens,
            "probe_tokens": result.usage.probe_output_tokens,
            "probe_calls": result.usage.probe_calls, "input_tokens": result.usage.input_tokens,
            "seconds": result.durations["end_to_end"], "error": result.error}), flush=True)


if __name__ == "__main__":
    main()
