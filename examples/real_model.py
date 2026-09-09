"""Pinned real-model mechanics example; use --local-files-only for no network.

python examples/real_model.py --cache-dir .model-cache --local-files-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from halt import Budget, HaltRunner, MultipleChoiceTask
from halt.backends import TransformersBackend
from halt.methods.baselines import FixedReasoningBudget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=".model-cache")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="runs/real_model.json")
    parser.add_argument("--reasoning-tokens", type=int, default=16)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    backend = TransformersBackend.from_pretrained(
        cache_dir=args.cache_dir, local_files_only=args.local_files_only,
    )
    runner = HaltRunner(backend)
    result = runner.run(
        MultipleChoiceTask("What is 2 + 2?", {"A": "3", "B": "4", "C": "5"}),
        FixedReasoningBudget(tokens=args.reasoning_tokens),
        Budget(max_reasoning_tokens=args.reasoning_tokens, max_answer_tokens=48,
               max_probe_output_tokens=0, max_total_generated_tokens=args.reasoning_tokens + 48),
        seed=19,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result.to_json(), encoding="utf-8")
    print(json.dumps({"status": result.status, "answer": result.answer,
                      "stop": result.stop.reason, "output": str(destination)}))


if __name__ == "__main__":
    main()
