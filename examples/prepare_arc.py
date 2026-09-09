"""Export a pinned established MCQ subset and identity for HALT's JSONL evaluator."""
import argparse
import json
from pathlib import Path

from halt.evaluation.huggingface import ARC_AUDITED_REVISION, load_arc

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--revision", default=ARC_AUDITED_REVISION)
parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
parser.add_argument("--subset", choices=["ARC-Easy", "ARC-Challenge"], default="ARC-Easy")
parser.add_argument("--limit", type=int, default=50)
parser.add_argument("--output", default="runs/arc_validation.jsonl")
args = parser.parse_args()
dataset = load_arc(revision=args.revision, split=args.split, subset=args.subset, limit=args.limit,
                   cache_dir=".dataset-cache")
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as target:
    for item in dataset.items:
        visible = item.task.visible()
        target.write(json.dumps({"id": item.item_id, "question": visible["question"],
                                 "choices": visible["choices"], "answer": item.reference}) + "\n")
output.with_suffix(".manifest.json").write_text(json.dumps(dataset.identity(), indent=2), encoding="utf-8")
print(output)
