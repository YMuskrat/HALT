"""Analyze saved measurements with the standard library; no model execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from halt.evaluation import compare, load_results, render_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Run directory or results CSV/JSONL")
    parser.add_argument("--baseline", default="full_reasoning")
    parser.add_argument("--output", type=Path, help="Optional custom comparison JSON path")
    args = parser.parse_args()
    rows = load_results(args.results)
    summary = compare(rows, baseline=args.baseline)
    print(render_comparison(summary))

    # Example custom analysis: find questions an early stop changed from correct to wrong.
    baseline = {(row["question_id"], row["seed"]): row for row in rows
                if row["method_id"] == args.baseline}
    regressions = []
    for row in rows:
        reference = baseline.get((row["question_id"], row["seed"]))
        if row["method_id"] != args.baseline and reference and reference["correct"] and not row["correct"]:
            regressions.append({key: row[key] for key in (
                "question_id", "seed", "method_id", "answer", "reference", "status", "stop_reason",
                "total_generated_tokens", "elapsed_seconds")})
    print(f"\nCorrect-to-wrong paired observations: {len(regressions)}")
    for row in regressions:
        print(f"  {row['question_id']} / seed {row['seed']} / {row['method_id']}: "
              f"{row['status']}, {row['stop_reason']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"comparison": summary, "correct_to_wrong": regressions},
                                          indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
