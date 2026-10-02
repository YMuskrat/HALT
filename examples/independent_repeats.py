"""Offline mechanics demonstration; no claim of model accuracy or savings."""
import argparse
import json
from pathlib import Path

from halt.config import load_config
from halt.evaluation.repeats import evaluate_repeats, summarize_repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/independent-repeats")
    args = parser.parse_args()
    config = load_config(Path(__file__).resolve().parents[1] / "configs/independent_repeats.json")
    summaries = evaluate_repeats(config, args.output, seeds=[11, 23, 37])
    report = summarize_repeats(summaries, samples=200, seed=7)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
