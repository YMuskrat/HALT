# Independent repeats and frozen benchmark inputs

Install the checkout with `python -m pip install -e ".[dev]"`, then run:

```sh
python examples/independent_repeats.py --output runs/repeats-demo
```

This runs three **scripted** repeats with separately initialized adaptive
sessions. It is a mechanics demonstration, not a benchmark of model quality,
latency, or token savings. Identical scripted outcomes can give a degenerate
interval; that is not evidence of certainty about a real model.

The Python API is `evaluate_repeats(config, output_dir, seeds=[...])` in
`halt.evaluation.repeats`. Each repeat has a fresh backend, unique session ID,
and checkpoint directory. It retains the dataset's original order. To use a
custom backend, pass a zero-argument `backend_factory` that constructs a fresh
instance each time. Each method still gets its own session through the evaluator.

Resume uses the same output directory and seed list. Configuration, dataset
bytes, source/environment identity, and seed-list changes are rejected; create
a new output directory for a changed experiment. Concurrent writers to the same
output directory are not supported. Preserve interrupted manifests for inspection
rather than deleting them to force a resume.

`summarize_repeats(summaries)` weights repeats equally and bootstraps whole-repeat
paired accuracy effects. It requires matching experiments/methods, distinct seeds,
and fully paired observations for an interval. Failed answers remain failures.
At least two repeats are required for an interval, but two is usually weak evidence.
Different seeds alone do not prove statistical independence: initialize all
adaptive state independently and inspect the backend's RNG behavior.

## Freeze before an expensive run

```sh
python -m halt.evaluation.freeze configs/independent_repeats.json benchmark-lock.json
python -m halt.evaluation.freeze configs/independent_repeats.json benchmark-lock.json --verify
```

The lock records the full configuration, dataset bytes hash, selected item IDs,
and source/environment identity. It does not contain benchmark results. Existing
locks cannot be overwritten. Transformers configurations need immutable 40-character
model and tokenizer revisions; validation does not load model weights. The lock
is environment-specific and includes resolved paths: a transfer to another machine
requires a newly reviewed lock rather than silently ignoring identity differences.

For a real study, freeze the dataset/split, methods, model revisions, seeds, and
budget before measuring. Keep calibration data separate, retain failures, and
publish the hardware and uncertainty assumptions with results. Review inference
cost before replacing the scripted backend with a real model.
