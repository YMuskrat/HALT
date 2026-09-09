# example_stopper

This independently packaged example stops at a text boundary. It is original demonstration code with no paper attribution.

Install HALT, then run from this directory:

```console
python -m pip install -e .
python -m pytest tests
halt methods inspect example_stopper
halt doctor --config config.json
halt run --config config.json --output run.json
```

Replace the decision rule, supply source citation and immutable revision, and add source-derived fixtures. Method code receives inference-only observations and requests typed probes; it must not call a model, manage caches, or access reference labels. API version 1 requires reset, observe, decide and a MethodSpec. Keep decide deterministic for unchanged state. No author review is implied by this scaffold.
