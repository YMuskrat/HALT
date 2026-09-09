# Runnable configurations

All thresholds and budgets here are experimental mechanics settings, not recommendations or reproduced benchmark recipes. The `.yaml` examples use the JSON subset of YAML so they run with HALT's dependency-free core; general YAML needs the evaluation extra. JSON and TOML are also supported.

`scripted_demo.json` exercises candidate scoring offline. `halt_cot_qwen3_demo.yaml` loads the pinned real Qwen3 model, using short CPU budgets and greedy decoding for mechanics. `halt doctor --config ...` performs static inspection; add `--load-model` to verify the actual tokenizer and model. The default model cache is relative to the invocation directory.

`benchmark_mcq.yaml` compares baselines and research recipes on original toy multiple-choice questions. Scripted answers are predetermined and do not depend on labels; this is deliberately not a performance benchmark. `calibrate_halt_cot.yaml` searches a separate original toy calibration file and records an empirical artifact without statistical guarantees.

Dataset paths resolve relative to the configuration file. Output paths resolve relative to the invocation directory. Never put credentials into these public examples.
