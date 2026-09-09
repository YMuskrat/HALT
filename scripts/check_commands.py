"""Exercise the documented offline CLI commands and separately installed plugin."""
import subprocess
import sys

commands = [
    ["demo", "--backend", "scripted"],
    ["demo", "--backend", "scripted", "--scenario", "incomplete"],
    ["methods", "list"], ["methods", "inspect", "halt_cot"],
    ["methods", "check", "example_stopper"],
    ["doctor", "--config", "configs/halt_cot_scripted_demo.json"],
    ["doctor", "--config", "configs/halt_cot_qwen3_demo.yaml"],
    ["run", "--config", "configs/halt_cot_scripted_demo.json", "--output", "runs/demo.json"],
    ["benchmark", "--config", "configs/benchmark_mcq.yaml", "--output-dir", "runs/mcq-final"],
    ["calibrate", "--config", "configs/calibrate_halt_cot.yaml", "--output", "runs/calibration.json"],
    ["replay", "--trace", "tests/fixtures/example_trace.jsonl", "--method", "halt_cot"],
]
for command in commands:
    completed = subprocess.run([sys.executable, "-m", "halt", *command], text=True, capture_output=True)
    # A demo succeeds by producing the requested outcome, including deliberate incompletion.
    expected = 0
    if completed.returncode != expected:
        print(completed.stdout, completed.stderr)
        raise SystemExit(f"Failed: {command}; expected {expected}, got {completed.returncode}")
    print("PASS", " ".join(command), flush=True)
