"""Build local artifacts, install them in a clean venv, and verify CLI/plugin imports."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from halt.provenance import software_identity

results = []


def portable_log(value: str) -> str:
    """Keep verification logs shareable without machine-specific personal paths."""
    roots = [(Path.cwd(), "<workspace>"), (Path(tempfile.gettempdir()), "<temp>"),
             (Path.home(), "<home>")]
    for root, label in sorted(roots, key=lambda item: len(str(item[0])), reverse=True):
        for spelling in (str(root).replace("\\", "\\\\"), str(root), root.as_posix()):
            value = re.sub(re.escape(spelling), label, value, flags=re.IGNORECASE)
    return value


def check(arguments: list[str]) -> None:
    started = time.monotonic()
    result = subprocess.run(arguments, capture_output=True, text=True)
    results.append({"command": [portable_log(argument) for argument in arguments],
                    "exit_code": result.returncode,
                    "seconds": time.monotonic() - started,
                    "stdout": portable_log(result.stdout[-5000:]),
                    "stderr": portable_log(result.stderr[-3000:])})
    print(("PASS" if result.returncode == 0 else "FAIL"), " ".join(arguments), flush=True)
    if result.returncode:
        print(result.stdout, result.stderr)
        raise SystemExit(result.returncode)


check([sys.executable, "-m", "build", "--no-isolation"])
artifacts = list(Path("dist").glob("*.whl")) + list(Path("dist").glob("*.tar.gz"))
check([sys.executable, "-m", "twine", "check", *[str(p) for p in artifacts]])
check([sys.executable, "-m", "pip", "wheel", "examples/external_plugin", "--no-build-isolation",
       "--no-deps", "--wheel-dir", "dist/plugins"])
destination = Path(".wheel-venv")
check([sys.executable, "-m", "venv", str(destination)])
interpreter = destination / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
wheel = next(Path("dist").glob("halt_reasoning-*.whl"))
plugin = next(Path("dist/plugins").glob("*.whl"))
check([str(interpreter), "-m", "pip", "install", "--no-index", "--force-reinstall", str(wheel), str(plugin)])
check([str(interpreter), "-c", "import halt,sys; from halt.registry import MethodRegistry; "
       "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
       "assert MethodRegistry().inspect('halt_cot')['method_card']['source_commit']; print(halt.__file__)"])
check([str(interpreter), "-m", "halt", "demo", "--backend", "scripted"])
check([str(interpreter), "-m", "halt", "methods", "check", "example_stopper"])
report = {"schema_version": "1.0", "python": sys.version, **software_identity(), "checks": results,
          "scope": "Local wheel/sdist build and clean wheel install; no public publishing."}
Path("docs/verification").mkdir(parents=True, exist_ok=True)
Path("docs/verification/release_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
