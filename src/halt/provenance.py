"""Content revisions work in a checkout, installed wheel or unpacked source archive."""
from importlib import metadata
from pathlib import Path
from typing import Any

from halt.types import stable_hash


def software_identity() -> dict[str, Any]:
    root = Path(__file__).parent
    source_hash = stable_hash({str(path.relative_to(root)).replace("\\", "/"):
        stable_hash(path.read_text(encoding="utf-8")) for path in sorted(root.rglob("*.py"))})
    plugins = []
    for entry in metadata.entry_points(group="halt.methods"):
        distribution = entry.dist
        plugins.append({"method_id": entry.name, "target": entry.value,
                        "distribution": distribution.name if distribution else None,
                        "version": distribution.version if distribution else None})
    return {"halt_source_sha256": source_hash, "plugins": sorted(plugins, key=lambda x: x["method_id"] or "")}
