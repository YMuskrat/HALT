"""Synchronize audited method metadata into wheel resources without importing models."""
import json
from pathlib import Path

from halt.registry import MethodRegistry
from halt.types import to_data

registry = MethodRegistry()
audited = {"halt_cot", "thinkbrake", "answer_convergence", "refrain", "deer"}
for path in Path("method_cards").glob("*.json"):
    card = json.loads(path.read_text(encoding="utf-8"))
    if card["method_id"] not in audited:
        continue
    method = registry.create(card["method_id"])
    card["implementation_status"] = "functional"
    card["requirements"] = to_data(method.spec.requirements)
    card["resolved_default_spec"] = to_data(method.spec)
    card["verification"] = [{"kind": "source_derived_component_fixtures",
                              "artifact": "tests/unit/test_research_methods.py",
                              "scope": "Hand-checked decision rules; no upstream execution or paper results reproduced."}]
    card["supported_combinations"] = [{"backend": "scripted", "model": "scripted-v1",
                                       "verification": "component_and_runtime_fixtures"}]
    if card["method_id"] in {"halt_cot", "thinkbrake", "answer_convergence"}:
        card["supported_combinations"].append({"backend": "transformers", "model": "Qwen/Qwen3-0.6B",
            "revision": "c1899de289a04d12100db370d81485cdf75e47ca", "profile": "qwen3_thinking",
            "verification": "one_item_cpu_mechanics", "evidence": "docs/backend_verification.md"})
    if card["method_id"] == "refrain":
        card["verification"].append({"kind": "session_fixtures",
                                     "artifact": "tests/unit/test_refrain_session.py"})
        card["verification"].append({"kind": "real_model_session_reward_and_encoder",
                                     "artifact": "docs/verification/adaptive_model_smoke.json",
                                     "scope": "One real budget-finalized run updated the session; no semantic-stop performance claim."})
    if card["method_id"] == "deer":
        card["verification"].append({"kind": "real_model_induced_branch_fixture",
                                     "artifact": "docs/verification/adaptive_model_smoke.json",
                                     "scope": "Wait prefix explicitly induced, real greedy trial and unchanged main prefix verified."})
    encoded = json.dumps(card, indent=2, ensure_ascii=False) + "\n"
    path.write_text(encoded, encoding="utf-8")
    resource = Path("src/halt/resources/method_cards") / path.name
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_text(encoded, encoding="utf-8")
print("Synchronized five source-pinned method cards.")
