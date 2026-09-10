"""Lazy built-in and packaging-entry-point method discovery."""
from __future__ import annotations

import importlib
import inspect
import json
import re
from dataclasses import dataclass
from importlib import metadata
from importlib.resources import files
from typing import Any

from halt.errors import ConfigurationError
from halt.types import PLUGIN_API_VERSION, MethodSpec, to_data

_BUILTINS = {
    "full_reasoning": ("halt.methods.baselines", "FullReasoning"),
    "fixed_reasoning_budget": ("halt.methods.baselines", "FixedReasoningBudget"),
    "immediate_answer": ("halt.methods.baselines", "ImmediateAnswer"),
    "halt_cot": ("halt.methods", "HaltCoT"),
    "thinkbrake": ("halt.methods", "ThinkBrake"),
    "answer_convergence": ("halt.methods", "AnswerConvergence"),
    "refrain": ("halt.methods", "Refrain"),
    "deer": ("halt.methods", "DEER"),
}


def validate_method_id(method_id: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", method_id):
        raise ConfigurationError(f"invalid method ID {method_id!r}; use lowercase snake_case")


@dataclass(frozen=True)
class MethodRegistration:
    method_id: str
    origin: str
    target: Any
    distribution_version: str | None = None

    def load(self) -> Any:
        if isinstance(self.target, tuple):
            module, name = self.target
            return getattr(importlib.import_module(module), name)
        return self.target.load()


class MethodRegistry:
    def __init__(self, *, entry_points: Any = None) -> None:
        self.entries = {
            name: MethodRegistration(name, "builtin", target)
            for name, target in _BUILTINS.items()
        }
        # Contribution cards are packaged with HALT. Only explicit local method
        # targets participate in registration; descriptive cards remain passive.
        cards = files("halt").joinpath("resources", "method_cards")
        for path in sorted(cards.iterdir(), key=lambda item: item.name):
            if not path.name.endswith(".json"):
                continue
            card = json.loads(path.read_text(encoding="utf-8"))
            target = card.get("implementation")
            if target is None:
                continue
            method_id = card.get("method_id", "")
            validate_method_id(method_id)
            if (path.name != f"{method_id}.json" or not isinstance(target, str)
                    or not re.fullmatch(rf"halt\.methods\.{re.escape(method_id)}:[A-Za-z][A-Za-z0-9_]*", target)):
                raise ConfigurationError(f"invalid local implementation target in {path.name}")
            if method_id in self.entries:
                raise ConfigurationError(f"method ID collision: {method_id!r} is registered more than once")
            self.entries[method_id] = MethodRegistration(method_id, "builtin", tuple(target.split(":")))
        discovered = metadata.entry_points(group="halt.methods") if entry_points is None else entry_points
        for entry in discovered:
            validate_method_id(entry.name)
            if entry.name in self.entries:
                raise ConfigurationError(f"method ID collision: {entry.name!r} is registered more than once")
            distribution = getattr(entry, "dist", None)
            self.entries[entry.name] = MethodRegistration(
                entry.name, getattr(distribution, "name", None) or "external_plugin", entry,
                getattr(distribution, "version", None),
            )

    def list(self) -> list[dict[str, Any]]:
        # Metadata discovery deliberately does not import plugin modules.
        return [{"method_id": value.method_id, "origin": value.origin,
                 "distribution_version": value.distribution_version}
                for _, value in sorted(self.entries.items())]

    def create(self, method_id: str, parameters: dict[str, Any] | None = None) -> Any:
        if method_id not in self.entries:
            raise ConfigurationError(f"unknown method {method_id!r}; use 'halt methods list'")
        factory = self.entries[method_id].load()
        try:
            inspect.signature(factory).bind(**(parameters or {}))
            instance = factory(**(parameters or {}))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid parameters for {method_id}: {exc}") from exc
        spec = getattr(instance, "spec", None)
        if not isinstance(spec, MethodSpec):
            raise ConfigurationError(f"plugin {method_id} must expose a MethodSpec")
        if spec.method_id != method_id:
            raise ConfigurationError(f"plugin registered as {method_id!r} declares {spec.method_id!r}")
        if spec.api_version != PLUGIN_API_VERSION:
            raise ConfigurationError(
                f"plugin {method_id} API {spec.api_version!r} is incompatible with HALT API {PLUGIN_API_VERSION}"
            )
        if any(not callable(getattr(instance, name, None)) for name in ("reset", "observe", "decide")):
            raise ConfigurationError(f"plugin {method_id} must implement reset, observe and decide")
        return instance

    def inspect(self, method_id: str) -> dict[str, Any]:
        instance = self.create(method_id)
        card_path = files("halt").joinpath("resources", "method_cards", f"{method_id}.json")
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.is_file() else None
        if card is None and self.entries[method_id].origin != "builtin":
            # Scaffolds ship metadata beside their implementation. Reading a card
            # occurs only during explicit inspect, after create imports the plugin.
            package = type(instance).__module__.split(".")[0]
            try:
                plugin_card = files(package).joinpath("method_card.json")
                if plugin_card.is_file():
                    card = json.loads(plugin_card.read_text(encoding="utf-8"))
            except (ModuleNotFoundError, TypeError):
                pass
        if card is not None and card.get("method_id") != method_id:
            raise ConfigurationError(f"method card does not describe {method_id!r}")
        return {"registration": next(x for x in self.list() if x["method_id"] == method_id),
                "spec": to_data(instance.spec),
                "method_card": card,
                "defaults": to_data(instance.configuration()) if hasattr(instance, "configuration") else {}}


def get_method(method_id: str, **parameters: Any) -> Any:
    return MethodRegistry().create(method_id, parameters)
