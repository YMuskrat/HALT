"""Original demonstration plugin; no published-method fidelity is claimed."""
from dataclasses import dataclass

from halt.errors import ConfigurationError
from halt.methods.base import BaseMethod
from halt.types import Continue, EventKind, Finalize, MethodSpec


@dataclass
class ExampleStopper(BaseMethod):
    """Stop after a configured number of observed boundaries."""

    boundaries: int = 1
    spec = MethodSpec("example_stopper", variant="author_example", api_version="1",
                      source_relationship="original_demo", verification=("plugin_fixture",))

    def __post_init__(self):
        if type(self.boundaries) is not int or self.boundaries < 1:
            raise ConfigurationError("boundaries must be a positive integer")

    def reset(self, context):
        super().reset(context)
        self._boundaries = 0

    def observe(self, event):
        if event.kind == EventKind.STEP_BOUNDARY:
            self._boundaries += 1

    def decide(self):
        return Finalize("example_stopper_boundary") if self._boundaries >= self.boundaries else Continue()
