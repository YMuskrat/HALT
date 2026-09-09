"""Incremental versioned segmentation; split delimiters work across token fragments."""
import re

from halt.errors import ConfigurationError
from halt.types import BoundaryPolicy, StepBoundary


class Segmenter:
    def __init__(self, policy: BoundaryPolicy) -> None:
        if policy.name not in {"blank_line", "newline", "sentence", "fixed_tokens", "token"}:
            raise ConfigurationError(f"unsupported boundary policy {policy.name!r}")
        if policy.version != "1" or policy.chunk_tokens <= 0:
            raise ConfigurationError("boundary version must be 1 and chunk_tokens positive")
        self.policy = policy
        self.buffer = ""
        self.tokens = 0
        self.index = 0

    def push(self, fragment: str) -> list[StepBoundary]:
        self.buffer += fragment
        self.tokens += 1
        pieces: list[str] = []
        if self.policy.name in {"fixed_tokens", "token"}:
            size = self.policy.chunk_tokens if self.policy.name == "fixed_tokens" else 1
            if self.tokens >= size:
                pieces.append(self.buffer)
                self.buffer = ""
                self.tokens = 0
        else:
            pattern = {"blank_line": r"\n\s*\n", "newline": r"\n", "sentence": r"[.!?]+"}[self.policy.name]
            while match := re.search(pattern, self.buffer):
                pieces.append(self.buffer[:match.end()])
                self.buffer = self.buffer[match.end():]
        result = []
        for piece in pieces:
            self.index += 1
            result.append(StepBoundary(piece, self.index, f"{self.policy.name}_v{self.policy.version}"))
        return result
