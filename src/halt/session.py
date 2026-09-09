"""Explicit sequential REFRAIN controller, implementing paper Algorithm 2 per-arm windows."""
from __future__ import annotations

import json
import math
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from halt.errors import ConfigurationError


class RefrainSession:
    """Request-order SW-UCB state. A failed run never supplies a fictitious reward."""

    def __init__(self, thresholds: tuple[float, ...] = (0.60, 0.65, 0.70, 0.75, 0.80), *,
                 window_size: int = 32, exploration: float = 1.0,
                 length_penalty: float = 0.1, seed: int = 42,
                 session_id: str | None = None) -> None:
        self.thresholds = tuple(thresholds)
        if (not self.thresholds or len(set(self.thresholds)) != len(self.thresholds)
                or any(not math.isfinite(t) or not 0 <= t <= 1 for t in self.thresholds)):
            raise ConfigurationError("thresholds must be distinct finite values in [0, 1]")
        if type(window_size) is not int or window_size < 1:
            raise ConfigurationError("window_size must be a positive integer")
        if not math.isfinite(exploration) or exploration <= 0:
            raise ConfigurationError("exploration must be finite and positive")
        if not math.isfinite(length_penalty) or length_penalty <= 0:
            raise ConfigurationError("length_penalty must be finite and positive")
        if type(seed) is not int:
            raise ConfigurationError("seed must be an integer")
        self.window_size = window_size
        self.exploration = exploration
        self.length_penalty = length_penalty
        self.seed = seed
        self.session_id = session_id or uuid.uuid4().hex
        self._buffers: list[deque[float]] = [deque(maxlen=window_size) for _ in self.thresholds]
        self._completed: dict[str, dict[str, Any]] = {}
        self._request_order: list[str] = []
        self._total_length = 0
        self._active: tuple[str, int] | None = None
        self._lock = threading.RLock()

    def acquire(self, run_id: str) -> float:
        with self._lock:
            if self._active is not None:
                raise ConfigurationError("REFRAIN session is in use; concurrent or interleaved requests are unsupported")
            if not run_id:
                raise ConfigurationError("run_id must not be empty")
            if run_id in self._completed:
                arm = int(self._completed[run_id]["arm_index"])
            else:
                unseen = [i for i, history in enumerate(self._buffers) if not history]
                if unseen:
                    arm = unseen[0]
                else:
                    round_index = len(self._completed) + 1
                    effective = min(round_index, self.window_size * len(self.thresholds))
                    scores = [math.fsum(history) / len(history) + self.exploration *
                              math.sqrt(2 * math.log(effective) / len(history))
                              for history in self._buffers]
                    arm = max(range(len(scores)), key=scores.__getitem__)
            self._active = (run_id, arm)
            self._request_order.append(run_id)
            return self.thresholds[arm]

    def update(self, run_id: str, *, answer_log_probs: tuple[float, ...], output_tokens: int,
               completed: bool) -> dict[str, Any]:
        with self._lock:
            if run_id in self._completed:
                if self._active is not None and self._active[0] == run_id:
                    self._active = None
                return {"updated": False, "reason": "already_updated", **self._completed[run_id]}
            if self._active is None or self._active[0] != run_id:
                return {"updated": False, "reason": "request_not_active"}
            _, arm = self._active
            self._active = None
            valid_logs = bool(answer_log_probs) and all(
                math.isfinite(p) and p <= 0 for p in answer_log_probs)
            if not completed:
                return {"updated": False, "reason": "run_not_completed"}
            if not valid_logs or type(output_tokens) is not int or output_tokens <= 0:
                return {"updated": False, "reason": "invalid_answer_likelihood_or_length"}
            score = math.exp(math.fsum(answer_log_probs) / len(answer_log_probs))
            count = len(self._completed)
            prior_mean = self._total_length / count if count else None
            penalty = (self.length_penalty * output_tokens / prior_mean
                       if prior_mean is not None else 0.0001 * output_tokens)
            reward = score - penalty
            record = {"arm_index": arm, "threshold": self.thresholds[arm], "reward": reward,
                      "answer_geometric_token_likelihood": score, "output_tokens": output_tokens,
                      "prior_mean_output_tokens": prior_mean, "length_penalty": penalty,
                      "update_version": "refrain_per_arm_prior_mean_v1"}
            self._buffers[arm].append(reward)
            self._completed[run_id] = record
            self._total_length += output_tokens
            return {"updated": True, **record}

    def abort(self, run_id: str) -> None:
        with self._lock:
            if self._active is not None and self._active[0] == run_id:
                self._active = None

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            if self._active is not None:
                raise ConfigurationError("save a REFRAIN session only at completed request boundaries")
            return {"schema_version": "1.0", "update_version": "refrain_per_arm_prior_mean_v1",
                    "session_id": self.session_id, "thresholds": list(self.thresholds),
                    "window_size": self.window_size, "exploration": self.exploration,
                    "length_penalty": self.length_penalty, "seed": self.seed,
                    "tie_break": "configured_arm_order", "window_scope": "per_arm",
                    "buffers": [list(history) for history in self._buffers],
                    "completed": {key: dict(value) for key, value in self._completed.items()},
                    "request_order": list(self._request_order), "total_length": self._total_length}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> RefrainSession:
        if (state.get("schema_version") != "1.0"
                or state.get("update_version") != "refrain_per_arm_prior_mean_v1"
                or state.get("window_scope") != "per_arm"
                or state.get("tie_break") != "configured_arm_order"):
            raise ConfigurationError("unsupported REFRAIN session schema/controller semantics")
        result = cls(tuple(state["thresholds"]), window_size=state["window_size"],
                     exploration=state["exploration"], length_penalty=state["length_penalty"],
                     seed=state["seed"], session_id=state["session_id"])
        buffers = state["buffers"]
        if len(buffers) != len(result.thresholds) or any(
            len(history) > result.window_size or any(not math.isfinite(x) for x in history)
            for history in buffers
        ):
            raise ConfigurationError("invalid REFRAIN reward buffers")
        result._buffers = [deque(history, maxlen=result.window_size) for history in buffers]
        result._completed = {key: dict(value) for key, value in state["completed"].items()}
        result._request_order = list(state["request_order"])
        result._total_length = state["total_length"]
        if (type(result._total_length) is not int or result._total_length < 0
                or sum(r["output_tokens"] for r in result._completed.values()) != result._total_length
                or any(r["arm_index"] not in range(len(result.thresholds))
                       for r in result._completed.values())):
            raise ConfigurationError("inconsistent REFRAIN completed history")
        return result

    def save(self, path: str | Path) -> None:
        target = Path(path)
        data = json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(data, encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> RefrainSession:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
