"""Lightweight metrics logger."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class Logger:
    """Accumulate scalar metrics and optionally print them."""

    def __init__(self, *, verbose: int = 1) -> None:
        self.verbose = verbose
        self.name_to_value: dict[str, float] = {}
        self.name_to_count: dict[str, int] = defaultdict(int)
        self.history: dict[str, list[float]] = defaultdict(list)

    def record(self, key: str, value: float, *, exclude: Any = None) -> None:
        del exclude  # API compatibility placeholder
        self.name_to_value[key] = float(value)
        self.name_to_count[key] += 1
        self.history[key].append(float(value))

    def dump(self, step: int) -> None:
        if self.verbose <= 0 or not self.name_to_value:
            self.name_to_value.clear()
            return
        parts = [f"step={step}"]
        for key in sorted(self.name_to_value):
            parts.append(f"{key}={self.name_to_value[key]:.4g}")
        print(" | ".join(parts))
        self.name_to_value.clear()
