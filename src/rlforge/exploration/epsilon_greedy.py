"""Epsilon-greedy exploration over discrete action spaces."""

from __future__ import annotations

import numpy as np

from rlforge.utils.schedule import ConstantSchedule, LinearSchedule, Schedule


class EpsilonGreedy:
    """Select random actions with probability epsilon, else greedy."""

    def __init__(
        self,
        n_actions: int,
        *,
        epsilon: float | Schedule = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_actions = int(n_actions)
        if isinstance(epsilon, Schedule):
            self.schedule: Schedule = epsilon
        else:
            self.schedule = ConstantSchedule(float(epsilon))
        self.rng = rng or np.random.default_rng()

    @classmethod
    def linear(
        cls,
        n_actions: int,
        start: float = 1.0,
        end: float = 0.05,
        end_fraction: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> EpsilonGreedy:
        return cls(
            n_actions,
            epsilon=LinearSchedule(start, end, end_fraction=end_fraction),
            rng=rng,
        )

    def epsilon(self, progress: float) -> float:
        return float(self.schedule(progress))

    def select(
        self,
        q_values: np.ndarray,
        *,
        progress: float = 1.0,
        deterministic: bool = False,
    ) -> int:
        if deterministic or self.rng.random() >= self.epsilon(progress):
            return int(np.argmax(q_values))
        return int(self.rng.integers(0, self.n_actions))
