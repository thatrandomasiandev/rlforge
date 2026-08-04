"""Tabular Q-value policy."""

from __future__ import annotations

import numpy as np

from rlforge.exploration.epsilon_greedy import EpsilonGreedy


class TabularQPolicy:
    """Q-table with epsilon-greedy action selection."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        *,
        exploration: EpsilonGreedy | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.rng = rng or np.random.default_rng()
        self.exploration = exploration or EpsilonGreedy(
            self.n_actions,
            epsilon=0.1,
            rng=self.rng,
        )

    def predict(
        self,
        state: int,
        *,
        progress: float = 1.0,
        deterministic: bool = False,
    ) -> int:
        return self.exploration.select(
            self.q_table[int(state)],
            progress=progress,
            deterministic=deterministic,
        )

    def update(
        self,
        state: int,
        action: int,
        target: float,
        learning_rate: float,
    ) -> float:
        """TD update; returns the TD error."""
        state_i = int(state)
        action_i = int(action)
        td_error = target - self.q_table[state_i, action_i]
        self.q_table[state_i, action_i] += learning_rate * td_error
        return float(td_error)
