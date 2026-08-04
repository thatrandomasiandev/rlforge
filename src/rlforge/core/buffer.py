"""Experience buffers for off-policy and on-policy algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Transition:
    """Single environment transition."""

    obs: np.ndarray
    action: int | np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    info: dict[str, Any] | None = None


class ReplayBuffer:
    """Fixed-size circular replay buffer for off-policy learning."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], action_dtype: Any = np.int64) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.action_dtype = action_dtype

        self.observations = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=action_dtype)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(
        self,
        obs: np.ndarray,
        action: int | np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.pos] = np.asarray(obs, dtype=np.float32)
        self.next_observations[self.pos] = np.asarray(next_obs, dtype=np.float32)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        size = len(self)
        if batch_size > size:
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer of size {size}")
        rng = rng or np.random.default_rng()
        indices = rng.integers(0, size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }


class RolloutBuffer:
    """Fixed-length rollout buffer with GAE advantage estimation."""

    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple[int, ...],
        action_dim: int,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self.buffer_size = int(buffer_size)
        self.obs_shape = tuple(obs_shape)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)

        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)

        self.pos = 0
        self.full = False

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        if self.pos >= self.buffer_size:
            raise RuntimeError("RolloutBuffer is full; call compute_returns_and_advantage then reset")
        self.observations[self.pos] = np.asarray(obs, dtype=np.float32)
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = float(done)
        self.values[self.pos] = float(value)
        self.log_probs[self.pos] = float(log_prob)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_value: float, last_done: bool) -> None:
        """Compute GAE advantages and returns for the collected rollout."""
        last_gae = 0.0
        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = float(last_value)
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_value * next_non_terminal
                - self.values[step]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        self.returns[: self.pos] = self.advantages[: self.pos] + self.values[: self.pos]

    def get(self) -> dict[str, np.ndarray]:
        if self.pos == 0:
            raise RuntimeError("RolloutBuffer is empty")
        adv = self.advantages[: self.pos]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "observations": self.observations[: self.pos],
            "actions": self.actions[: self.pos],
            "log_probs": self.log_probs[: self.pos],
            "advantages": adv.astype(np.float32),
            "returns": self.returns[: self.pos],
            "values": self.values[: self.pos],
        }
