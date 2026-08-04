"""Base agent interface shared by all algorithms."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlforge.core.callback import BaseCallback, CallbackList
from rlforge.core.env import make_env
from rlforge.core.logger import Logger
from rlforge.utils.seeding import set_random_seed


class MonitorEpisode(gym.Wrapper):
    """Track episodic return/length and expose them via ``info['episode']``."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.episode_returns = 0.0
        self.episode_lengths = 0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self.episode_returns = 0.0
        self.episode_lengths = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_returns += float(reward)
        self.episode_lengths += 1
        if terminated or truncated:
            info = dict(info)
            info["episode"] = {"r": self.episode_returns, "l": self.episode_lengths}
        return obs, reward, terminated, truncated, info


class BaseAgent(ABC):
    """Abstract base class for RLForge agents."""

    def __init__(
        self,
        env: str | gym.Env,
        *,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        seed: int | None = None,
        verbose: int = 1,
        device: str = "cpu",
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.seed = seed
        self.verbose = verbose
        self.device = device
        self.num_timesteps = 0
        self._last_infos: dict[str, Any] = {}
        self.logger = Logger(verbose=verbose)

        if seed is not None:
            set_random_seed(seed)

        raw_env = make_env(env, seed=seed)
        self.env = MonitorEpisode(raw_env)
        self.observation_space: spaces.Space = self.env.observation_space
        self.action_space: spaces.Space = self.env.action_space
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback: BaseCallback | list[BaseCallback] | None = None,
        log_interval: int = 1_000,
    ) -> BaseAgent:
        """Train the agent for ``total_timesteps`` environment steps."""

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Select an action for the given observation."""

    @abstractmethod
    def _get_save_data(self) -> dict[str, Any]:
        """Return algorithm-specific state for serialization."""

    @abstractmethod
    def _load_save_data(self, data: dict[str, Any]) -> None:
        """Restore algorithm-specific state from ``data``."""

    def _init_callback(
        self,
        callback: BaseCallback | list[BaseCallback] | None,
    ) -> CallbackList:
        if callback is None:
            callback_list = CallbackList([])
        elif isinstance(callback, list):
            callback_list = CallbackList(callback)
        elif isinstance(callback, CallbackList):
            callback_list = callback
        else:
            callback_list = CallbackList([callback])
        callback_list.init_callback(self)
        return callback_list

    def save(self, path: str | Path) -> None:
        """Save agent parameters and metadata to ``path`` (directory)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "algorithm": self.__class__.__name__,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "seed": self.seed,
            "num_timesteps": self.num_timesteps,
            "observation_space": _space_to_dict(self.observation_space),
            "action_space": _space_to_dict(self.action_space),
        }
        with (path / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        data = self._get_save_data()
        np.savez_compressed(path / "state.npz", **_numpy_safe(data.get("numpy", {})))

        torch_state = data.get("torch")
        if torch_state is not None:
            import torch

            torch.save(torch_state, path / "model.pt")

    def load(self, path: str | Path) -> BaseAgent:
        """Load agent parameters from ``path`` into this instance."""
        path = Path(path)
        with (path / "meta.json").open(encoding="utf-8") as f:
            meta = json.load(f)
        self.num_timesteps = int(meta.get("num_timesteps", 0))

        npz = np.load(path / "state.npz", allow_pickle=True)
        numpy_data = {key: npz[key] for key in npz.files}

        torch_state = None
        model_path = path / "model.pt"
        if model_path.exists():
            import torch

            torch_state = torch.load(model_path, map_location=self.device, weights_only=False)

        self._load_save_data({"numpy": numpy_data, "torch": torch_state, "meta": meta})
        return self

    @classmethod
    def load_from(
        cls,
        path: str | Path,
        env: str | gym.Env,
        **kwargs: Any,
    ) -> BaseAgent:
        """Construct an agent and load weights from disk."""
        agent = cls(env, **kwargs)
        return agent.load(path)


def _space_to_dict(space: spaces.Space) -> dict[str, Any]:
    if isinstance(space, spaces.Discrete):
        return {"type": "Discrete", "n": int(space.n)}
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "low": space.low.tolist(),
            "high": space.high.tolist(),
            "dtype": str(space.dtype),
        }
    return {"type": type(space).__name__}


def _numpy_safe(mapping: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in mapping.items():
        out[key] = np.asarray(value)
    return out
