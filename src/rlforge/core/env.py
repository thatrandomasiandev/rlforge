"""Environment creation and Gymnasium adapters."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces


def make_env(
    env: str | gym.Env,
    *,
    seed: int | None = None,
    **kwargs: Any,
) -> gym.Env:
    """Create or wrap a Gymnasium environment.

    Args:
        env: Environment id string or an already-constructed ``gym.Env``.
        seed: Optional seed applied to ``reset`` and action space.
        **kwargs: Forwarded to ``gym.make`` when ``env`` is a string.
    """
    if isinstance(env, str):
        environment = gym.make(env, **kwargs)
    elif isinstance(env, gym.Env):
        environment = env
    else:
        raise TypeError(f"env must be a str or gymnasium.Env, got {type(env)!r}")

    if seed is not None:
        environment.reset(seed=seed)
        environment.action_space.seed(seed)
        if hasattr(environment.observation_space, "seed"):
            environment.observation_space.seed(seed)

    return environment


def get_obs_shape(observation_space: spaces.Space) -> tuple[int, ...]:
    if isinstance(observation_space, spaces.Discrete):
        return (1,)
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    raise TypeError(f"Unsupported observation space: {type(observation_space)!r}")


def get_action_dim(action_space: spaces.Space) -> int:
    if isinstance(action_space, spaces.Discrete):
        return int(action_space.n)
    if isinstance(action_space, spaces.Box):
        return int(np_prod(action_space.shape))
    raise TypeError(f"Unsupported action space: {type(action_space)!r}")


def np_prod(shape: tuple[int, ...]) -> int:
    prod = 1
    for dim in shape:
        prod *= int(dim)
    return prod
