"""Helpers for validating and inspecting Gymnasium spaces."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces


def is_discrete(space: spaces.Space) -> bool:
    return isinstance(space, spaces.Discrete)


def is_box(space: spaces.Space) -> bool:
    return isinstance(space, spaces.Box)


def flat_dim(space: spaces.Space) -> int:
    """Return the flattened observation/action dimension for supported spaces."""
    if isinstance(space, spaces.Discrete):
        return int(space.n)
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    raise TypeError(f"Unsupported space type: {type(space)!r}")


def require_discrete(space: spaces.Space, name: str) -> spaces.Discrete:
    if not isinstance(space, spaces.Discrete):
        raise TypeError(f"{name} must be Discrete, got {type(space)!r}")
    return space


def require_box_or_discrete(space: spaces.Space, name: str) -> spaces.Space:
    if not (isinstance(space, (spaces.Box, spaces.Discrete))):
        raise TypeError(f"{name} must be Box or Discrete, got {type(space)!r}")
    return space
