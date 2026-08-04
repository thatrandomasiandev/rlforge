"""PyTorch MLP policies for deep RL algorithms."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def _mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Sequence[int] = (64, 64),
    activation: type[nn.Module] = nn.Tanh,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(prev, hidden))
        layers.append(activation())
        prev = hidden
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """MLP Q-network for discrete actions."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        self.net = _mlp(obs_dim, n_actions, hidden_sizes=hidden_sizes, activation=nn.ReLU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MlpPolicy(nn.Module):
    """Discrete-action Q-policy with epsilon-greedy selection helpers."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        hidden_sizes: Sequence[int] = (64, 64),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.q_net = QNetwork(obs_dim, n_actions, hidden_sizes=hidden_sizes).to(self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_net(obs)

    def predict(
        self,
        observation: np.ndarray,
        *,
        epsilon: float = 0.0,
        deterministic: bool = True,
        rng: np.random.Generator | None = None,
    ) -> int:
        if (not deterministic) and rng is not None and rng.random() < epsilon:
            return int(rng.integers(0, self.n_actions))
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return int(torch.argmax(q_values, dim=1).item())


class ActorCriticPolicy(nn.Module):
    """Shared-trunk actor-critic for discrete PPO."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        hidden_sizes: Sequence[int] = (64, 64),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(device)

        layers: list[nn.Module] = []
        prev = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.Tanh())
            prev = hidden
        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, n_actions)
        self.value_head = nn.Linear(prev, 1)
        self.to(self.device)

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        features = self.shared(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return Categorical(logits=logits), value

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[int, float, float]:
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            dist, value = self.forward(obs_t)
            action = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, values = self.forward(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy
