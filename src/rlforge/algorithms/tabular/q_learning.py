"""Off-policy tabular Q-Learning."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlforge.core.agent import BaseAgent
from rlforge.core.callback import BaseCallback
from rlforge.core.spaces import require_discrete
from rlforge.exploration.epsilon_greedy import EpsilonGreedy
from rlforge.policies.tabular import TabularQPolicy
from rlforge.utils.schedule import LinearSchedule, Schedule


class QLearning(BaseAgent):
    """Tabular Q-Learning with epsilon-greedy exploration."""

    def __init__(
        self,
        env: str | gym.Env,
        *,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float | Schedule | None = None,
        seed: int | None = None,
        verbose: int = 1,
        policy: TabularQPolicy | None = None,
        exploration: EpsilonGreedy | None = None,
    ) -> None:
        super().__init__(
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            seed=seed,
            verbose=verbose,
        )
        obs_space = require_discrete(self.observation_space, "observation_space")
        act_space = require_discrete(self.action_space, "action_space")

        if epsilon is None:
            epsilon = LinearSchedule(1.0, 0.05, end_fraction=0.8)
        self.exploration = exploration or EpsilonGreedy(
            int(act_space.n),
            epsilon=epsilon,
            rng=self._rng,
        )
        self.policy = policy or TabularQPolicy(
            n_states=int(obs_space.n),
            n_actions=int(act_space.n),
            exploration=self.exploration,
            rng=self._rng,
        )

    def learn(
        self,
        total_timesteps: int,
        callback: BaseCallback | list[BaseCallback] | None = None,
        log_interval: int = 1_000,
    ) -> QLearning:
        callback_list = self._init_callback(callback)
        callback_list.on_training_start()

        obs, _ = self.env.reset(seed=self.seed)
        state = _to_state(obs)
        episode_reward = 0.0

        for step in range(1, total_timesteps + 1):
            self.num_timesteps = step
            progress = 1.0 - (step - 1) / max(total_timesteps, 1)
            action = self.policy.predict(state, progress=progress, deterministic=False)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            next_state = _to_state(next_obs)
            episode_reward += float(reward)

            best_next = float(np.max(self.policy.q_table[next_state]))
            target = float(reward) + (0.0 if terminated else self.gamma * best_next)
            td_error = self.policy.update(state, action, target, self.learning_rate)

            self._last_infos = info
            self.logger.record("train/td_error", abs(td_error))

            if not callback_list.on_step():
                break

            if done:
                self.logger.record("rollout/ep_rew", episode_reward)
                obs, _ = self.env.reset()
                state = _to_state(obs)
                episode_reward = 0.0
            else:
                state = next_state

            if log_interval > 0 and step % log_interval == 0:
                self.logger.record("train/epsilon", self.exploration.epsilon(progress))
                self.logger.dump(step)

        callback_list.on_training_end()
        return self

    def predict(
        self,
        observation: np.ndarray | int,
        *,
        deterministic: bool = True,
    ) -> tuple[int, dict[str, Any] | None]:
        state = _to_state(observation)
        action = self.policy.predict(state, deterministic=deterministic)
        return action, None

    def _get_save_data(self) -> dict[str, Any]:
        return {"numpy": {"q_table": self.policy.q_table}}

    def _load_save_data(self, data: dict[str, Any]) -> None:
        self.policy.q_table = np.asarray(data["numpy"]["q_table"], dtype=np.float64)


def _to_state(observation: np.ndarray | int | spaces.Space) -> int:
    if isinstance(observation, (int, np.integer)):
        return int(observation)
    arr = np.asarray(observation)
    if arr.ndim == 0:
        return int(arr.item())
    return int(arr.reshape(-1)[0])
