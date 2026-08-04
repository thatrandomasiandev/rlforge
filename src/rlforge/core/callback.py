"""Training callbacks for logging, early stopping, and research hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rlforge.core.agent import BaseAgent


class BaseCallback:
    """Base callback with hooks into the training loop."""

    def __init__(self) -> None:
        self.agent: BaseAgent | None = None
        self.n_calls = 0

    def init_callback(self, agent: BaseAgent) -> None:
        self.agent = agent

    def on_training_start(self) -> None:
        pass

    def on_training_end(self) -> None:
        pass

    def on_step(self) -> bool:
        """Called after each environment step.

        Returns:
            ``False`` to stop training early.
        """
        self.n_calls += 1
        return True

    def on_rollout_start(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        pass


class CallbackList(BaseCallback):
    """Compose multiple callbacks into one."""

    def __init__(self, callbacks: list[BaseCallback] | None = None) -> None:
        super().__init__()
        self.callbacks = list(callbacks or [])

    def init_callback(self, agent: BaseAgent) -> None:
        super().init_callback(agent)
        for callback in self.callbacks:
            callback.init_callback(agent)

    def on_training_start(self) -> None:
        for callback in self.callbacks:
            callback.on_training_start()

    def on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()

    def on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            continue_training = callback.on_step() and continue_training
        self.n_calls += 1
        return continue_training

    def on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()


class EpisodeRewardCallback(BaseCallback):
    """Track episodic returns and optionally stop when a threshold is reached."""

    def __init__(self, reward_threshold: float | None = None, window: int = 100) -> None:
        super().__init__()
        self.reward_threshold = reward_threshold
        self.window = window
        self.episode_rewards: list[float] = []
        self._current_reward = 0.0

    def on_step(self) -> bool:
        super().on_step()
        assert self.agent is not None
        infos = getattr(self.agent, "_last_infos", None)
        if infos and "episode" in infos:
            ep_reward = float(infos["episode"]["r"])
            self.episode_rewards.append(ep_reward)
            if self.agent.logger is not None:
                self.agent.logger.record("rollout/ep_rew_mean", self.mean_reward)
            if (
                self.reward_threshold is not None
                and len(self.episode_rewards) >= self.window
                and self.mean_reward >= self.reward_threshold
            ):
                return False
        return True

    @property
    def mean_reward(self) -> float:
        if not self.episode_rewards:
            return 0.0
        recent = self.episode_rewards[-self.window :]
        return float(sum(recent) / len(recent))
