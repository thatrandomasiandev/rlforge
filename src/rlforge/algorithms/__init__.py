"""RL algorithms."""

from rlforge.algorithms.tabular.q_learning import QLearning
from rlforge.algorithms.tabular.sarsa import SARSA

__all__ = ["QLearning", "SARSA"]

try:
    from rlforge.algorithms.deep.dqn import DQN
    from rlforge.algorithms.deep.ppo import PPO

    __all__ += ["DQN", "PPO"]
except ImportError:
    pass
