"""Policy and network modules."""

from rlforge.policies.tabular import TabularQPolicy

__all__ = ["TabularQPolicy"]

try:
    from rlforge.policies.mlp import ActorCriticPolicy, MlpPolicy, QNetwork

    __all__ += ["ActorCriticPolicy", "MlpPolicy", "QNetwork"]
except ImportError:
    pass
