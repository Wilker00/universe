"""UAS environments."""

from gymnasium.envs.registration import register, registry

from uas.envs.continuous_world import (
    ACTION_SCHEMA,
    ENVIRONMENT_ID,
    OBSERVATION_SCHEMA,
    TASK_FAMILY,
    TASKS,
    ContinuousWorldEnv,
    TaskSpec,
)
from uas.envs.robotics_panda import PANDA_TASKS, PandaManipulationEnv

GYM_ENV_ID = "UAS/ContinuousWorld-v0"
PANDA_GYM_ENV_ID = "UAS/PandaManipulation-v0"

if GYM_ENV_ID not in registry:
    register(
        id=GYM_ENV_ID,
        entry_point="uas.envs:ContinuousWorldEnv",
    )

if PANDA_GYM_ENV_ID not in registry:
    register(
        id=PANDA_GYM_ENV_ID,
        entry_point="uas.envs:PandaManipulationEnv",
    )


def make_env(task_id: str = "open", **kwargs: object) -> ContinuousWorldEnv:
    """Create an unwrapped environment for direct use and testing."""
    return ContinuousWorldEnv(task_id=task_id, **kwargs)


def make_panda_env(task_id: str = "panda_reach", **kwargs: object) -> PandaManipulationEnv:
    """Create a Franka Panda manipulation environment."""
    return PandaManipulationEnv(task_id=task_id, **kwargs)


__all__ = [
    "ACTION_SCHEMA",
    "ENVIRONMENT_ID",
    "GYM_ENV_ID",
    "OBSERVATION_SCHEMA",
    "PANDA_GYM_ENV_ID",
    "PANDA_TASKS",
    "PandaManipulationEnv",
    "TASK_FAMILY",
    "TASKS",
    "ContinuousWorldEnv",
    "TaskSpec",
    "make_env",
    "make_panda_env",
]

