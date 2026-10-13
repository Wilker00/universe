"""Robotics adapter for UAS MemoryBridge.

Provides standardized schemas and converters for continuous robotic manipulation
environments (e.g., Franka Panda, UR5, Panda-Gym, Isaac Lab, ROS 2).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from uas.bridge.models import Episode


# Standard manipulation embodiment identifiers
ROBOT_EMBODIMENTS = {
    "franka_panda": {
        "joint_count": 7,
        "default_action_dim": 4,  # [dx, dy, dz, gripper]
        "description": "7-DoF Franka Emika Panda arm with parallel-jaw gripper",
    },
    "ur5": {
        "joint_count": 6,
        "default_action_dim": 4,
        "description": "6-DoF Universal Robots UR5 manipulator",
    },
    "kuka_iiwa": {
        "joint_count": 7,
        "default_action_dim": 4,
        "description": "7-DoF KUKA LBR iiwa collaborative arm",
    },
}


def robotics_to_uas_episode(
    *,
    observations: np.ndarray | Sequence[Any],
    actions: np.ndarray | Sequence[Any],
    rewards: np.ndarray | Sequence[float],
    terminated: np.ndarray | Sequence[bool],
    truncated: np.ndarray | Sequence[bool],
    task: str = "panda_reach",
    embodiment: str = "franka_panda",
    framework: str = "panda_gym",
    success: bool | None = None,
    total_reward: float | None = None,
    tags: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Episode:
    """Convert a robotics rollout into a framework-agnostic UAS Episode."""
    meta = dict(metadata or {})
    if tags:
        meta["tags"] = list(tags)
    if embodiment in ROBOT_EMBODIMENTS:
        meta["robot_spec"] = ROBOT_EMBODIMENTS[embodiment]

    obs_arr = np.asarray(observations, dtype=np.float32)
    acts_arr = np.asarray(actions, dtype=np.float32)
    rews_arr = np.asarray(rewards, dtype=np.float32)
    term_arr = np.asarray(terminated, dtype=np.bool_)
    trunc_arr = np.asarray(truncated, dtype=np.bool_)

    computed_reward = (
        float(total_reward)
        if total_reward is not None
        else float(np.sum(rews_arr))
    )
    computed_success = (
        bool(success) if success is not None else bool(computed_reward > 0.0)
    )

    return Episode(
        task=task,
        embodiment=embodiment,
        framework=framework,
        success=computed_success,
        total_reward=computed_reward,
        observations=obs_arr,
        actions=acts_arr,
        rewards=rews_arr,
        terminated=term_arr,
        truncated=trunc_arr,
        observation_schema="continuous",
        action_schema="continuous",
        action_type="continuous",
        metadata=meta,
    )
