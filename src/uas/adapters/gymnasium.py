"""Gymnasium and UAS trajectory adapter for UAS MemoryBridge."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from uas.bridge.models import Episode
from uas.memory.models import Trajectory


def gymnasium_to_uas_episode(
    *,
    trajectory: Trajectory | None = None,
    observations: np.ndarray | Sequence[Any] | None = None,
    actions: np.ndarray | Sequence[Any] | None = None,
    rewards: np.ndarray | Sequence[float] | None = None,
    terminated: np.ndarray | Sequence[bool] | None = None,
    truncated: np.ndarray | Sequence[bool] | None = None,
    task: str = "gymnasium_task",
    embodiment: str = "gymnasium_agent",
    success: bool | None = None,
    total_reward: float | None = None,
    tags: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Episode:
    """Convert Gymnasium / SB3 rollout data or Trajectory into a UAS Episode."""
    meta = dict(metadata or {})
    if tags:
        meta["tags"] = list(tags)

    if trajectory is not None:
        obs_arr = np.asarray(trajectory.observations, dtype=np.float32)
        acts_arr = np.asarray(trajectory.actions)
        rew_arr = np.asarray(trajectory.rewards, dtype=np.float32)
        term_arr = np.asarray(trajectory.terminated, dtype=np.bool_)
        trunc_arr = np.asarray(trajectory.truncated, dtype=np.bool_)
    else:
        if (
            observations is None
            or actions is None
            or rewards is None
            or terminated is None
            or truncated is None
        ):
            raise ValueError(
                "Either trajectory or all rollout arrays (observations, actions, "
                "rewards, terminated, truncated) must be provided"
            )
        obs_arr = np.asarray(observations, dtype=np.float32)
        acts_arr = np.asarray(actions)
        rew_arr = np.asarray(rewards, dtype=np.float32)
        term_arr = np.asarray(terminated, dtype=np.bool_)
        trunc_arr = np.asarray(truncated, dtype=np.bool_)

    if np.issubdtype(acts_arr.dtype, np.integer):
        actions_final = acts_arr.astype(np.int64)
        action_type = "discrete"
        action_schema = "discrete"
    else:
        actions_final = acts_arr.astype(np.float32)
        action_type = "continuous"
        action_schema = "continuous"

    computed_reward = (
        float(total_reward)
        if total_reward is not None
        else float(np.sum(rew_arr))
    )
    computed_success = (
        bool(success) if success is not None else bool(computed_reward > 0.0)
    )

    return Episode(
        task=task,
        embodiment=embodiment,
        framework="gymnasium",
        success=computed_success,
        total_reward=computed_reward,
        observations=obs_arr,
        actions=actions_final,
        rewards=rew_arr,
        terminated=term_arr,
        truncated=trunc_arr,
        observation_schema="continuous",
        action_schema=action_schema,
        action_type=action_type,
        metadata=meta,
    )


def uas_episode_to_trajectory(episode: Episode) -> Trajectory:
    """Convert a UAS Episode to a UAS Trajectory (ensuring T+1 observations)."""
    obs = np.asarray(episode.observations, dtype=np.float32)
    acts = np.asarray(episode.actions)
    rews = np.asarray(episode.rewards, dtype=np.float32)
    terms = np.asarray(episode.terminated, dtype=np.bool_)
    truncs = np.asarray(episode.truncated, dtype=np.bool_)

    num_steps = len(acts)
    if len(obs) == num_steps:
        # Pad final observation with duplicate if only T observations were recorded
        last_obs = obs[-1:]
        obs = np.concatenate([obs, last_obs], axis=0)
    elif len(obs) != num_steps + 1:
        raise ValueError(
            f"Expected {num_steps} or {num_steps + 1} observations, got {len(obs)}"
        )

    trajectory = Trajectory(
        observations=obs,
        actions=acts.astype(np.float32) if np.issubdtype(acts.dtype, np.floating) else acts,
        rewards=rews,
        terminated=terms,
        truncated=truncs,
    )
    trajectory.validate()
    return trajectory
