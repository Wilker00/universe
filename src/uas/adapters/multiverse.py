"""Multiverse framework adapter for UAS MemoryBridge.

Translates between Multiverse native execution artifacts (StepEvent sequences,
Transition batches, Verse specifications) and framework-agnostic UAS Episodes.
"""

from __future__ import annotations

import uuid
from typing import Any, Sequence

import numpy as np

from uas.bridge.client import MemoryBridge
from uas.bridge.models import Episode


def _extract_attr_or_key(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _to_numpy_array(value: Any, default_dtype: type = np.float32) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(np.bool_)
    try:
        return arr.astype(default_dtype)
    except (ValueError, TypeError):
        return arr


def multiverse_to_uas_episode(
    *,
    events: Sequence[Any] | None = None,
    transitions: Sequence[Any] | None = None,
    task: str | None = None,
    embodiment: str = "multiverse_agent",
    success: bool | None = None,
    total_reward: float | None = None,
    tags: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Episode:
    """Convert Multiverse StepEvents or Transitions into a framework-agnostic Episode."""
    meta = dict(metadata or {})
    if tags:
        meta["tags"] = list(tags)

    obs_list: list[Any] = []
    actions_list: list[Any] = []
    rewards_list: list[float] = []
    terminated_list: list[bool] = []
    truncated_list: list[bool] = []
    inferred_task = task
    inferred_success = success

    if transitions is not None and len(transitions) > 0:
        for idx, trans in enumerate(transitions):
            obs = _extract_attr_or_key(trans, "obs")
            action = _extract_attr_or_key(trans, "action")
            reward = float(_extract_attr_or_key(trans, "reward", 0.0))
            done = bool(_extract_attr_or_key(trans, "done", False))
            trunc = bool(_extract_attr_or_key(trans, "truncated", False))
            next_obs = _extract_attr_or_key(trans, "next_obs")
            info = _extract_attr_or_key(trans, "info", {})

            if idx == 0:
                obs_list.append(obs)
            obs_list.append(next_obs if next_obs is not None else obs)

            actions_list.append(action)
            rewards_list.append(reward)
            terminated_list.append(done)
            truncated_list.append(trunc)

            if inferred_success is None and isinstance(info, dict):
                if "success" in info:
                    inferred_success = bool(info["success"])
                elif "is_success" in info:
                    inferred_success = bool(info["is_success"])

    elif events is not None and len(events) > 0:
        for idx, event in enumerate(events):
            obs = _extract_attr_or_key(event, "obs")
            action = _extract_attr_or_key(event, "action")
            reward = float(_extract_attr_or_key(event, "reward", 0.0))
            done = bool(_extract_attr_or_key(event, "done", False))
            trunc = bool(_extract_attr_or_key(event, "truncated", False))
            info = _extract_attr_or_key(event, "info", {})

            if inferred_task is None:
                inferred_task = _extract_attr_or_key(event, "verse_name")

            obs_list.append(obs)
            actions_list.append(action)
            rewards_list.append(reward)
            terminated_list.append(done)
            truncated_list.append(trunc)

            # Check for next_obs or success in final step info
            if idx == len(events) - 1 and isinstance(info, dict):
                if "next_obs" in info:
                    obs_list.append(info["next_obs"])
                if inferred_success is None:
                    if "success" in info:
                        inferred_success = bool(info["success"])
                    elif "is_success" in info:
                        inferred_success = bool(info["is_success"])
    else:
        raise ValueError("Either events or transitions must be provided and non-empty")

    if not inferred_task:
        inferred_task = "multiverse_verse"

    rewards_arr = np.asarray(rewards_list, dtype=np.float32)
    computed_reward = (
        float(total_reward)
        if total_reward is not None
        else float(np.sum(rewards_arr))
    )
    if inferred_success is None:
        inferred_success = bool(computed_reward > 0.0)

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    actions_raw = np.asarray(actions_list)
    if np.issubdtype(actions_raw.dtype, np.integer):
        actions_arr = actions_raw.astype(np.int64)
        action_type = "discrete"
        action_schema = "discrete"
    else:
        actions_arr = actions_raw.astype(np.float32)
        action_type = "continuous"
        action_schema = "continuous"

    return Episode(
        task=inferred_task,
        embodiment=embodiment,
        framework="multiverse",
        success=inferred_success,
        total_reward=computed_reward,
        observations=obs_arr,
        actions=actions_arr,
        rewards=rewards_arr,
        terminated=np.asarray(terminated_list, dtype=np.bool_),
        truncated=np.asarray(truncated_list, dtype=np.bool_),
        observation_schema="continuous",
        action_schema=action_schema,
        action_type=action_type,
        metadata=meta,
    )


def uas_episode_to_multiverse_transitions(
    episode: Episode,
) -> list[dict[str, Any]]:
    """Convert a UAS Episode into a list of Multiverse-compatible transition dictionaries."""
    num_steps = episode.steps
    has_next_obs = len(episode.observations) == num_steps + 1
    transitions: list[dict[str, Any]] = []

    for t in range(num_steps):
        obs = episode.observations[t].tolist()
        if has_next_obs:
            next_obs = episode.observations[t + 1].tolist()
        else:
            next_obs = (
                episode.observations[t].tolist()
                if t == num_steps - 1
                else episode.observations[t + 1].tolist()
            )

        act = episode.actions[t]
        action_val: Any = act.item() if act.ndim == 0 or act.size == 1 else act.tolist()

        trans = {
            "obs": obs,
            "action": action_val,
            "reward": float(episode.rewards[t]),
            "next_obs": next_obs,
            "done": bool(episode.terminated[t]),
            "truncated": bool(episode.truncated[t]),
            "info": {
                "step_idx": t,
                "task": episode.task,
                "framework": episode.framework,
                "success": episode.success,
            },
        }
        transitions.append(trans)

    return transitions


def uas_episode_to_multiverse_events(
    episode: Episode,
) -> list[dict[str, Any]]:
    """Convert a UAS Episode into a list of Multiverse StepEvent-compatible dictionaries."""
    num_steps = episode.steps
    events: list[dict[str, Any]] = []
    run_id = str(episode.metadata.get("run_id", f"run_{episode.episode_id[:8]}"))

    for t in range(num_steps):
        act = episode.actions[t]
        action_val: Any = act.item() if act.ndim == 0 or act.size == 1 else act.tolist()
        step_event = {
            "schema_version": "1.0",
            "run_id": run_id,
            "episode_id": episode.episode_id,
            "step_idx": t,
            "verse_name": episode.task,
            "obs": episode.observations[t].tolist(),
            "action": action_val,
            "reward": float(episode.rewards[t]),
            "done": bool(episode.terminated[t]),
            "truncated": bool(episode.truncated[t]),
            "info": dict(episode.metadata),
        }
        events.append(step_event)

    return events


class MultiverseAdapter:
    """Convenience bridge wrapper for Multiverse environments and agents."""

    def __init__(self, bridge: MemoryBridge) -> None:
        self.bridge = bridge

    def record_rollout(
        self,
        *,
        events: Sequence[Any] | None = None,
        transitions: Sequence[Any] | None = None,
        task: str | None = None,
        embodiment: str = "multiverse_agent",
        success: bool | None = None,
        total_reward: float | None = None,
        tags: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        """Convert a Multiverse rollout and record it in the shared UAS MemoryBridge."""
        episode = multiverse_to_uas_episode(
            events=events,
            transitions=transitions,
            task=task,
            embodiment=embodiment,
            success=success,
            total_reward=total_reward,
            tags=tags,
            metadata=metadata,
        )
        self.bridge.record(episode)
        return episode

    def retrieve_for_verse(
        self,
        verse_name: str,
        *,
        embodiment: str | None = None,
        success_only: bool = True,
        min_reward: float | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[Episode]:
        """Search UAS MemoryBridge for prior experience matching a Verse."""
        return self.bridge.search(
            task=verse_name,
            embodiment=embodiment,
            success_only=success_only,
            min_reward=min_reward,
            tags=tags,
            limit=limit,
        )

    def retrieve_transitions(
        self,
        verse_name: str,
        *,
        embodiment: str | None = None,
        success_only: bool = True,
        min_reward: float | None = None,
        tags: list[str] | None = None,
        limit: int = 5,
    ) -> list[list[dict[str, Any]]]:
        """Retrieve relevant past experience formatted directly for Multiverse learners."""
        episodes = self.retrieve_for_verse(
            verse_name=verse_name,
            embodiment=embodiment,
            success_only=success_only,
            min_reward=min_reward,
            tags=tags,
            limit=limit,
        )
        return [uas_episode_to_multiverse_transitions(ep) for ep in episodes]
