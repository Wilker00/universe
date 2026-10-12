"""Data structures shared by the memory store and training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np


@dataclass(frozen=True)
class TaskQuery:
    task_id: str
    task_family: str
    task_description: str
    observation_schema: str
    action_schema: str
    task_vector: tuple[float, ...]


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    schema_version: int
    environment_id: str
    task_id: str
    task_family: str
    task_description: str
    observation_schema: str
    action_schema: str
    agent_id: str
    policy_version: str
    policy_reference: str | None
    seed: int
    success: bool
    total_reward: float
    normalized_score: float
    steps: int
    termination_reason: str
    collision_count: int
    task_vector: tuple[float, ...]
    trajectory_file: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def create(
        cls,
        *,
        environment_id: str,
        task_id: str,
        task_family: str,
        task_description: str,
        observation_schema: str,
        action_schema: str,
        agent_id: str,
        policy_version: str,
        policy_reference: str | None,
        seed: int,
        success: bool,
        total_reward: float,
        normalized_score: float,
        steps: int,
        termination_reason: str,
        collision_count: int,
        task_vector: tuple[float, ...],
    ) -> "EpisodeRecord":
        episode_id = str(uuid4())
        return cls(
            episode_id=episode_id,
            schema_version=1,
            environment_id=environment_id,
            task_id=task_id,
            task_family=task_family,
            task_description=task_description,
            observation_schema=observation_schema,
            action_schema=action_schema,
            agent_id=agent_id,
            policy_version=policy_version,
            policy_reference=policy_reference,
            seed=int(seed),
            success=bool(success),
            total_reward=float(total_reward),
            normalized_score=float(np.clip(normalized_score, 0.0, 1.0)),
            steps=int(steps),
            termination_reason=termination_reason,
            collision_count=int(collision_count),
            task_vector=tuple(float(value) for value in task_vector),
            trajectory_file=f"{episode_id}.npz",
        )


@dataclass(frozen=True)
class Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def validate(self) -> None:
        transition_count = len(self.actions)
        if len(self.observations) != transition_count + 1:
            raise ValueError("A trajectory needs one more observation than actions")
        arrays = (self.rewards, self.terminated, self.truncated)
        if any(len(array) != transition_count for array in arrays):
            raise ValueError("Actions, rewards, and terminal flags must have equal length")
        if transition_count == 0:
            raise ValueError("Cannot store an empty trajectory")


@dataclass(frozen=True)
class RetrievedEpisode:
    record: EpisodeRecord
    similarity: float
    retrieval_score: float
