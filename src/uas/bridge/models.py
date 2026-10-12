"""Universal, framework-agnostic episode schema and query models."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass
class Episode:
    """A framework-agnostic record of an embodied training or evaluation episode.

    Can be written by any framework (Gymnasium, Multiverse, Isaac Lab, ROS 2, PyTorch)
    and retrieved by any compatible learner.
    """

    task: str
    embodiment: str
    framework: str
    success: bool
    total_reward: float
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        self.terminated = np.asarray(self.terminated, dtype=np.bool_)
        self.truncated = np.asarray(self.truncated, dtype=np.bool_)

        transition_count = len(self.actions)
        if len(self.rewards) != transition_count:
            raise ValueError(
                f"Reward count ({len(self.rewards)}) must match actions ({transition_count})"
            )
        if len(self.terminated) != transition_count:
            raise ValueError(
                f"Terminated count ({len(self.terminated)}) must match actions ({transition_count})"
            )
        if len(self.truncated) != transition_count:
            raise ValueError(
                f"Truncated count ({len(self.truncated)}) must match actions ({transition_count})"
            )
        if len(self.observations) not in (transition_count, transition_count + 1):
            raise ValueError(
                f"Observation count ({len(self.observations)}) must be T or T+1 "
                f"for actions ({transition_count})"
            )

    @property
    def steps(self) -> int:
        return len(self.actions)

    @property
    def observation_dim(self) -> tuple[int, ...]:
        return self.observations.shape[1:]

    @property
    def action_dim(self) -> tuple[int, ...]:
        return self.actions.shape[1:]


@dataclass
class SearchQuery:
    """Filter criteria for searching prior experiences across frameworks."""

    task: str | None = None
    embodiment: str | None = None
    framework: str | None = None
    success_only: bool = False
    min_reward: float | None = None
    tags: list[str] | None = None
    limit: int = 10
