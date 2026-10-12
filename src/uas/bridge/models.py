"""Universal, framework-agnostic episode schema and query models."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass
class CompatibilityReport:
    """Detailed report on whether two episodes or agents share execution compatibility."""

    compatible: bool
    reason: str
    observation_compatible: bool
    action_compatible: bool


@dataclass
class Episode:
    """A framework-agnostic record of an embodied training or evaluation episode.

    Can be written by any framework (Gymnasium, Multiverse, Isaac Lab, ROS 2, PyTorch)
    and retrieved by any compatible learner. Preserves both continuous and discrete
    action semantics.
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
    observation_schema: str = "continuous"
    action_schema: str = "continuous"
    action_type: str = "continuous"
    metadata: dict[str, Any] = field(default_factory=dict)
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)

        # Preserve integer semantics for discrete action spaces
        raw_actions = np.asarray(self.actions)
        if np.issubdtype(raw_actions.dtype, np.integer):
            self.actions = raw_actions.astype(np.int64)
            if self.action_type == "continuous":
                self.action_type = "discrete"
        else:
            self.actions = raw_actions.astype(np.float32)

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

    def check_compatibility(
        self, other_obs_schema: str, other_action_schema: str, other_obs_dim: tuple[int, ...], other_act_dim: tuple[int, ...]
    ) -> CompatibilityReport:
        """Check whether another agent or task is observation and action compatible."""
        obs_ok = (
            self.observation_schema == other_obs_schema
            and self.observation_dim == other_obs_dim
        )
        act_ok = (
            self.action_schema == other_action_schema
            and self.action_dim == other_act_dim
        )
        if obs_ok and act_ok:
            return CompatibilityReport(
                compatible=True,
                reason="Schemas and dimensions match exactly",
                observation_compatible=True,
                action_compatible=True,
            )
        reasons = []
        if not obs_ok:
            reasons.append(
                f"Observation mismatch ({self.observation_schema} {self.observation_dim} vs {other_obs_schema} {other_obs_dim})"
            )
        if not act_ok:
            reasons.append(
                f"Action mismatch ({self.action_schema} {self.action_dim} vs {other_action_schema} {other_act_dim})"
            )
        return CompatibilityReport(
            compatible=False,
            reason="; ".join(reasons),
            observation_compatible=obs_ok,
            action_compatible=act_ok,
        )


@dataclass
class SearchQuery:
    """Filter criteria for searching prior experiences across frameworks."""

    task: str | None = None
    embodiment: str | None = None
    framework: str | None = None
    observation_schema: str | None = None
    action_schema: str | None = None
    action_type: str | None = None
    success_only: bool = False
    min_reward: float | None = None
    tags: list[str] | None = None
    limit: int = 10
