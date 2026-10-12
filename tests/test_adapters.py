from pathlib import Path

import numpy as np
import pytest

from uas.adapters import (
    MultiverseAdapter,
    gymnasium_to_uas_episode,
    multiverse_to_uas_episode,
    uas_episode_to_multiverse_events,
    uas_episode_to_multiverse_transitions,
    uas_episode_to_trajectory,
)
from uas.bridge.client import MemoryBridge
from uas.memory.models import Trajectory


class DummyStepEvent:
    """Mock of Multiverse StepEvent."""

    def __init__(
        self,
        step_idx: int,
        verse_name: str,
        obs: list[float],
        action: int,
        reward: float,
        done: bool,
        truncated: bool = False,
        info: dict | None = None,
    ) -> None:
        self.step_idx = step_idx
        self.verse_name = verse_name
        self.obs = obs
        self.action = action
        self.reward = reward
        self.done = done
        self.truncated = truncated
        self.info = info or {}


class DummyTransition:
    """Mock of Multiverse Transition."""

    def __init__(
        self,
        obs: list[float],
        action: list[float],
        reward: float,
        next_obs: list[float],
        done: bool,
        truncated: bool = False,
        info: dict | None = None,
    ) -> None:
        self.obs = obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.done = done
        self.truncated = truncated
        self.info = info or {}


def test_multiverse_to_uas_from_step_events() -> None:
    events = [
        DummyStepEvent(0, "grid_navigation", [0.0, 1.0], 2, 0.0, False),
        DummyStepEvent(1, "grid_navigation", [1.0, 1.0], 3, 0.0, False),
        DummyStepEvent(
            2,
            "grid_navigation",
            [2.0, 1.0],
            1,
            10.0,
            True,
            info={"success": True, "next_obs": [3.0, 1.0]},
        ),
    ]
    episode = multiverse_to_uas_episode(events=events, tags=["grid", "demo"])

    assert episode.task == "grid_navigation"
    assert episode.framework == "multiverse"
    assert episode.success is True
    assert episode.total_reward == 10.0
    assert episode.steps == 3
    assert episode.action_type == "discrete"
    assert episode.actions.dtype == np.int64
    assert len(episode.observations) == 4
    assert episode.metadata.get("tags") == ["grid", "demo"]


def test_multiverse_to_uas_from_transitions() -> None:
    transitions = [
        DummyTransition([0.1, 0.2], [0.5, -0.5], 1.0, [0.2, 0.3], False),
        DummyTransition(
            [0.2, 0.3],
            [0.1, 0.0],
            5.0,
            [0.3, 0.4],
            True,
            info={"success": True},
        ),
    ]
    episode = multiverse_to_uas_episode(
        transitions=transitions,
        task="continuous_reach",
        embodiment="franka_arm",
    )

    assert episode.task == "continuous_reach"
    assert episode.embodiment == "franka_arm"
    assert episode.action_type == "continuous"
    assert episode.actions.dtype == np.float32
    assert episode.steps == 2
    assert len(episode.observations) == 3
    assert episode.total_reward == 6.0


def test_multiverse_adapter_record_and_retrieve(tmp_path: Path) -> None:
    bridge = MemoryBridge(tmp_path / "bridge")
    adapter = MultiverseAdapter(bridge)

    events_high = [
        DummyStepEvent(0, "warehouse_sort", [1.0, 2.0], 1, 5.0, False),
        DummyStepEvent(1, "warehouse_sort", [2.0, 3.0], 0, 10.0, True),
    ]
    adapter.record_rollout(
        events=events_high,
        task="warehouse_sort",
        tags=["logistics", "priority"],
    )

    events_low = [
        DummyStepEvent(0, "warehouse_sort", [1.0, 2.0], 0, 0.5, False),
        DummyStepEvent(1, "warehouse_sort", [1.5, 2.0], 1, 1.0, True),
    ]
    adapter.record_rollout(
        events=events_low,
        task="warehouse_sort",
        tags=["logistics", "slow"],
    )

    assert bridge.count() == 2

    # Query matching high reward
    high_eps = adapter.retrieve_for_verse(
        "warehouse_sort", min_reward=10.0, limit=5
    )
    assert len(high_eps) == 1
    assert high_eps[0].total_reward == 15.0

    # Retrieve ready-to-train transitions
    trans_list = adapter.retrieve_transitions("warehouse_sort", limit=1)
    assert len(trans_list) == 1
    episode_transitions = trans_list[0]
    assert len(episode_transitions) == 2
    assert "obs" in episode_transitions[0]
    assert "next_obs" in episode_transitions[0]
    assert "action" in episode_transitions[0]
    assert episode_transitions[0]["reward"] == 5.0


def test_uas_episode_to_multiverse_events() -> None:
    transitions = [
        DummyTransition([1.0, 2.0], [0.0], 1.0, [2.0, 3.0], False),
        DummyTransition([2.0, 3.0], [1.0], 2.0, [3.0, 4.0], True),
    ]
    episode = multiverse_to_uas_episode(
        transitions=transitions, task="line_world"
    )
    events = uas_episode_to_multiverse_events(episode)

    assert len(events) == 2
    assert events[0]["verse_name"] == "line_world"
    assert events[0]["step_idx"] == 0
    assert events[1]["step_idx"] == 1
    assert events[1]["done"] is True


def test_gymnasium_adapter_roundtrip() -> None:
    length = 5
    traj = Trajectory(
        observations=np.zeros((length + 1, 4), dtype=np.float32),
        actions=np.ones((length, 2), dtype=np.float32),
        rewards=np.ones(length, dtype=np.float32),
        terminated=np.asarray([False] * (length - 1) + [True]),
        truncated=np.zeros(length, dtype=np.bool_),
    )

    ep = gymnasium_to_uas_episode(
        trajectory=traj,
        task="cartpole_balance",
        embodiment="cartpole",
        success=True,
    )
    assert ep.framework == "gymnasium"
    assert ep.steps == length
    assert ep.action_type == "continuous"

    reconstructed_traj = uas_episode_to_trajectory(ep)
    reconstructed_traj.validate()
    assert np.allclose(reconstructed_traj.observations, traj.observations)
    assert np.allclose(reconstructed_traj.actions, traj.actions)


def test_cross_framework_compatibility() -> None:
    # Multiverse episode with (4,) observation and (2,) action
    mv_trans = [
        DummyTransition([0.1, 0.2, 0.3, 0.4], [1.0, 0.0], 1.0, [0.2, 0.3, 0.4, 0.5], True)
    ]
    mv_ep = multiverse_to_uas_episode(
        transitions=mv_trans,
        task="multiverse_nav",
        embodiment="mobile_robot",
    )

    # Gymnasium episode with identical spaces
    gym_ep = gymnasium_to_uas_episode(
        observations=np.zeros((2, 4), dtype=np.float32),
        actions=np.zeros((1, 2), dtype=np.float32),
        rewards=np.zeros(1, dtype=np.float32),
        terminated=np.array([True]),
        truncated=np.array([False]),
        task="gym_nav",
        embodiment="mobile_robot",
    )

    report = mv_ep.check_compatibility(
        other_obs_schema=gym_ep.observation_schema,
        other_action_schema=gym_ep.action_schema,
        other_obs_dim=gym_ep.observation_dim,
        other_act_dim=gym_ep.action_dim,
    )
    assert report.compatible is True
    assert report.observation_compatible is True
    assert report.action_compatible is True
