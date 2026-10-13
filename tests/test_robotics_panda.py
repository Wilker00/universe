from pathlib import Path

import numpy as np
import pytest

from uas.adapters.robotics import (
    panda_gym_to_uas_episode,
    uas_episode_to_panda_transitions,
)
from uas.bridge.client import MemoryBridge
from uas.envs.robotics_panda import PANDA_TASKS, PandaManipulationEnv, make_panda_env


def test_panda_manipulation_env_reach() -> None:
    env = make_panda_env("panda_reach", max_steps=20)
    obs, info = env.reset(seed=42)

    assert obs.shape == (13,)
    assert "is_success" in info
    assert env.action_space.shape == (4,)

    # Step towards target
    action = np.array([0.5, 0.5, 0.2, -1.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert next_obs.shape == (13,)
    assert isinstance(reward, float)
    assert not truncated

    # Render check
    env.render_mode = "rgb_array"
    frame = env.render()
    assert frame is not None
    assert frame.shape == (64, 64, 3)
    env.close()


def test_panda_push_and_pick_place_mechanics() -> None:
    push_env = make_panda_env("panda_push", max_steps=10)
    obs, _ = push_env.reset(seed=1)
    assert obs.shape == (13,)
    push_env.close()

    pick_env = make_panda_env("panda_pick_and_place", max_steps=10)
    obs, _ = pick_env.reset(seed=2)
    # Action closing gripper
    close_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    for _ in range(3):
        obs, rew, term, trunc, _ = pick_env.step(close_action)
    assert obs[6] < 0.08  # Gripper width narrowed
    pick_env.close()


def test_panda_rollout_conversion_and_bridge_record(tmp_path: Path) -> None:
    env = make_panda_env("panda_reach", max_steps=15)
    obs, _ = env.reset(seed=123)

    observations = [obs]
    actions = []
    rewards = []
    terms = []
    truncs = []

    for _ in range(5):
        act = np.array([0.1, 0.1, 0.0, 0.0], dtype=np.float32)
        next_obs, r, term, trunc, _ = env.step(act)
        observations.append(next_obs)
        actions.append(act)
        rewards.append(r)
        terms.append(term)
        truncs.append(trunc)
        if term or trunc:
            break
    env.close()

    ep = panda_gym_to_uas_episode(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminated=terms,
        truncated=truncs,
        task="panda_reach",
        success=True,
        tags=["3d_robotics", "franka_panda"],
    )

    assert ep.framework == "panda_gym"
    assert ep.embodiment == "franka_panda"
    assert ep.steps == len(actions)
    assert ep.action_dim == (4,)
    assert ep.observation_dim == (13,)
    assert "franka_panda" in ep.metadata.get("tags", [])

    # Record into MemoryBridge
    bridge = MemoryBridge(tmp_path / "bridge")
    ep_id = bridge.record(ep)
    assert ep_id == ep.episode_id

    # Retrieve from bridge
    results = bridge.search(embodiment="franka_panda")
    assert len(results) == 1
    loaded = results[0]
    assert loaded.task == "panda_reach"

    # Convert to panda transitions
    transitions = uas_episode_to_panda_transitions(loaded)
    assert len(transitions) == len(actions)
    assert "obs" in transitions[0]
    assert "action" in transitions[0]
    assert len(transitions[0]["action"]) == 4
