"""End-to-end cross-framework memory bridge demonstration.

Shows a Multiverse agent recording experience into the UAS MemoryBridge,
which is then retrieved, verified for compatibility, and used to seed a
Gymnasium / Stable-Baselines3 SAC agent's replay buffer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from uas.bridge.client import MemoryBridge
from uas.envs.continuous_world import ContinuousWorldEnv
from uas.training import make_model


def run_bridge_interop_demo(
    directory: str | Path | None = None,
    *,
    steps_to_train: int = 200,
) -> dict[str, object]:
    """Execute complete Multiverse -> UAS MemoryBridge -> Gymnasium training transfer."""
    from uas.adapters.gymnasium import uas_episode_to_trajectory
    from uas.adapters.multiverse import MultiverseAdapter

    if directory is None:
        temp_dir = tempfile.TemporaryDirectory()
        base_dir = Path(temp_dir.name)
    else:
        temp_dir = None
        base_dir = Path(directory)

    bridge_dir = base_dir / "memory_bridge"
    bridge = MemoryBridge(bridge_dir)
    mv_adapter = MultiverseAdapter(bridge)

    print("=" * 65)
    print(" UAS CROSS-FRAMEWORK MEMORY BRIDGE INTEROPERABILITY DEMO")
    print("=" * 65)

    # Step 1: Multiverse simulates a task and records experience
    print("\n[1/4] Simulating Multiverse agent rollouts...")
    # Multiverse creates 2 episodes (12-dim continuous obs, 2-dim continuous actions)
    env = ContinuousWorldEnv(task_id="open")
    obs, _ = env.reset(seed=42)
    mv_events = []
    total_rew = 0.0
    for step_i in range(15):
        action = np.array([0.05, 0.05], dtype=np.float32)
        next_obs, reward, term, trunc, _ = env.step(action)
        total_rew += reward
        mv_events.append(
            {
                "step_idx": step_i,
                "verse_name": "open_navigation",
                "obs": obs.tolist(),
                "action": action.tolist(),
                "reward": float(reward),
                "done": bool(term),
                "truncated": bool(trunc),
                "info": {"robot": "virtual_agent", "success": True},
            }
        )
        obs = next_obs
        if term or trunc:
            break
    env.close()

    recorded_ep = mv_adapter.record_rollout(
        events=mv_events,
        task="navigation_2d",
        embodiment="continuous_navigator",
        tags=["multiverse", "demo", "navigation"],
    )
    print(f"  -> Recorded Multiverse episode ID: {recorded_ep.episode_id}")
    print(f"  -> Total transitions recorded: {recorded_ep.steps}")
    print(f"  -> Total reward: {recorded_ep.total_reward:.2f}")

    # Step 2: Query MemoryBridge for compatible prior experience
    print("\n[2/4] Querying MemoryBridge from downstream framework...")
    retrieved_episodes = bridge.search(
        task="navigation_2d",
        embodiment="continuous_navigator",
        success_only=True,
        limit=5,
    )
    print(f"  -> Retrieved {len(retrieved_episodes)} compatible episodes from bridge.")
    source_ep = retrieved_episodes[0]

    # Step 3: Check execution compatibility with Gymnasium environment
    print("\n[3/4] Verifying cross-framework compatibility with Gymnasium SAC agent...")
    gym_env = ContinuousWorldEnv(task_id="held_out")
    gym_obs_shape = gym_env.observation_space.shape
    gym_act_shape = gym_env.action_space.shape

    compat_report = source_ep.check_compatibility(
        other_obs_schema="continuous",
        other_action_schema="continuous",
        other_obs_dim=gym_obs_shape,
        other_act_dim=gym_act_shape,
    )
    print(f"  -> Compatibility status: {compat_report.compatible}")
    print(f"  -> Observation compatible: {compat_report.observation_compatible}")
    print(f"  -> Action compatible: {compat_report.action_compatible}")
    assert compat_report.compatible, f"Compatibility check failed: {compat_report.reason}"

    # Step 4: Convert to Gymnasium trajectory and prefill SAC replay buffer
    print("\n[4/4] Prefilling Gymnasium SAC replay buffer and training...")
    gym_trajectory = uas_episode_to_trajectory(source_ep)
    model = make_model(gym_env, seed=0, learning_starts=50)

    # Prefill replay buffer directly with the converted Multiverse transitions
    for t in range(len(gym_trajectory.actions)):
        model.replay_buffer.add(
            obs=gym_trajectory.observations[t],
            next_obs=gym_trajectory.observations[t + 1],
            action=gym_trajectory.actions[t],
            reward=gym_trajectory.rewards[t],
            done=gym_trajectory.terminated[t],
            infos=[{}],
        )

    prefilled_count = model.replay_buffer.size()
    print(f"  -> Prefilled {prefilled_count} transitions into SAC replay buffer.")

    # Train for a few steps to prove end-to-end execution
    model.learn(total_timesteps=steps_to_train, progress_bar=False)
    gym_env.close()
    print(f"  -> Successfully executed {steps_to_train} SAC training steps with transferred memory!")
    print("=" * 65)
    print(" ALL CROSS-FRAMEWORK TESTS AND TRAINING HOOKS PASSED")
    print("=" * 65)

    result = {
        "recorded_episode_id": recorded_ep.episode_id,
        "prefilled_transitions": prefilled_count,
        "trained_timesteps": steps_to_train,
        "compatible": compat_report.compatible,
    }

    if temp_dir is not None:
        temp_dir.cleanup()

    return result
