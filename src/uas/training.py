"""Stable-Baselines3 integration and experiment primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uas.envs import (
    ACTION_SCHEMA,
    OBSERVATION_SCHEMA,
    TASK_FAMILY,
    ContinuousWorldEnv,
    TASKS,
)
from uas.memory import MemoryStore, RetrievedEpisode, TaskQuery
from uas.rollouts import run_episode


@dataclass(frozen=True)
class Evaluation:
    success_rate: float
    mean_reward: float
    mean_steps: float
    mean_collisions: float


def make_model(
    environment: ContinuousWorldEnv,
    *,
    seed: int,
    learning_starts: int = 100,
    buffer_size: int = 50_000,
) -> Any:
    """Create the small SAC model used by every experiment condition."""
    from stable_baselines3 import SAC

    return SAC(
        "MlpPolicy",
        environment,
        learning_rate=1e-3,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=64,
        tau=0.01,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=0,
        seed=seed,
        device="cpu",
    )


def query_for_task(task_id: str) -> TaskQuery:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}")
    task = TASKS[task_id]
    return TaskQuery(
        task_id=task.task_id,
        task_family=TASK_FAMILY,
        task_description=task.description,
        observation_schema=OBSERVATION_SCHEMA,
        action_schema=ACTION_SCHEMA,
        task_vector=tuple(float(value) for value in task.vector()),
    )


def prefill_replay_buffer(
    model: Any,
    store: MemoryStore,
    episodes: list[RetrievedEpisode],
    *,
    max_transitions: int = 10_000,
) -> int:
    """Load retrieved transitions through SB3's public replay-buffer API."""
    if model.replay_buffer is None:
        raise RuntimeError("The SAC replay buffer has not been initialized")
    added = 0
    for episode in episodes:
        trajectory = store.load_trajectory(episode.record)
        for index in range(len(trajectory.actions)):
            if added >= max_transitions:
                return added
            terminated = bool(trajectory.terminated[index])
            truncated = bool(trajectory.truncated[index])
            model.replay_buffer.add(
                trajectory.observations[index][None, :],
                trajectory.observations[index + 1][None, :],
                trajectory.actions[index][None, :],
                np.asarray([trajectory.rewards[index]], dtype=np.float32),
                np.asarray([terminated or truncated], dtype=np.float32),
                [{"TimeLimit.truncated": truncated and not terminated}],
            )
            added += 1
    return added


def evaluate_model(
    model: Any,
    *,
    task_id: str,
    episodes: int,
    seed_start: int,
) -> Evaluation:
    """Evaluate on an environment separate from the training instance."""
    rewards: list[float] = []
    successes: list[float] = []
    steps: list[float] = []
    collisions: list[float] = []
    environment = ContinuousWorldEnv(task_id=task_id)
    try:
        for index in range(episodes):
            def predict(observation: np.ndarray) -> np.ndarray:
                action, _ = model.predict(observation, deterministic=True)
                return np.asarray(action, dtype=np.float32)

            _, summary = run_episode(
                environment,
                predict,
                seed=seed_start + index,
            )
            rewards.append(float(summary["total_reward"]))
            successes.append(float(summary["success"]))
            steps.append(float(summary["steps"]))
            collisions.append(float(summary["collision_count"]))
    finally:
        environment.close()
    return Evaluation(
        success_rate=float(np.mean(successes)),
        mean_reward=float(np.mean(rewards)),
        mean_steps=float(np.mean(steps)),
        mean_collisions=float(np.mean(collisions)),
    )


def safe_reset_memory(directory: str | Path) -> None:
    """Remove only known generated UAS memory artifacts."""
    root = Path(directory).resolve()
    database = root / "memory.sqlite3"
    database.unlink(missing_ok=True)
    for child, pattern in ((root / "trajectories", "*.npz"), (root / "models", "*.zip")):
        if child.is_dir() and child.resolve().parent == root:
            for path in child.glob(pattern):
                if path.is_file() and path.resolve().parent == child.resolve():
                    path.unlink()
