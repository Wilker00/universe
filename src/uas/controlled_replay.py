"""Controlled episode selection and equal-budget replay sampling for v0.2."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from uas.memory import MemoryStore, RetrievedEpisode, TaskQuery

CONTROLLED_MODES = (
    "no_memory",
    "random",
    "naive",
    "similarity",
    "governed",
    "governed_diverse",
)


@dataclass(frozen=True)
class SampledTransition:
    """A reproducible reference to one transition in a stored episode."""

    episode: RetrievedEpisode
    transition_index: int


def select_diverse_governed_episodes(
    candidates: list[RetrievedEpisode],
    budget: int,
    diversity_weight: float = 0.5,
) -> list[RetrievedEpisode]:
    """Select episodes balancing task retrieval score with task-space diversity (MMR)."""
    if len(candidates) <= budget:
        return list(candidates)

    selected: list[RetrievedEpisode] = []
    remaining = list(candidates)

    first = max(remaining, key=lambda item: item.retrieval_score)
    selected.append(first)
    remaining.remove(first)

    while len(selected) < budget and remaining:
        best_item = None
        best_score = -float("inf")

        for item in remaining:
            vec_item = np.asarray(item.record.task_vector, dtype=np.float32)
            min_dist = min(
                float(
                    np.linalg.norm(
                        vec_item
                        - np.asarray(s.record.task_vector, dtype=np.float32)
                    )
                )
                for s in selected
            )
            mmr_score = (
                1.0 - diversity_weight
            ) * item.retrieval_score + diversity_weight * min_dist
            if mmr_score > best_score:
                best_score = mmr_score
                best_item = item

        if best_item is not None:
            selected.append(best_item)
            remaining.remove(best_item)

    return selected


def select_controlled_episodes(
    store: MemoryStore,
    query: TaskQuery,
    *,
    mode: str,
    episode_budget: int,
    seed: int,
) -> list[RetrievedEpisode]:
    """Select an exact episode budget according to one v0.2 condition."""
    if mode not in CONTROLLED_MODES:
        raise ValueError(f"Unsupported controlled mode: {mode}")
    if mode == "no_memory":
        return []
    if episode_budget <= 0:
        raise ValueError("Episode budget must be positive for memory conditions")

    compatible = store.retrieve(query, mode="naive", limit=store.count())
    random = np.random.default_rng(seed)
    if mode == "random":
        candidates = [
            item
            for item in compatible
            if item.record.policy_version.startswith("random")
        ]
        random.shuffle(candidates)
    elif mode == "naive":
        candidates = list(compatible)
        random.shuffle(candidates)
    elif mode == "similarity":
        candidates = store.retrieve(
            query, mode="similarity", limit=store.count()
        )
    elif mode == "governed_diverse":
        governed_candidates = store.retrieve(
            query, mode="governed", limit=store.count()
        )
        candidates = select_diverse_governed_episodes(
            governed_candidates, budget=episode_budget
        )
    else:
        candidates = store.retrieve(query, mode="governed", limit=store.count())

    if len(candidates) < episode_budget:
        raise ValueError(
            f"Mode '{mode}' has {len(candidates)} eligible episodes, "
            f"but the controlled budget requires {episode_budget}"
        )
    return candidates[:episode_budget]


def sample_balanced_transitions(
    store: MemoryStore,
    episodes: list[RetrievedEpisode],
    *,
    transition_budget: int,
    seed: int,
) -> list[SampledTransition]:
    """Sample an exact transition budget with near-equal episode contribution."""
    if transition_budget < 0:
        raise ValueError("Transition budget cannot be negative")
    if not episodes:
        if transition_budget:
            raise ValueError("Cannot sample transitions without episodes")
        return []
    if transition_budget < len(episodes):
        raise ValueError(
            "Transition budget must be at least the selected episode count"
        )

    quotient, remainder = divmod(transition_budget, len(episodes))
    quotas = [quotient + int(index < remainder) for index in range(len(episodes))]
    random = np.random.default_rng(seed)
    sampled: list[SampledTransition] = []
    for episode, quota in zip(episodes, quotas, strict=True):
        trajectory = store.load_trajectory(episode.record)
        available = len(trajectory.actions)
        if available < quota:
            raise ValueError(
                f"Episode {episode.record.episode_id} has {available} transitions, "
                f"but its balanced quota is {quota}. Lower the replay budget or "
                "collect longer source episodes."
            )
        indices = random.choice(available, size=quota, replace=False)
        sampled.extend(
            SampledTransition(episode=episode, transition_index=int(index))
            for index in indices
        )
    random.shuffle(sampled)
    return sampled


def prefill_sampled_transitions(
    model: Any,
    store: MemoryStore,
    sampled: list[SampledTransition],
) -> int:
    """Add a previously sampled transition set to an SB3 replay buffer."""
    if model.replay_buffer is None:
        raise RuntimeError("The SAC replay buffer has not been initialized")
    trajectories = {
        item.episode.record.episode_id: store.load_trajectory(item.episode.record)
        for item in sampled
    }
    for item in sampled:
        trajectory = trajectories[item.episode.record.episode_id]
        index = item.transition_index
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
    return len(sampled)


def replay_manifest(
    store: MemoryStore,
    episodes: list[RetrievedEpisode],
    sampled: list[SampledTransition],
    *,
    selection_seed: int,
    sampling_seed: int,
) -> dict[str, Any]:
    """Return an audit record for one controlled mode/task/seed run."""
    indices: dict[str, list[int]] = defaultdict(list)
    for item in sampled:
        indices[item.episode.record.episode_id].append(item.transition_index)
    counts = Counter(item.episode.record.episode_id for item in sampled)
    episode_rows = []
    for item in episodes:
        record = item.record
        episode_rows.append(
            {
                "episode_id": record.episode_id,
                "source_task": record.task_id,
                "agent_id": record.agent_id,
                "policy_version": record.policy_version,
                "success": record.success,
                "normalized_score": record.normalized_score,
                "similarity": item.similarity,
                "retrieval_score": item.retrieval_score,
                "available_transitions": len(store.load_trajectory(record).actions),
                "sampled_transitions": counts[record.episode_id],
                "transition_indices": sorted(indices[record.episode_id]),
            }
        )
    return {
        "selection_seed": selection_seed,
        "sampling_seed": sampling_seed,
        "selected_episodes": len(episodes),
        "selected_successful_episodes": sum(
            item.record.success for item in episodes
        ),
        "sampled_transitions": len(sampled),
        "episodes": episode_rows,
    }


__all__ = [
    "CONTROLLED_MODES",
    "SampledTransition",
    "prefill_sampled_transitions",
    "replay_manifest",
    "sample_balanced_transitions",
    "select_controlled_episodes",
]
