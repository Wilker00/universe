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
    "random_uniform",
    "stratified_success",
    "random_concentrated",
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
    """Select episodes balancing task retrieval score with task-space diversity (MMR).

    Note: With 8 identical top-layout successful episodes and diversity_weight=0.5
    normalized by sqrt(8), MMR picks the top candidate, selects 1 candidate from each
    remaining distinct layout to maximize novelty, and fills the remaining slots greedily.
    """
    if len(candidates) <= budget:
        return list(candidates)

    selected: list[RetrievedEpisode] = []
    remaining = list(candidates)

    first = max(remaining, key=lambda item: item.retrieval_score)
    selected.append(first)
    remaining.remove(first)

    norm_const = np.sqrt(8.0)

    while len(selected) < budget and remaining:
        best_item = None
        best_score = -float("inf")

        for item in remaining:
            vec_item = np.asarray(item.record.task_vector, dtype=np.float32)
            raw_min_dist = min(
                float(
                    np.linalg.norm(
                        vec_item
                        - np.asarray(s.record.task_vector, dtype=np.float32)
                    )
                )
                for s in selected
            )
            min_dist = raw_min_dist / norm_const
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
    elif mode == "random_uniform":
        candidates = list(compatible)
        random.shuffle(candidates)
    elif mode == "stratified_success":
        source_layouts = ("open", "upper_gap", "lower_gap")
        quota_per_layout = episode_budget // len(source_layouts)
        if episode_budget % len(source_layouts) != 0:
            raise ValueError(
                f"Episode budget {episode_budget} must be evenly divisible by {len(source_layouts)} for stratified_success"
            )
        by_layout: dict[str, list[RetrievedEpisode]] = {k: [] for k in source_layouts}
        for item in compatible:
            if item.record.success and item.record.task_id in by_layout:
                by_layout[item.record.task_id].append(item)

        selected: list[RetrievedEpisode] = []
        for layout in source_layouts:
            layout_candidates = by_layout[layout]
            if len(layout_candidates) < quota_per_layout:
                raise ValueError(
                    f"Mode 'stratified_success' requires {quota_per_layout} successful episodes for layout '{layout}', "
                    f"but only found {len(layout_candidates)}."
                )
            random.shuffle(layout_candidates)
            selected.extend(layout_candidates[:quota_per_layout])
        return selected
    elif mode == "random_concentrated":
        primary_quota = int(round(episode_budget * (8.0 / 12.0)))
        secondary_quota = episode_budget - primary_quota

        query_vec = np.asarray(query.task_vector, dtype=np.float32)
        random_candidates = [
            item for item in compatible
            if item.record.policy_version.startswith("random")
        ]
        by_layout: dict[str, list[RetrievedEpisode]] = {}
        layout_mean_vec: dict[str, np.ndarray] = {}
        for item in random_candidates:
            by_layout.setdefault(item.record.task_id, []).append(item)
            vec = np.asarray(item.record.task_vector, dtype=np.float32)
            if item.record.task_id not in layout_mean_vec:
                layout_mean_vec[item.record.task_id] = vec

        if len(layout_mean_vec) < 2:
            raise ValueError(
                f"Mode 'random_concentrated' requires at least 2 candidate layouts, but found {len(layout_mean_vec)}"
            )

        ranked_layouts = sorted(
            layout_mean_vec.keys(),
            key=lambda lid: float(np.linalg.norm(query_vec - layout_mean_vec[lid])),
        )
        primary_layout = ranked_layouts[0]
        secondary_layout = ranked_layouts[1]

        primary_pool = by_layout.get(primary_layout, [])
        secondary_pool = by_layout.get(secondary_layout, [])

        if len(primary_pool) < primary_quota:
            raise ValueError(
                f"Mode 'random_concentrated' requires {primary_quota} random episodes from layout '{primary_layout}', "
                f"but only found {len(primary_pool)}."
            )
        if len(secondary_pool) < secondary_quota:
            raise ValueError(
                f"Mode 'random_concentrated' requires {secondary_quota} random episodes from layout '{secondary_layout}', "
                f"but only found {len(secondary_pool)}."
            )

        random.shuffle(primary_pool)
        random.shuffle(secondary_pool)
        return primary_pool[:primary_quota] + secondary_pool[:secondary_quota]
    elif mode == "similarity":
        candidates = store.retrieve(
            query, mode="similarity", limit=store.count()
        )
    elif mode == "governed_diverse":
        governed_candidates = store.retrieve(
            query, mode="governed", limit=store.count()
        )
        candidates = select_diverse_governed_episodes(
            governed_candidates,
            budget=episode_budget,
            diversity_weight=0.5,
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
