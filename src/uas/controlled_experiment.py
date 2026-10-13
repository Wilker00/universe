"""UAS v0.2 equal-budget memory-transfer benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from uas.controlled_replay import (
    CONTROLLED_MODES,
    SampledTransition,
    prefill_sampled_transitions,
    replay_manifest,
    sample_balanced_transitions,
    select_controlled_episodes,
)
from uas.envs import TASKS, ContinuousWorldEnv
from uas.experiment import _safe_replace, _write_json_atomic
from uas.memory import MemoryStore, RetrievedEpisode
from uas.training import evaluate_model, make_model, query_for_task

DEFAULT_CONTROLLED_TASKS = ("hard_upper", "hard_lower")
SUCCESS_THRESHOLD = 0.80
COMPARISON_FIELDS = (
    "task_id",
    "mode",
    "paired_runs",
    "mean_auc_delta_vs_no_memory",
    "auc_delta_ci95_low",
    "auc_delta_ci95_high",
    "negative_transfer_rate",
    "mean_final_success_delta",
    "mean_first_threshold_step_delta",
    "mean_sustained_threshold_step_delta",
)


@dataclass(frozen=True)
class ControlledRunPlan:
    episodes: list[RetrievedEpisode]
    transitions: list[SampledTransition]
    selection_seed: int
    sampling_seed: int


def run_controlled_experiment(
    memory_directory: str | Path,
    output_directory: str | Path,
    *,
    task_ids: Iterable[str] = DEFAULT_CONTROLLED_TASKS,
    modes: Iterable[str] = CONTROLLED_MODES,
    seeds: Iterable[int] = range(10),
    train_steps: int = 3_000,
    evaluation_interval: int = 250,
    evaluation_episodes: int = 25,
    episode_budget: int = 12,
    replay_transition_budget: int = 200,
    learning_starts: int = 100,
    save_models: bool = False,
) -> Path:
    """Run the v0.2 benchmark with equal replay budgets across memory modes."""
    selected_tasks = tuple(task_ids)
    selected_modes = tuple(modes)
    selected_seeds = tuple(int(seed) for seed in seeds)
    _validate_configuration(
        selected_tasks,
        selected_modes,
        selected_seeds,
        train_steps=train_steps,
        evaluation_interval=evaluation_interval,
        evaluation_episodes=evaluation_episodes,
        episode_budget=episode_budget,
        replay_transition_budget=replay_transition_budget,
        learning_starts=learning_starts,
    )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model_directory = output / "models"
    if save_models:
        model_directory.mkdir(exist_ok=True)

    store = MemoryStore(memory_directory)
    plans: dict[tuple[str, str, int], ControlledRunPlan] = {}
    manifest: dict[str, dict[str, dict[str, Any]]] = {}
    print("Preparing controlled replay samples...")
    for task_id in selected_tasks:
        manifest[task_id] = {}
        query = query_for_task(task_id)
        for mode in selected_modes:
            manifest[task_id][mode] = {}
            for seed in selected_seeds:
                selection_seed = _derived_seed(task_id, mode, seed, "episodes")
                sampling_seed = _derived_seed(task_id, mode, seed, "transitions")
                episodes = select_controlled_episodes(
                    store,
                    query,
                    mode=mode,
                    episode_budget=episode_budget,
                    seed=selection_seed,
                )
                transition_count = (
                    0 if mode == "no_memory" else replay_transition_budget
                )
                transitions = sample_balanced_transitions(
                    store,
                    episodes,
                    transition_budget=transition_count,
                    seed=sampling_seed,
                )
                if mode != "no_memory" and len(transitions) != replay_transition_budget:
                    raise RuntimeError(
                        f"Mode '{mode}' received {len(transitions)} transitions; "
                        f"expected {replay_transition_budget}"
                    )
                plans[(task_id, mode, seed)] = ControlledRunPlan(
                    episodes=episodes,
                    transitions=transitions,
                    selection_seed=selection_seed,
                    sampling_seed=sampling_seed,
                )
                manifest[task_id][mode][str(seed)] = replay_manifest(
                    store,
                    episodes,
                    transitions,
                    selection_seed=selection_seed,
                    sampling_seed=sampling_seed,
                )

    store_records = sorted(store.records(), key=lambda r: r.episode_id)
    store_hasher = hashlib.sha256()
    for r in store_records:
        t = store.load_trajectory(r)
        store_hasher.update(r.episode_id.encode("utf-8"))
        store_hasher.update(r.task_id.encode("utf-8"))
        store_hasher.update(str(r.success).encode("utf-8"))
        store_hasher.update(r.policy_version.encode("utf-8"))
        store_hasher.update(t.observations.tobytes())
        store_hasher.update(t.actions.tobytes())
        store_hasher.update(t.rewards.tobytes())
        store_hasher.update(t.terminated.tobytes())
        store_hasher.update(t.truncated.tobytes())
    memory_hash = store_hasher.hexdigest()

    configuration = {
        "protocol_version": "0.2",
        "tasks": selected_tasks,
        "modes": selected_modes,
        "seeds": selected_seeds,
        "memory_hash": memory_hash,
        "train_steps": train_steps,
        "evaluation_interval": evaluation_interval,
        "evaluation_episodes": evaluation_episodes,
        "success_threshold": SUCCESS_THRESHOLD,
        "episode_budget_per_memory_run": episode_budget,
        "replay_transition_budget_per_memory_run": replay_transition_budget,
        "replay_sampling": "episode-balanced-without-replacement",
        "learning_starts_all_modes": learning_starts,
        "save_policy_checkpoints": save_models,
        "memory_episodes": store.count(),
        "versions": {
            "universe-agent-system": _package_version(
                "universe-agent-system"
            ),
            "gymnasium": _package_version("gymnasium"),
            "stable-baselines3": _package_version("stable-baselines3"),
            "numpy": _package_version("numpy"),
        },
    }
    _write_json_atomic(output / "run_config.json", configuration)
    _write_json_atomic(output / "replay_manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    results_path = output / "learning_curves.csv"
    expected_runs = len(selected_tasks) * len(selected_modes) * len(selected_seeds)
    completed_runs: set[tuple[str, str, int]] = set()
    if results_path.is_file():
        try:
            with results_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                existing_rows = list(reader)
                for r in existing_rows:
                    if int(r["timestep"]) >= train_steps:
                        completed_runs.add(
                            (str(r["task_id"]), str(r["mode"]), int(r["seed"]))
                        )
                rows = [
                    {
                        "task_id": str(r["task_id"]),
                        "mode": str(r["mode"]),
                        "seed": int(r["seed"]),
                        "timestep": int(r["timestep"]),
                        "success_rate": float(r["success_rate"]),
                        "mean_reward": float(r["mean_reward"]),
                        "mean_steps": float(r["mean_steps"]),
                        "mean_collisions": float(r["mean_collisions"]),
                        "selected_episodes": int(r["selected_episodes"]),
                        "selected_successful_episodes": int(
                            r["selected_successful_episodes"]
                        ),
                        "prefilled_transitions": int(r["prefilled_transitions"]),
                    }
                    for r in existing_rows
                    if (str(r["task_id"]), str(r["mode"]), int(r["seed"]))
                    in completed_runs
                ]
                if completed_runs:
                    print(
                        f"Resuming experiment: found {len(completed_runs)} already completed runs."
                    )
        except Exception:
            rows = []
            completed_runs = set()

    for task_id in selected_tasks:
        for mode in selected_modes:
            for seed in selected_seeds:
                if (task_id, mode, seed) in completed_runs:
                    print(
                        f"Skipping completed task='{task_id}', mode='{mode}', seed={seed}."
                    )
                    continue
                plan = plans[(task_id, mode, seed)]
                successful_episodes = sum(
                    item.record.success for item in plan.episodes
                )
                print(
                    f"Running task='{task_id}', mode='{mode}', seed={seed}, "
                    f"episodes={len(plan.episodes)}, "
                    f"replay_transitions={len(plan.transitions)}..."
                )
                environment = ContinuousWorldEnv(task_id=task_id)
                try:
                    model = make_model(
                        environment,
                        seed=seed,
                        learning_starts=learning_starts,
                    )
                    prefilled = prefill_sampled_transitions(
                        model, store, plan.transitions
                    )
                    _append_evaluation(
                        rows,
                        model,
                        task_id=task_id,
                        mode=mode,
                        seed=seed,
                        timestep=0,
                        evaluation_episodes=evaluation_episodes,
                        selected_episodes=len(plan.episodes),
                        selected_successful_episodes=successful_episodes,
                        prefilled_transitions=prefilled,
                    )
                    _checkpoint(
                        output,
                        rows,
                        expected_runs=expected_runs,
                        train_steps=train_steps,
                    )
                    completed = 0
                    while completed < train_steps:
                        chunk = min(
                            evaluation_interval, train_steps - completed
                        )
                        model.learn(
                            total_timesteps=chunk,
                            reset_num_timesteps=False,
                            progress_bar=False,
                        )
                        completed += chunk
                        _append_evaluation(
                            rows,
                            model,
                            task_id=task_id,
                            mode=mode,
                            seed=seed,
                            timestep=completed,
                            evaluation_episodes=evaluation_episodes,
                            selected_episodes=len(plan.episodes),
                            selected_successful_episodes=successful_episodes,
                            prefilled_transitions=prefilled,
                        )
                        _checkpoint(
                            output,
                            rows,
                            expected_runs=expected_runs,
                            train_steps=train_steps,
                        )
                    if save_models:
                        model.save(
                            model_directory
                            / f"{task_id}_{mode}_seed_{seed}"
                        )
                    _checkpoint(
                        output,
                        rows,
                        expected_runs=expected_runs,
                        train_steps=train_steps,
                        include_plot=True,
                    )
                finally:
                    environment.close()

    _checkpoint(
        output,
        rows,
        expected_runs=expected_runs,
        train_steps=train_steps,
        include_plot=True,
    )
    print(f"Wrote controlled experiment results to {output.resolve()}")
    return results_path


def _validate_configuration(
    tasks: tuple[str, ...],
    modes: tuple[str, ...],
    seeds: tuple[int, ...],
    *,
    train_steps: int,
    evaluation_interval: int,
    evaluation_episodes: int,
    episode_budget: int,
    replay_transition_budget: int,
    learning_starts: int,
) -> None:
    unknown_tasks = set(tasks) - set(TASKS)
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {', '.join(sorted(unknown_tasks))}")
    unknown_modes = set(modes) - set(CONTROLLED_MODES)
    if unknown_modes:
        raise ValueError(f"Unknown modes: {', '.join(sorted(unknown_modes))}")
    if not tasks or not modes or not seeds:
        raise ValueError("At least one task, mode, and seed are required")
    if (
        len(tasks) != len(set(tasks))
        or len(modes) != len(set(modes))
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("Tasks, modes, and seeds cannot contain duplicates")
    if train_steps <= 0 or evaluation_interval <= 0 or evaluation_episodes <= 0:
        raise ValueError("Training and evaluation values must be positive")
    if episode_budget <= 0 or replay_transition_budget <= 0:
        raise ValueError("Episode and replay-transition budgets must be positive")
    if replay_transition_budget < episode_budget:
        raise ValueError("Replay-transition budget must cover every selected episode")
    if learning_starts < 0:
        raise ValueError("learning_starts cannot be negative")


def _derived_seed(task_id: str, mode: str, seed: int, purpose: str) -> int:
    value = f"uas-v0.2|{task_id}|{mode}|{seed}|{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def _append_evaluation(
    rows: list[dict[str, Any]],
    model: Any,
    *,
    task_id: str,
    mode: str,
    seed: int,
    timestep: int,
    evaluation_episodes: int,
    selected_episodes: int,
    selected_successful_episodes: int,
    prefilled_transitions: int,
) -> None:
    evaluation = evaluate_model(
        model,
        task_id=task_id,
        episodes=evaluation_episodes,
        seed_start=100_000 + seed * 1_000,
    )
    rows.append(
        {
            "task_id": task_id,
            "mode": mode,
            "seed": seed,
            "timestep": timestep,
            "success_rate": evaluation.success_rate,
            "mean_reward": evaluation.mean_reward,
            "mean_steps": evaluation.mean_steps,
            "mean_collisions": evaluation.mean_collisions,
            "selected_episodes": selected_episodes,
            "selected_successful_episodes": selected_successful_episodes,
            "prefilled_transitions": prefilled_transitions,
        }
    )
    print(
        f"  step={timestep:>6} success={evaluation.success_rate:.2f} "
        f"reward={evaluation.mean_reward:.2f}"
    )


def _write_table_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> None:
    columns = list(fieldnames or (list(rows[0]) if rows else []))
    if not columns:
        raise ValueError("CSV output requires field names")
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        _safe_replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _checkpoint(
    output: Path,
    rows: list[dict[str, Any]],
    *,
    expected_runs: int,
    train_steps: int,
    include_plot: bool = False,
) -> None:
    _write_table_atomic(output / "learning_curves.csv", rows)
    summary, comparisons = summarize_controlled_results(
        rows, train_steps=train_steps
    )
    completed_runs = {
        (str(row["task_id"]), str(row["mode"]), int(row["seed"]))
        for row in rows
        if int(row["timestep"]) >= train_steps
    }
    summary["_checkpoint"] = {
        "evaluation_rows": len(rows),
        "completed_runs": len(completed_runs),
        "expected_runs": expected_runs,
        "complete": len(completed_runs) == expected_runs,
    }
    _write_json_atomic(output / "summary.json", summary)
    _write_table_atomic(
        output / "comparisons.csv",
        comparisons,
        fieldnames=COMPARISON_FIELDS,
    )
    if include_plot:
        temporary_plot = output / ".success_rate.tmp.png"
        try:
            plot_controlled_results(rows, temporary_plot)
            _safe_replace(temporary_plot, output / "success_rate.png")
        finally:
            temporary_plot.unlink(missing_ok=True)


def summarize_controlled_results(
    rows: list[dict[str, Any]],
    *,
    train_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate v0.2 curves and matched negative-transfer comparisons."""
    histories: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        histories[
            (str(row["task_id"]), str(row["mode"]), int(row["seed"]))
        ].append(row)
    metrics = [
        _run_metrics(key, history)
        for key, history in sorted(histories.items())
    ]

    by_task_mode: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        by_task_mode[(metric["task_id"], metric["mode"])].append(metric)
        by_mode[metric["mode"]].append(metric)

    task_summary: dict[str, dict[str, Any]] = defaultdict(dict)
    for (task_id, mode), values in sorted(by_task_mode.items()):
        task_summary[task_id][mode] = _aggregate_metrics(
            values, train_steps=train_steps
        )
    overall = {
        mode: _aggregate_metrics(values, train_steps=train_steps)
        for mode, values in sorted(by_mode.items())
    }
    comparisons = _build_comparisons(metrics, train_steps=train_steps)
    summary: dict[str, Any] = {
        "success_threshold": SUCCESS_THRESHOLD,
        "confidence_interval": "mean plus/minus 1.96 standard errors",
        "negative_transfer_definition": (
            "A matched task/seed run whose normalized success AUC is below "
            "the no_memory run"
        ),
        "tasks": dict(task_summary),
        "overall": overall,
        "comparisons_vs_no_memory": comparisons,
    }
    return summary, comparisons


def _run_metrics(
    key: tuple[str, str, int], history: list[dict[str, Any]]
) -> dict[str, Any]:
    task_id, mode, seed = key
    ordered = sorted(history, key=lambda row: int(row["timestep"]))
    timesteps = np.asarray([int(row["timestep"]) for row in ordered], dtype=float)
    success = np.asarray(
        [float(row["success_rate"]) for row in ordered], dtype=float
    )
    final = ordered[-1]
    first_threshold = next(
        (
            int(row["timestep"])
            for row in ordered
            if float(row["success_rate"]) >= SUCCESS_THRESHOLD
        ),
        None,
    )
    sustained_threshold = next(
        (
            int(ordered[index]["timestep"])
            for index in range(len(ordered))
            if np.all(success[index:] >= SUCCESS_THRESHOLD)
        ),
        None,
    )
    horizon = float(timesteps[-1])
    if horizon <= 0.0:
        auc = 0.0
    elif hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(success, timesteps) / horizon)
    else:  # NumPy 1.26 compatibility.
        auc = float(np.trapz(success, timesteps) / horizon)
    return {
        "task_id": task_id,
        "mode": mode,
        "seed": seed,
        "final_timestep": int(final["timestep"]),
        "final_success_rate": float(final["success_rate"]),
        "final_reward": float(final["mean_reward"]),
        "final_collisions": float(final["mean_collisions"]),
        "best_success_rate": float(np.max(success)),
        "normalized_success_auc": auc,
        "first_threshold_steps": first_threshold,
        "sustained_threshold_steps": sustained_threshold,
    }


def _aggregate_metrics(
    metrics: list[dict[str, Any]], *, train_steps: int
) -> dict[str, Any]:
    complete = [
        item for item in metrics if int(item["final_timestep"]) >= train_steps
    ]
    first = [
        float(item["first_threshold_steps"])
        for item in metrics
        if item["first_threshold_steps"] is not None
    ]
    sustained = [
        float(item["sustained_threshold_steps"])
        for item in metrics
        if item["sustained_threshold_steps"] is not None
    ]
    return {
        "runs": len(metrics),
        "complete_runs": len(complete),
        "final_timestep": max(int(item["final_timestep"]) for item in metrics),
        "final_success_rate": _mean_ci(
            [float(item["final_success_rate"]) for item in metrics],
            lower=0.0,
            upper=1.0,
        ),
        "final_reward": _mean_ci(
            [float(item["final_reward"]) for item in metrics]
        ),
        "final_collisions": _mean_ci(
            [float(item["final_collisions"]) for item in metrics], lower=0.0
        ),
        "best_success_rate": _mean_ci(
            [float(item["best_success_rate"]) for item in metrics],
            lower=0.0,
            upper=1.0,
        ),
        "normalized_success_auc": _mean_ci(
            [float(item["normalized_success_auc"]) for item in metrics],
            lower=0.0,
            upper=1.0,
        ),
        "first_threshold": {
            "runs_reaching": len(first),
            "mean_steps": None if not first else float(np.mean(first)),
        },
        "sustained_threshold": {
            "runs_reaching": len(sustained),
            "mean_steps": None if not sustained else float(np.mean(sustained)),
        },
    }


def _mean_ci(
    values: list[float],
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    margin = (
        0.0
        if len(array) < 2
        else float(1.96 * np.std(array, ddof=1) / np.sqrt(len(array)))
    )
    low = mean - margin
    high = mean + margin
    if lower is not None:
        low = max(lower, low)
    if upper is not None:
        high = min(upper, high)
    return {
        "mean": mean,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "samples": len(array),
    }


def _build_comparisons(
    metrics: list[dict[str, Any]], *, train_steps: int
) -> list[dict[str, Any]]:
    complete = [
        item for item in metrics if int(item["final_timestep"]) >= train_steps
    ]
    by_key = {
        (str(item["task_id"]), str(item["mode"]), int(item["seed"])): item
        for item in complete
    }
    tasks = sorted({str(item["task_id"]) for item in complete})
    modes = sorted(
        {str(item["mode"]) for item in complete if item["mode"] != "no_memory"}
    )
    comparisons: list[dict[str, Any]] = []
    overall_pairs: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for task_id in tasks:
        for mode in modes:
            pairs = []
            seeds = sorted(
                int(item["seed"])
                for item in complete
                if item["task_id"] == task_id and item["mode"] == mode
            )
            for seed in seeds:
                baseline = by_key.get((task_id, "no_memory", seed))
                candidate = by_key.get((task_id, mode, seed))
                if baseline is not None and candidate is not None:
                    pairs.append((baseline, candidate))
            if pairs:
                comparisons.append(_comparison_row(task_id, mode, pairs))
                overall_pairs[mode].extend(pairs)
    for mode, pairs in sorted(overall_pairs.items()):
        comparisons.append(_comparison_row("__overall__", mode, pairs))
    return comparisons


def _comparison_row(
    task_id: str,
    mode: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    auc_deltas = [
        float(candidate["normalized_success_auc"])
        - float(baseline["normalized_success_auc"])
        for baseline, candidate in pairs
    ]
    final_deltas = [
        float(candidate["final_success_rate"])
        - float(baseline["final_success_rate"])
        for baseline, candidate in pairs
    ]
    first_deltas = _threshold_deltas(pairs, "first_threshold_steps")
    sustained_deltas = _threshold_deltas(pairs, "sustained_threshold_steps")
    auc_summary = _mean_ci(auc_deltas, lower=-1.0, upper=1.0)
    return {
        "task_id": task_id,
        "mode": mode,
        "paired_runs": len(pairs),
        "mean_auc_delta_vs_no_memory": auc_summary["mean"],
        "auc_delta_ci95_low": auc_summary["ci95_low"],
        "auc_delta_ci95_high": auc_summary["ci95_high"],
        "negative_transfer_rate": float(np.mean(np.asarray(auc_deltas) < 0.0)),
        "mean_final_success_delta": float(np.mean(final_deltas)),
        "mean_first_threshold_step_delta": (
            None if not first_deltas else float(np.mean(first_deltas))
        ),
        "mean_sustained_threshold_step_delta": (
            None if not sustained_deltas else float(np.mean(sustained_deltas))
        ),
    }


def _threshold_deltas(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], field: str
) -> list[float]:
    return [
        float(candidate[field]) - float(baseline[field])
        for baseline, candidate in pairs
        if baseline[field] is not None and candidate[field] is not None
    ]


def plot_controlled_results(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot one mean success curve panel per held-out task."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted({str(row["task_id"]) for row in rows})
    figure, axes = plt.subplots(
        len(tasks),
        1,
        figsize=(9, max(5, 4 * len(tasks))),
        squeeze=False,
        sharex=True,
    )
    for axis, task_id in zip(axes[:, 0], tasks, strict=True):
        grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in rows:
            if str(row["task_id"]) == task_id:
                grouped[(str(row["mode"]), int(row["timestep"]))].append(
                    float(row["success_rate"])
                )
        for mode in CONTROLLED_MODES:
            points = sorted(
                (timestep, values)
                for (row_mode, timestep), values in grouped.items()
                if row_mode == mode
            )
            if not points:
                continue
            x = np.asarray([point[0] for point in points], dtype=np.int64)
            means = np.asarray([np.mean(point[1]) for point in points])
            deviations = np.asarray([np.std(point[1]) for point in points])
            axis.plot(x, means, marker="o", label=mode)
            axis.fill_between(
                x,
                np.clip(means - deviations, 0.0, 1.0),
                np.clip(means + deviations, 0.0, 1.0),
                alpha=0.12,
            )
        axis.axhline(SUCCESS_THRESHOLD, color="black", linestyle="--", alpha=0.35)
        axis.set_title(task_id)
        axis.set_ylabel("Success rate")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")
    axes[-1, 0].set_xlabel("Environment steps")
    figure.suptitle("UAS v0.2 equal-budget memory comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "development"


__all__ = [
    "DEFAULT_CONTROLLED_TASKS",
    "run_controlled_experiment",
    "summarize_controlled_results",
]
