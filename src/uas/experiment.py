"""End-to-end memory construction and held-out task experiments."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from uas.envs import ContinuousWorldEnv
from uas.memory import MemoryStore
from uas.rollouts import collect_policy_episodes, collect_random_episodes
from uas.training import (
    evaluate_model,
    make_model,
    prefill_replay_buffer,
    query_for_task,
)

VALID_MODES = ("no_memory", "naive", "similarity", "governed")
SUCCESS_THRESHOLD = 0.80


def build_source_memory(
    memory_directory: str | Path,
    *,
    tasks: Iterable[str] = ("open", "upper_gap", "lower_gap"),
    train_steps: int = 5_000,
    random_episodes: int = 8,
    policy_episodes: int = 8,
    seed: int = 0,
) -> MemoryStore:
    """Train source-task policies and store good and poor episodes."""
    store = MemoryStore(memory_directory)
    for task_index, task_id in enumerate(tasks):
        task_seed = seed + task_index * 10_000
        print(f"Training source task '{task_id}' for {train_steps} steps...")
        environment = ContinuousWorldEnv(task_id=task_id)
        try:
            collect_random_episodes(
                store,
                environment,
                count=random_episodes,
                seed_start=task_seed + 1_000,
            )
            model = make_model(environment, seed=task_seed)
            model.learn(total_timesteps=train_steps, progress_bar=False)
            model_path = store.model_directory / f"sac_{task_id}_seed_{task_seed}"
            model.save(model_path)
            saved_model = model_path.with_suffix(".zip")
            records = collect_policy_episodes(
                store,
                environment,
                model,
                count=policy_episodes,
                seed_start=task_seed + 2_000,
                policy_reference=saved_model,
            )
            successes = sum(record.success for record in records)
            print(
                f"Stored {random_episodes + policy_episodes} '{task_id}' episodes "
                f"({successes}/{policy_episodes} trained-policy successes)."
            )
        finally:
            environment.close()
    return store


def run_experiment(
    memory_directory: str | Path,
    output_directory: str | Path,
    *,
    task_id: str = "held_out",
    modes: Iterable[str] = VALID_MODES,
    seeds: Iterable[int] = (0, 1, 2),
    train_steps: int = 5_000,
    evaluation_interval: int = 250,
    evaluation_episodes: int = 10,
    retrieval_limit: int = 100,
    max_prefill_transitions: int = 10_000,
) -> Path:
    """Compare memory strategies on a held-out task and write results."""
    selected_modes = tuple(modes)
    selected_seeds = tuple(int(seed) for seed in seeds)
    invalid_modes = set(selected_modes) - set(VALID_MODES)
    if invalid_modes:
        raise ValueError(f"Unknown modes: {', '.join(sorted(invalid_modes))}")
    if evaluation_interval <= 0 or train_steps <= 0:
        raise ValueError("Training steps and evaluation interval must be positive")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model_directory = output / "models"
    model_directory.mkdir(exist_ok=True)
    store = MemoryStore(memory_directory)
    query = query_for_task(task_id)
    rows: list[dict[str, Any]] = []
    retrieval_manifest: dict[str, list[dict[str, Any]]] = {}

    configuration = {
        "task_id": task_id,
        "modes": selected_modes,
        "seeds": selected_seeds,
        "train_steps": train_steps,
        "evaluation_interval": evaluation_interval,
        "evaluation_episodes": evaluation_episodes,
        "retrieval_limit": retrieval_limit,
        "max_prefill_transitions": max_prefill_transitions,
        "memory_episodes": store.count(),
        "versions": {
            "universe-agent-system": _package_version("universe-agent-system"),
            "gymnasium": _package_version("gymnasium"),
            "stable-baselines3": _package_version("stable-baselines3"),
            "numpy": _package_version("numpy"),
        },
    }
    (output / "run_config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )

    for mode in selected_modes:
        retrieved = store.retrieve(query, mode=mode, limit=retrieval_limit)
        retrieval_manifest[mode] = [
            {
                "episode_id": item.record.episode_id,
                "source_task": item.record.task_id,
                "agent_id": item.record.agent_id,
                "success": item.record.success,
                "normalized_score": item.record.normalized_score,
                "similarity": item.similarity,
                "retrieval_score": item.retrieval_score,
                "steps": item.record.steps,
                "policy_reference": item.record.policy_reference,
            }
            for item in retrieved
        ]
        for seed in selected_seeds:
            print(
                f"Running mode='{mode}', seed={seed}, "
                f"retrieved_episodes={len(retrieved)}..."
            )
            environment = ContinuousWorldEnv(task_id=task_id)
            try:
                model = make_model(
                    environment,
                    seed=seed,
                    learning_starts=0 if retrieved else 100,
                )
                prefilled = prefill_replay_buffer(
                    model,
                    store,
                    retrieved,
                    max_transitions=max_prefill_transitions,
                )
                _append_evaluation(
                    rows,
                    model,
                    task_id=task_id,
                    mode=mode,
                    seed=seed,
                    timestep=0,
                    evaluation_episodes=evaluation_episodes,
                    retrieved_episodes=len(retrieved),
                    prefilled_transitions=prefilled,
                )
                completed = 0
                while completed < train_steps:
                    chunk = min(evaluation_interval, train_steps - completed)
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
                        retrieved_episodes=len(retrieved),
                        prefilled_transitions=prefilled,
                    )
                model.save(model_directory / f"{task_id}_{mode}_seed_{seed}")
            finally:
                environment.close()

    results_path = output / "learning_curves.csv"
    _write_csv(results_path, rows)
    summary = summarize_results(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "retrieval_manifest.json").write_text(
        json.dumps(retrieval_manifest, indent=2) + "\n", encoding="utf-8"
    )
    plot_results(rows, output / "success_rate.png")
    print(f"Wrote experiment results to {output.resolve()}")
    return results_path


def _append_evaluation(
    rows: list[dict[str, Any]],
    model: Any,
    *,
    task_id: str,
    mode: str,
    seed: int,
    timestep: int,
    evaluation_episodes: int,
    retrieved_episodes: int,
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
            "retrieved_episodes": retrieved_episodes,
            "prefilled_transitions": prefilled_transitions,
        }
    )
    print(
        f"  step={timestep:>6} success={evaluation.success_rate:.2f} "
        f"reward={evaluation.mean_reward:.2f}"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No experiment rows were produced")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the last evaluation from every mode and seed."""
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    histories: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["mode"]), int(row["seed"]))
        histories[key].append(row)
        if key not in latest or int(row["timestep"]) > int(latest[key]["timestep"]):
            latest[key] = row
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (mode, _), row in latest.items():
        by_mode[mode].append(row)
    summary: dict[str, Any] = {}
    for mode, values in by_mode.items():
        steps_to_threshold: list[int] = []
        best_success_rates: list[float] = []
        for row in values:
            key = (mode, int(row["seed"]))
            history = sorted(histories[key], key=lambda item: int(item["timestep"]))
            best_success_rates.append(
                max(float(item["success_rate"]) for item in history)
            )
            reached = next(
                (
                    int(item["timestep"])
                    for item in history
                    if float(item["success_rate"]) >= SUCCESS_THRESHOLD
                ),
                None,
            )
            if reached is not None:
                steps_to_threshold.append(reached)
        summary[mode] = {
            "seeds": len(values),
            "final_timestep": max(int(value["timestep"]) for value in values),
            "mean_success_rate": float(
                np.mean([float(value["success_rate"]) for value in values])
            ),
            "mean_reward": float(
                np.mean([float(value["mean_reward"]) for value in values])
            ),
            "mean_collisions": float(
                np.mean([float(value["mean_collisions"]) for value in values])
            ),
            "mean_best_success_rate": float(np.mean(best_success_rates)),
            "success_threshold": SUCCESS_THRESHOLD,
            "seeds_reaching_threshold": len(steps_to_threshold),
            "mean_steps_to_threshold": (
                None
                if not steps_to_threshold
                else float(np.mean(steps_to_threshold))
            ),
        }
    return summary


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "development"


def plot_results(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a compact mean learning-curve plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode"]), int(row["timestep"]))].append(
            float(row["success_rate"])
        )

    figure, axis = plt.subplots(figsize=(8, 5))
    for mode in VALID_MODES:
        points = sorted(
            (timestep, values)
            for (row_mode, timestep), values in grouped.items()
            if row_mode == mode
        )
        if not points:
            continue
        x = np.asarray([point[0] for point in points], dtype=np.int64)
        means = np.asarray([np.mean(point[1]) for point in points], dtype=float)
        standard_deviations = np.asarray(
            [np.std(point[1]) for point in points], dtype=float
        )
        axis.plot(x, means, marker="o", label=mode)
        axis.fill_between(
            x,
            np.clip(means - standard_deviations, 0.0, 1.0),
            np.clip(means + standard_deviations, 0.0, 1.0),
            alpha=0.15,
        )
    axis.set_title("UAS memory comparison on the held-out task")
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Success rate")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
