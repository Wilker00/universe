"""Command-line interface for UAS research experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from uas.controlled_experiment import (
    DEFAULT_CONTROLLED_TASKS,
    run_controlled_experiment,
)
from uas.controlled_replay import CONTROLLED_MODES
from uas.envs import GYM_ENV_ID, TASKS, ContinuousWorldEnv
from uas.experiment import VALID_MODES, build_source_memory, run_experiment
from uas.memory import MemoryStore
from uas.training import safe_reset_memory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uas",
        description="Universe Agent System research prototype",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Validate the custom environment")
    check.add_argument("--task", choices=sorted(TASKS), default="held_out")

    build = subparsers.add_parser(
        "build-memory", help="Train source policies and populate shared memory"
    )
    build.add_argument("--memory-dir", type=Path, default=Path("artifacts/memory"))
    build.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASKS),
        default=["open", "upper_gap", "lower_gap"],
    )
    build.add_argument("--train-steps", type=int, default=5_000)
    build.add_argument("--random-episodes", type=int, default=8)
    build.add_argument("--policy-episodes", type=int, default=8)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument(
        "--reset",
        action="store_true",
        help="Remove known generated files in the selected memory directory first",
    )

    experiment = subparsers.add_parser(
        "experiment", help="Compare memory modes on a held-out task"
    )
    experiment.add_argument(
        "--memory-dir", type=Path, default=Path("artifacts/memory")
    )
    experiment.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/results")
    )
    experiment.add_argument("--task", choices=sorted(TASKS), default="held_out")
    experiment.add_argument(
        "--modes", nargs="+", choices=VALID_MODES, default=list(VALID_MODES)
    )
    experiment.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    experiment.add_argument("--train-steps", type=int, default=5_000)
    experiment.add_argument("--eval-interval", type=int, default=250)
    experiment.add_argument("--eval-episodes", type=int, default=10)
    experiment.add_argument("--retrieval-limit", type=int, default=100)
    experiment.add_argument("--max-prefill-transitions", type=int, default=10_000)

    controlled = subparsers.add_parser(
        "experiment-v2",
        help="Run the equal-budget v0.2 memory-transfer benchmark",
    )
    controlled.add_argument(
        "--memory-dir", type=Path, default=Path("artifacts/memory")
    )
    controlled.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        type=Path,
        default=Path("artifacts/v0.2-results"),
    )
    controlled.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASKS),
        default=list(DEFAULT_CONTROLLED_TASKS),
    )
    controlled.add_argument(
        "--modes",
        nargs="+",
        choices=CONTROLLED_MODES,
        default=list(CONTROLLED_MODES),
    )
    controlled.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(10))
    )
    controlled.add_argument("--train-steps", type=int, default=3_000)
    controlled.add_argument("--eval-interval", type=int, default=250)
    controlled.add_argument("--eval-episodes", type=int, default=25)
    controlled.add_argument("--episode-budget", type=int, default=12)
    controlled.add_argument("--replay-transitions", type=int, default=200)
    controlled.add_argument("--learning-starts", type=int, default=100)
    controlled.add_argument(
        "--save-models",
        action="store_true",
        help="Save local policy checkpoints (disabled by default)",
    )

    inspect = subparsers.add_parser("inspect-memory", help="Show memory contents")
    inspect.add_argument("--memory-dir", type=Path, default=Path("artifacts/memory"))

    demo = subparsers.add_parser(
        "demo", help="Run the complete pipeline with a small smoke-test budget"
    )
    demo.add_argument("--directory", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--seed", type=int, default=0)

    bridge_demo = subparsers.add_parser(
        "bridge-demo",
        help="Run cross-framework Multiverse -> MemoryBridge -> Gymnasium demo",
    )
    bridge_demo.add_argument(
        "--directory",
        type=Path,
        default=None,
        help="Optional directory to persist bridge records (uses tempdir if omitted)",
    )
    bridge_demo.add_argument(
        "--train-steps",
        type=int,
        default=200,
        help="Number of SAC training steps to run with transferred transitions",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        _check_environment(args.task)
    elif args.command == "build-memory":
        if args.reset:
            safe_reset_memory(args.memory_dir)
        store = build_source_memory(
            args.memory_dir,
            tasks=args.tasks,
            train_steps=args.train_steps,
            random_episodes=args.random_episodes,
            policy_episodes=args.policy_episodes,
            seed=args.seed,
        )
        print(f"Memory contains {store.count()} episodes: {store.task_counts()}")
    elif args.command == "experiment":
        if not (args.memory_dir / "memory.sqlite3").is_file():
            raise SystemExit(
                f"No memory database at {args.memory_dir}. Run 'uas build-memory' first."
            )
        run_experiment(
            args.memory_dir,
            args.output_dir,
            task_id=args.task,
            modes=args.modes,
            seeds=args.seeds,
            train_steps=args.train_steps,
            evaluation_interval=args.eval_interval,
            evaluation_episodes=args.eval_episodes,
            retrieval_limit=args.retrieval_limit,
            max_prefill_transitions=args.max_prefill_transitions,
        )
    elif args.command == "experiment-v2":
        if not (args.memory_dir / "memory.sqlite3").is_file():
            raise SystemExit(
                f"No memory database at {args.memory_dir}. Run 'uas build-memory' first."
            )
        run_controlled_experiment(
            args.memory_dir,
            args.output_dir,
            task_ids=args.tasks,
            modes=args.modes,
            seeds=args.seeds,
            train_steps=args.train_steps,
            evaluation_interval=args.eval_interval,
            evaluation_episodes=args.eval_episodes,
            episode_budget=args.episode_budget,
            replay_transition_budget=args.replay_transitions,
            learning_starts=args.learning_starts,
            save_models=args.save_models,
        )
    elif args.command == "inspect-memory":
        store = MemoryStore(args.memory_dir)
        successes = sum(record.success for record in store.records())
        print(f"Episodes: {store.count()}")
        print(f"Successful: {successes}")
        print(f"By task: {store.task_counts()}")
    elif args.command == "demo":
        memory_directory = args.directory / "memory"
        output_directory = args.directory / "results"
        safe_reset_memory(memory_directory)
        print("Running a smoke demo; its scores are not research evidence.")
        build_source_memory(
            memory_directory,
            tasks=("open", "upper_gap"),
            train_steps=300,
            random_episodes=1,
            policy_episodes=1,
            seed=args.seed,
        )
        run_experiment(
            memory_directory,
            output_directory,
            modes=("no_memory", "naive", "governed"),
            seeds=(args.seed,),
            train_steps=300,
            evaluation_interval=150,
            evaluation_episodes=2,
            retrieval_limit=4,
            max_prefill_transitions=500,
        )
    elif args.command == "bridge-demo":
        from uas.bridge import run_bridge_interop_demo

        run_bridge_interop_demo(args.directory, steps_to_train=args.train_steps)


def _check_environment(task_id: str) -> None:
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env as gym_check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check_env

    wrapped_environment = gym.make(
        GYM_ENV_ID, task_id=task_id, render_mode="rgb_array"
    )
    environment = wrapped_environment.unwrapped
    gym_check_env(environment, skip_render_check=False)
    sb3_check_env(environment, warn=True, skip_render_check=False)
    observation, _ = environment.reset(seed=0)
    for _ in range(10):
        observation, _, terminated, truncated, _ = environment.step(
            environment.action_space.sample()
        )
        if terminated or truncated:
            observation, _ = environment.reset()
    frame = environment.render()
    wrapped_environment.close()
    print(
        "Environment checks passed: "
        f"observation_shape={observation.shape}, frame_shape={frame.shape}"
    )


if __name__ == "__main__":
    main()
