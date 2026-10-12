# UAS v0.1: Governed Memory Experiment

UAS v0.1 is a runnable research prototype for one narrow question:

> Does a governed memory of compatible, successful episodes improve adaptation to a held-out continuous-control task compared with learning from scratch or sharing experience indiscriminately?

It is not yet a multi-simulator intelligence or a robotics system. The prototype deliberately keeps the domain small so the effect of memory can be measured before the project adds more complexity.

## Continuous non-grid world

The first environment is a continuous two-dimensional room. Its actions are continuous movement vectors rather than grid-cell commands. Four layouts use the same observation and action schemas:

| Task | Purpose |
| --- | --- |
| `open` | Learn direct goal navigation. |
| `upper_gap` | Learn to navigate above a vertical obstacle. |
| `lower_gap` | Learn to navigate below a vertical obstacle. |
| `held_out` | Evaluate adaptation to a shifted, unfamiliar obstacle layout. |

An observation contains the agent position, goal position, goal delta, obstacle bounds, obstacle-presence flag, and normalized episode time. An action is a two-dimensional movement vector in `[-1, 1]`.

## Memory contents

SQLite stores searchable episode metadata. Compressed NumPy files store full trajectories. Each record includes:

- environment, task, and task-family identifiers;
- observation and action schema identifiers;
- task features used for similarity scoring;
- agent and policy versions;
- success, reward, steps, collisions, and termination reason;
- provenance, creation time, and trajectory reference; and
- an optional policy-checkpoint reference.

Schema and task-family checks are mandatory because an episode cannot be replayed safely into a model with incompatible observations or actions.

## Compared conditions

| Mode | Behavior |
| --- | --- |
| `no_memory` | Start with an empty SAC replay buffer. |
| `naive` | Replay every schema-compatible source episode, including failures. |
| `similarity` | Rank schema-compatible episodes by task-feature similarity without requiring success. |
| `governed` | Require compatibility and successful outcomes, then rank using similarity and normalized outcome quality. |

Retrieved transitions prefill a new Soft Actor-Critic agent's replay buffer. Every condition then learns online in the held-out environment using the same model size and training budget.

## Installation

Python 3.10 or newer is required. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Validate the environment and its Stable-Baselines3 compatibility:

```powershell
uas check
```

## Build memory

Train policies on the three source tasks and collect both random and trained-policy episodes:

```powershell
uas build-memory --reset
```

The defaults use 5,000 training steps for each source task and store eight random plus eight trained-policy episodes per task. Inspect the result with:

```powershell
uas inspect-memory
```

## Run the comparison

```powershell
uas experiment
```

The default experiment evaluates all four modes over three seeds. Generated artifacts are placed under `artifacts/` and include:

- `learning_curves.csv` with every evaluation;
- `summary.json` with final results, best success, and steps to an 80% success threshold;
- `retrieval_manifest.json` identifying every selected source episode and its scores;
- `run_config.json` with the experimental settings and package versions;
- `success_rate.png` with mean learning curves; and
- trained model checkpoints.

For a fast pipeline check that is not scientifically meaningful:

```powershell
uas demo
```

## Interpretation

A useful result requires more than one successful run. The governed condition should reach a predefined success level with fewer environment interactions than both `no_memory` and `naive` across multiple seeds. Failed transfer, retrieval cost, collision rate, and memory size should also be reported.

The smoke demo only verifies that the pipeline runs. Its small training budget must not be presented as research evidence.

## Current limitations

- The task descriptor is a hand-designed numeric vector, not a learned embedding.
- Stored experience comes from one environment family with identical schemas.
- Replay-buffer transfer is only one possible form of memory reuse.
- The current governance policy uses simple compatibility and outcome rules.
- Hyperparameters have not yet been tuned or benchmarked rigorously.
- No simulation-to-real or physical-robot claim is tested by this prototype.

These constraints make the first result easier to interpret. Later versions can add learned task representations, distilled skills, manipulation environments, richer validation, and robot-specific adapters if v0.1 shows a measurable benefit.
