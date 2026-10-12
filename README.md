# Universe Agent System

The Universe Agent System (UAS) is a research framework in which learning agents contribute experiences to shared memory. A retrieval and validation layer selects compatible knowledge that may help an agent adapt to an unfamiliar but related task.

> **Project status:** Early research prototype. UAS v0.2 adds a controlled, equal-budget transfer benchmark with harder held-out tasks, five comparison conditions, ten-seed defaults, confidence intervals, and negative-transfer reporting. It does not yet support robotics or unrelated environment families.

![Universe Agent System concept diagram](Screenshot_2025-02-08_180627.png)

## How it works

1. Multiple agents explore distinct virtual environments and learn through interaction.
2. Each agent sends useful experiences to a centralized memory repository.
3. A scenario-matching engine retrieves relevant knowledge for new tasks.
4. Agents and the central AI continuously exchange feedback and improve their behavior.
5. Knowledge developed in simulation can eventually be transferred to real-world systems.

## Proposed components

- **Multi-agent framework** — assigns goals, runs agents in parallel environments, and supports knowledge transfer between agents.
- **Centralized memory** — stores experiences using embeddings, knowledge graphs, and memory networks.
- **Scenario matching** — finds relevant prior experiences with similarity search and predictive models.
- **Continuous learning loop** — updates agents and shared memory as new experiences arrive.
- **Environment generation** — creates and evolves training environments from descriptive inputs.
- **Movement learning** — analyzes and reproduces human or machine movement while accounting for physical constraints.

## Potential applications

- Adaptive game characters and dynamic storytelling
- Robots that transfer knowledge across tasks and environments
- Autonomous-system simulation and edge-case training
- Interactive creative tools and personalized media

## Proposed technology stack

| Area | Candidate tools |
| --- | --- |
| Reinforcement learning | Stable-Baselines3, Ray RLlib |
| Simulation | Gymnasium/OpenAI Gym, Unity ML-Agents, PyBullet |
| Knowledge storage | SQLite, Neo4j, Elasticsearch |
| Memory representation | Sentence Transformers, NumPy, knowledge graphs |
| Movement analysis | MediaPipe, OpenPose |
| Visualization | TensorBoard, Matplotlib |

## Roadmap

- Preserve the published v0.1 result as the first pipeline baseline.
- Run the controlled v0.2 benchmark with equal episode and replay-transition budgets.
- Publish v0.2 results, including uncertainty and negative-transfer cases.
- Add procedural skill checkpoints after the episode-memory baseline is understood.
- Reproduce a useful result in a robot-manipulation simulator.
- Add evaluation metrics for transfer quality, adaptability, and memory relevance.
- Explore procedural environment generation and simulation-to-reality transfer.

## Run UAS

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
uas check
uas demo
```

`uas demo` is a fast pipeline check, not a research result. The full experiment first builds source-task memory and then compares all retrieval conditions:

```powershell
uas build-memory --reset
uas experiment-v2
```

The v0.2 command runs two harder tasks, five conditions, and ten seeds by default. It uses equal replay budgets and does not save policy checkpoints unless `--save-models` is provided. See the [v0.2 controlled experiment guide](docs/EXPERIMENT_V0_2.md) for the protocol, commands, metrics, outputs, and limitations. The original `uas experiment` command remains available for reproducing v0.1.

Published benchmark results:
- [v0.1 held-out benchmark](results/v0.1/README.md) — initial pipeline verification across 4 conditions and 3 seeds.
- [v0.2 controlled benchmark](results/v0.2/README.md) — 100-run, 10-seed equalized-budget experiment across 5 conditions and harder tasks.

## Project documents

- [Research landscape and supporting evidence](docs/RESEARCH_LANDSCAPE.md) — related projects, papers, benchmarks, and tools organized around the UAS concept.
- [How existing research fits UAS](docs/RESEARCH_TO_UAS_MAP.md) — maps prior work to each proposed UAS component, separates established evidence from open hypotheses, and defines a focused first experiment.
- [UAS v0.1 governed-memory experiment](docs/EXPERIMENT_V0_1.md) — installation, architecture, comparison conditions, commands, outputs, and current limitations.
- [UAS v0.2 controlled-memory experiment](docs/EXPERIMENT_V0_2.md) — equal-budget controls, harder held-out tasks, metrics, commands, and publication criteria.
- [Project description](Project%20Description%20Universe%20Agent%20System%20%28UAS%29%2016e3df2ced6d8028aaceed7a9f4923e9.md)
- [Initial case study](Universe%20Agent%20System%2016e3df2ced6d80e49c50f6e40b90df82.md)
- [Expanded case study](Universe%20Agent%20System%201963df2ced6d80cca9a3de1f01471efe.md)

## Contributing

Ideas, architecture proposals, experiments, and prototype implementations are welcome. Open an issue to discuss a substantial change before beginning work.

## License

Licensed under the [Apache License 2.0](LICENSE). You may use, modify, and distribute the project under its terms.
