"""SQLite metadata and compressed NumPy trajectory storage."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Iterable

import numpy as np

from uas.memory.models import EpisodeRecord, RetrievedEpisode, TaskQuery, Trajectory


class MemoryStore:
    """Persist, inspect, and retrieve schema-compatible episodes."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.trajectory_directory = self.directory / "trajectories"
        self.model_directory = self.directory / "models"
        self.database_path = self.directory / "memory.sqlite3"
        self.trajectory_directory.mkdir(parents=True, exist_ok=True)
        self.model_directory.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    environment_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    task_family TEXT NOT NULL,
                    task_description TEXT NOT NULL,
                    observation_schema TEXT NOT NULL,
                    action_schema TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    policy_reference TEXT,
                    seed INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    total_reward REAL NOT NULL,
                    normalized_score REAL NOT NULL,
                    steps INTEGER NOT NULL,
                    termination_reason TEXT NOT NULL,
                    collision_count INTEGER NOT NULL,
                    task_vector TEXT NOT NULL,
                    trajectory_file TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS episodes_compatibility
                ON episodes(task_family, observation_schema, action_schema)
                """
            )

    def add_episode(self, record: EpisodeRecord, trajectory: Trajectory) -> None:
        trajectory.validate()
        final_path = self.trajectory_directory / record.trajectory_file
        if final_path.exists():
            raise ValueError(f"Trajectory already exists: {final_path}")

        handle = tempfile.NamedTemporaryFile(
            dir=self.trajectory_directory, suffix=".npz", delete=False
        )
        temporary_path = Path(handle.name)
        handle.close()
        try:
            np.savez_compressed(
                temporary_path,
                observations=np.asarray(trajectory.observations, dtype=np.float32),
                actions=np.asarray(trajectory.actions, dtype=np.float32),
                rewards=np.asarray(trajectory.rewards, dtype=np.float32),
                terminated=np.asarray(trajectory.terminated, dtype=np.bool_),
                truncated=np.asarray(trajectory.truncated, dtype=np.bool_),
            )
            os.replace(temporary_path, final_path)
            values = {
                field.name: getattr(record, field.name) for field in fields(record)
            }
            values["success"] = int(record.success)
            values["task_vector"] = json.dumps(record.task_vector)
            columns = ", ".join(values)
            placeholders = ", ".join(f":{name}" for name in values)
            with self._connect() as connection:
                connection.execute(
                    f"INSERT INTO episodes ({columns}) VALUES ({placeholders})",
                    values,
                )
        except Exception:
            temporary_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM episodes").fetchone()
        return int(row["count"])

    def records(self) -> list[EpisodeRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM episodes ORDER BY created_at, episode_id"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def retrieve(
        self,
        query: TaskQuery,
        *,
        mode: str,
        limit: int = 20,
        minimum_score: float = 0.5,
    ) -> list[RetrievedEpisode]:
        if mode == "no_memory":
            return []
        if mode not in {"naive", "similarity", "governed"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        compatible = [
            record
            for record in self.records()
            if record.task_family == query.task_family
            and record.observation_schema == query.observation_schema
            and record.action_schema == query.action_schema
            and record.task_id != query.task_id
        ]
        scored = [self._score(record, query) for record in compatible]

        if mode == "naive":
            scored.sort(key=lambda item: (item.record.created_at, item.record.episode_id))
        elif mode == "similarity":
            scored.sort(
                key=lambda item: (item.similarity, item.record.normalized_score),
                reverse=True,
            )
        else:
            scored = [
                item
                for item in scored
                if item.record.success
                and item.record.normalized_score >= minimum_score
                and item.record.schema_version == 1
            ]
            scored.sort(key=lambda item: item.retrieval_score, reverse=True)
        return scored[: max(0, int(limit))]

    def load_trajectory(self, record: EpisodeRecord) -> Trajectory:
        path = self.trajectory_directory / record.trajectory_file
        if not path.is_file():
            raise FileNotFoundError(f"Missing trajectory for {record.episode_id}: {path}")
        with np.load(path, allow_pickle=False) as data:
            trajectory = Trajectory(
                observations=data["observations"].astype(np.float32),
                actions=data["actions"].astype(np.float32),
                rewards=data["rewards"].astype(np.float32),
                terminated=data["terminated"].astype(np.bool_),
                truncated=data["truncated"].astype(np.bool_),
            )
        trajectory.validate()
        return trajectory

    def task_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT task_id, COUNT(*) AS count FROM episodes GROUP BY task_id"
            ).fetchall()
        return {str(row["task_id"]): int(row["count"]) for row in rows}

    def _score(self, record: EpisodeRecord, query: TaskQuery) -> RetrievedEpisode:
        first = np.asarray(record.task_vector, dtype=np.float32)
        second = np.asarray(query.task_vector, dtype=np.float32)
        if first.shape != second.shape:
            similarity = 0.0
        else:
            distance = float(np.linalg.norm(first - second) / np.sqrt(first.size))
            similarity = 1.0 / (1.0 + distance)
        retrieval_score = 0.70 * similarity + 0.30 * record.normalized_score
        return RetrievedEpisode(
            record=record,
            similarity=similarity,
            retrieval_score=retrieval_score,
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> EpisodeRecord:
        return EpisodeRecord(
            episode_id=str(row["episode_id"]),
            schema_version=int(row["schema_version"]),
            environment_id=str(row["environment_id"]),
            task_id=str(row["task_id"]),
            task_family=str(row["task_family"]),
            task_description=str(row["task_description"]),
            observation_schema=str(row["observation_schema"]),
            action_schema=str(row["action_schema"]),
            agent_id=str(row["agent_id"]),
            policy_version=str(row["policy_version"]),
            policy_reference=(
                None if row["policy_reference"] is None else str(row["policy_reference"])
            ),
            seed=int(row["seed"]),
            success=bool(row["success"]),
            total_reward=float(row["total_reward"]),
            normalized_score=float(row["normalized_score"]),
            steps=int(row["steps"]),
            termination_reason=str(row["termination_reason"]),
            collision_count=int(row["collision_count"]),
            task_vector=tuple(float(value) for value in json.loads(row["task_vector"])),
            trajectory_file=str(row["trajectory_file"]),
            created_at=str(row["created_at"]),
        )

    def load_many(self, episodes: Iterable[RetrievedEpisode]) -> list[Trajectory]:
        return [self.load_trajectory(episode.record) for episode in episodes]
