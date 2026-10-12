"""Lightweight, 3-line importable memory bridge for AI training frameworks."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from uas.bridge.models import Episode, SearchQuery


def _safe_replace(
    source: Path, target: Path, *, retries: int = 20, delay: float = 0.05
) -> None:
    for attempt in range(retries):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))


class MemoryBridge:
    """A framework-neutral client for recording and querying experience.

    Usage:
        bridge = MemoryBridge("path/to/memory")
        bridge.record(episode)
        episodes = bridge.search(task="door_opening", embodiment="franka_panda")
    """

    def __init__(self, directory: str | Path = "artifacts/memory_bridge") -> None:
        self.directory = Path(directory)
        self.trajectory_directory = self.directory / "trajectories"
        self.database_path = self.directory / "bridge.sqlite3"
        self.trajectory_directory.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize_database(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    embodiment TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    total_reward REAL NOT NULL,
                    steps INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    trajectory_file TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task ON episodes(task)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embodiment ON episodes(embodiment)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_framework ON episodes(framework)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_success ON episodes(success)"
            )

    def record(self, episode: Episode) -> str:
        """Persist an episode from any framework into the shared memory bridge."""
        trajectory_file = f"{episode.episode_id}.npz"
        final_path = self.trajectory_directory / trajectory_file
        temporary_file = final_path.with_name(f".{final_path.stem}.tmp.npz")

        try:
            np.savez_compressed(
                temporary_file,
                observations=episode.observations,
                actions=episode.actions,
                rewards=episode.rewards,
                terminated=episode.terminated,
                truncated=episode.truncated,
            )
            _safe_replace(temporary_file, final_path)
        finally:
            temporary_file.unlink(missing_ok=True)

        metadata_json = json.dumps(episode.metadata)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes (
                    episode_id, task, embodiment, framework, success,
                    total_reward, steps, created_at, metadata_json, trajectory_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    episode.task,
                    episode.embodiment,
                    episode.framework,
                    int(episode.success),
                    float(episode.total_reward),
                    episode.steps,
                    episode.created_at,
                    metadata_json,
                    trajectory_file,
                ),
            )
        return episode.episode_id

    def search(
        self,
        task: str | None = None,
        embodiment: str | None = None,
        framework: str | None = None,
        success_only: bool = False,
        min_reward: float | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[Episode]:
        """Search and load matching episodes from the memory bridge."""
        query_parts = ["1=1"]
        params: list[Any] = []

        if task is not None:
            query_parts.append("task = ?")
            params.append(task)
        if embodiment is not None:
            query_parts.append("embodiment = ?")
            params.append(embodiment)
        if framework is not None:
            query_parts.append("framework = ?")
            params.append(framework)
        if success_only:
            query_parts.append("success = 1")
        if min_reward is not None:
            query_parts.append("total_reward >= ?")
            params.append(float(min_reward))

        sql = f"""
            SELECT * FROM episodes
            WHERE {' AND '.join(query_parts)}
            ORDER BY total_reward DESC
            LIMIT ?
        """
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        results: list[Episode] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if tags:
                record_tags = metadata.get("tags", [])
                if not any(tag in record_tags for tag in tags):
                    continue

            traj_path = self.trajectory_directory / row["trajectory_file"]
            if not traj_path.is_file():
                continue

            with np.load(traj_path) as data:
                episode = Episode(
                    episode_id=row["episode_id"],
                    task=row["task"],
                    embodiment=row["embodiment"],
                    framework=row["framework"],
                    success=bool(row["success"]),
                    total_reward=float(row["total_reward"]),
                    observations=data["observations"],
                    actions=data["actions"],
                    rewards=data["rewards"],
                    terminated=data["terminated"],
                    truncated=data["truncated"],
                    metadata=metadata,
                    created_at=row["created_at"],
                )
                results.append(episode)
        return results

    def get(self, episode_id: str) -> Episode | None:
        """Fetch a single episode by its unique ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
        if row is None:
            return None

        traj_path = self.trajectory_directory / row["trajectory_file"]
        if not traj_path.is_file():
            return None

        with np.load(traj_path) as data:
            return Episode(
                episode_id=row["episode_id"],
                task=row["task"],
                embodiment=row["embodiment"],
                framework=row["framework"],
                success=bool(row["success"]),
                total_reward=float(row["total_reward"]),
                observations=data["observations"],
                actions=data["actions"],
                rewards=data["rewards"],
                terminated=data["terminated"],
                truncated=data["truncated"],
                metadata=json.loads(row["metadata_json"]),
                created_at=row["created_at"],
            )

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def tasks(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT task FROM episodes").fetchall()
            return [r[0] for r in rows]

    def embodiments(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT embodiment FROM episodes"
            ).fetchall()
            return [r[0] for r in rows]
