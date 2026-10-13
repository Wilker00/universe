"""Lightweight, 3-line importable memory bridge for AI training frameworks."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import numpy as np

from uas.bridge.models import CompatibilityReport, Episode, SearchQuery


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

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that guarantees connection is closed on Windows."""
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize_database(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    embodiment TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    observation_schema TEXT NOT NULL,
                    action_schema TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    task_description TEXT NOT NULL DEFAULT '',
                    success INTEGER NOT NULL,
                    total_reward REAL NOT NULL,
                    steps INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    trajectory_file TEXT NOT NULL
                )
                """
            )
            # Migration check for existing databases
            try:
                conn.execute(
                    "ALTER TABLE episodes ADD COLUMN task_description TEXT NOT NULL DEFAULT ''"
                )
            except sqlite3.OperationalError:
                pass

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_tags (
                    episode_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (episode_id, tag),
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task ON episodes(task)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embodiment ON episodes(embodiment)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_framework ON episodes(framework)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_success ON episodes(success)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tag ON episode_tags(tag)")

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
        tags = episode.metadata.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes (
                    episode_id, task, embodiment, framework, observation_schema,
                    action_schema, action_type, task_description, success, total_reward, steps,
                    created_at, metadata_json, trajectory_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    episode.task,
                    episode.embodiment,
                    episode.framework,
                    episode.observation_schema,
                    episode.action_schema,
                    episode.action_type,
                    episode.task_description,
                    int(episode.success),
                    float(episode.total_reward),
                    episode.steps,
                    episode.created_at,
                    metadata_json,
                    trajectory_file,
                ),
            )
            for tag in tags:
                conn.execute(
                    "INSERT OR IGNORE INTO episode_tags (episode_id, tag) VALUES (?, ?)",
                    (episode.episode_id, str(tag)),
                )
        return episode.episode_id

    def search(
        self,
        task: str | None = None,
        embodiment: str | None = None,
        framework: str | None = None,
        observation_schema: str | None = None,
        action_schema: str | None = None,
        action_type: str | None = None,
        success_only: bool = False,
        min_reward: float | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[Episode]:
        """Search and load matching episodes from the memory bridge with correct SQL tag filtering."""
        where_clauses = ["1=1"]
        params: list[Any] = []
        join_clause = ""

        if tags:
            tag_placeholders = ",".join("?" for _ in tags)
            join_clause = f"JOIN episode_tags t ON e.episode_id = t.episode_id"
            where_clauses.append(f"t.tag IN ({tag_placeholders})")
            params.extend(tags)

        if task is not None:
            where_clauses.append("e.task = ?")
            params.append(task)
        if embodiment is not None:
            where_clauses.append("e.embodiment = ?")
            params.append(embodiment)
        if framework is not None:
            where_clauses.append("e.framework = ?")
            params.append(framework)
        if observation_schema is not None:
            where_clauses.append("e.observation_schema = ?")
            params.append(observation_schema)
        if action_schema is not None:
            where_clauses.append("e.action_schema = ?")
            params.append(action_schema)
        if action_type is not None:
            where_clauses.append("e.action_type = ?")
            params.append(action_type)
        if success_only:
            where_clauses.append("e.success = 1")
        if min_reward is not None:
            where_clauses.append("e.total_reward >= ?")
            params.append(float(min_reward))

        sql = f"""
            SELECT DISTINCT e.* FROM episodes e
            {join_clause}
            WHERE {' AND '.join(where_clauses)}
            ORDER BY e.total_reward DESC
            LIMIT ?
        """
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        results: list[Episode] = []
        for row in rows:
            traj_path = self.trajectory_directory / row["trajectory_file"]
            if not traj_path.is_file():
                continue

            with np.load(traj_path) as data:
                task_desc = (
                    str(row["task_description"])
                    if "task_description" in row.keys() and row["task_description"]
                    else str(row["task"])
                )
                episode = Episode(
                    episode_id=row["episode_id"],
                    task=row["task"],
                    embodiment=row["embodiment"],
                    framework=row["framework"],
                    observation_schema=row["observation_schema"],
                    action_schema=row["action_schema"],
                    action_type=row["action_type"],
                    task_description=task_desc,
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

        task_desc = (
            str(row["task_description"])
            if "task_description" in row.keys() and row["task_description"]
            else str(row["task"])
        )
        with np.load(traj_path) as data:
            return Episode(
                episode_id=row["episode_id"],
                task=row["task"],
                embodiment=row["embodiment"],
                framework=row["framework"],
                observation_schema=row["observation_schema"],
                action_schema=row["action_schema"],
                action_type=row["action_type"],
                task_description=task_desc,
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

    def search_semantic(
        self,
        query_text: str,
        limit: int = 10,
        min_similarity: float = 0.05,
    ) -> list[tuple[Episode, float]]:
        """Search episodes using natural language text similarity over tasks, descriptions, and tags."""
        all_episodes = self.search(limit=1000)
        scored: list[tuple[Episode, float]] = []

        query_tokens = set(re.findall(r"\w+", query_text.lower()))
        if not query_tokens:
            return []

        for ep in all_episodes:
            tags = ep.metadata.get("tags", [])
            tag_str = " ".join(tags) if isinstance(tags, list) else str(tags)
            doc_text = f"{ep.task_description} {ep.task} {tag_str}".lower()
            doc_tokens = set(re.findall(r"\w+", doc_text))

            overlap = len(query_tokens & doc_tokens)
            union = len(query_tokens | doc_tokens)
            jaccard = float(overlap / union) if union > 0 else 0.0

            # Substring / exact phrase match bonus
            substring_bonus = 0.4 if query_text.lower() in doc_text else 0.0
            similarity = min(1.0, jaccard + substring_bonus)

            if similarity >= min_similarity:
                scored.append((ep, similarity))

        scored.sort(key=lambda item: (item[1], item[0].total_reward), reverse=True)
        return scored[: max(0, int(limit))]

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
