from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import TracebackType

from pydantic import JsonValue

from adaptiveedge.config.schema import BenchmarkConfiguration
from adaptiveedge.core.models import BenchmarkResult, HardwareSnapshot, utcnow

SCHEMA_VERSION = 2
SCHEMA = """
CREATE TABLE experiments (
    id TEXT PRIMARY KEY, config_hash TEXT NOT NULL, name TEXT NOT NULL,
    created_at TEXT NOT NULL, configuration TEXT NOT NULL, manifest TEXT NOT NULL,
    split TEXT NOT NULL DEFAULT 'calibration'
);
CREATE TABLE models (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE tasks (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE hardware (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE runs (
    id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id),
    model_id TEXT NOT NULL REFERENCES models(id), task_id TEXT NOT NULL REFERENCES tasks(id),
    hardware_id TEXT NOT NULL REFERENCES hardware(id), payload TEXT NOT NULL,
    split TEXT NOT NULL DEFAULT 'calibration'
);
CREATE TABLE inference_measurements (
    run_id TEXT PRIMARY KEY REFERENCES runs(id), payload TEXT NOT NULL
);
CREATE TABLE quality_metrics (
    run_id TEXT NOT NULL REFERENCES runs(id), name TEXT NOT NULL, value REAL NOT NULL,
    PRIMARY KEY (run_id, name)
);
CREATE TABLE failures (run_id TEXT PRIMARY KEY REFERENCES runs(id), payload TEXT NOT NULL);
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id), kind TEXT NOT NULL,
    timestamp TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE INDEX runs_experiment ON runs(experiment_id);
"""


class Repository:
    def __init__(self, path: Path | str = "adaptiveedge.db") -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path, timeout=30)
        try:
            self.connection.execute("PRAGMA foreign_keys = ON")
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, SCHEMA_VERSION):
                raise ValueError(
                    f"unsupported database schema {version}; expected {SCHEMA_VERSION}"
                )
            if version == 0:
                self.connection.executescript("BEGIN IMMEDIATE;\n" + SCHEMA)
                self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                self.connection.commit()
            if version == 1:
                with self.connection:
                    self.connection.execute("BEGIN IMMEDIATE")
                    for table in ("experiments", "runs"):
                        self.connection.execute(
                            f"ALTER TABLE {table} ADD COLUMN split "
                            "TEXT NOT NULL DEFAULT 'calibration'"
                        )
                    self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._install_immutability()
        except Exception:
            self.connection.close()
            raise

    def _install_immutability(self) -> None:
        with self.connection:
            for table in (
                "experiments",
                "models",
                "tasks",
                "hardware",
                "runs",
                "inference_measurements",
                "quality_metrics",
                "failures",
                "events",
            ):
                for operation in ("UPDATE", "DELETE"):
                    self.connection.execute(
                        f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation} "
                        f"BEFORE {operation} ON {table} BEGIN "
                        "SELECT RAISE(ABORT, 'raw records are immutable'); END"
                    )

    def __enter__(self) -> Repository:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def create_experiment(
        self,
        identifier: str,
        configuration: BenchmarkConfiguration,
        hardware: HardwareSnapshot,
        software: dict[str, str],
        extra: dict[str, JsonValue] | None = None,
    ) -> None:
        manifest: dict[str, JsonValue] = {
            "config_hash": configuration.identity,
            "split": configuration.experiment.split,
            "hardware": hardware.model_dump(mode="json"),
            "software": dict(software),
            "extra": extra or {},
            "privacy": "Raw prompts and outputs are stored locally; exports include them",
        }
        with self.connection:
            self.connection.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    identifier,
                    configuration.identity,
                    configuration.experiment.name,
                    utcnow().isoformat(),
                    configuration.model_dump_json(),
                    json.dumps(manifest),
                    configuration.experiment.split,
                ),
            )

    def save_result(self, result: BenchmarkResult) -> None:
        with self.connection:
            experiment = self.connection.execute(
                "SELECT split FROM experiments WHERE id = ?", (result.experiment_id,)
            ).fetchone()
            if experiment is None or experiment[0] != result.split:
                raise ValueError("observation split must match its existing experiment")
            for table, record in (
                ("models", result.configuration),
                ("tasks", result.task),
                ("hardware", result.hardware),
            ):
                self.connection.execute(
                    f"INSERT OR IGNORE INTO {table} VALUES (?, ?)",
                    (record.identity, record.model_dump_json()),
                )
            self.connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    result.run_id,
                    result.experiment_id,
                    result.configuration.identity,
                    result.task.identity,
                    result.hardware.identity,
                    result.model_dump_json(),
                    result.split,
                ),
            )
            self.connection.execute(
                "INSERT INTO inference_measurements VALUES (?, ?)",
                (result.run_id, result.inference.model_dump_json()),
            )
            self.connection.executemany(
                "INSERT INTO quality_metrics VALUES (?, ?, ?)",
                [(result.run_id, name, value) for name, value in result.quality.items()],
            )
            if result.inference.error is not None:
                self.connection.execute(
                    "INSERT INTO failures VALUES (?, ?)",
                    (result.run_id, result.inference.error.model_dump_json()),
                )

    def event(self, experiment_id: str, kind: str, payload: dict[str, JsonValue]) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO events (experiment_id, kind, timestamp, payload) VALUES (?, ?, ?, ?)",
                (experiment_id, kind, utcnow().isoformat(), json.dumps(payload)),
            )

    def results(self, experiment_id: str | None = None) -> tuple[BenchmarkResult, ...]:
        query = "SELECT payload FROM runs"
        arguments: tuple[str, ...] = ()
        if experiment_id is not None:
            query += " WHERE experiment_id = ?"
            arguments = (experiment_id,)
        query += " ORDER BY rowid"
        return tuple(
            BenchmarkResult.model_validate_json(row[0])
            for row in self.connection.execute(query, arguments)
        )

    def experiments(self) -> list[dict[str, str]]:
        rows = self.connection.execute(
            "SELECT id, name, config_hash, created_at FROM experiments ORDER BY rowid"
        )
        return [
            dict(zip(("id", "name", "config_hash", "created_at"), row, strict=True)) for row in rows
        ]

    def configuration(self, experiment_id: str) -> BenchmarkConfiguration:
        row = self.connection.execute(
            "SELECT configuration FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown experiment: {experiment_id}")
        return BenchmarkConfiguration.model_validate_json(row[0])

    def manifest(self, experiment_id: str) -> dict[str, JsonValue]:
        row = self.connection.execute(
            "SELECT manifest FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown experiment: {experiment_id}")
        value: dict[str, JsonValue] = json.loads(row[0])
        value["events"] = [
            {"kind": e[0], "timestamp": e[1], "payload": json.loads(e[2])}
            for e in self.connection.execute(
                "SELECT kind, timestamp, payload FROM events WHERE experiment_id = ? ORDER BY id",
                (experiment_id,),
            )
        ]
        return value
