"""SQLite persistence for compact ATA alert records."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Mapping


class AlertStore:
    """Thread-safe SQLite store for alerts; raw flow/packet data is not stored."""

    def __init__(self, database: str | Path = "data/ata_alerts.sqlite3"):
        self.database = str(database)
        if self.database != ":memory:":
            Path(self.database).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.database,
            check_same_thread=False,
            timeout=10.0,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                window_index INTEGER NOT NULL,
                event_timestamp REAL,
                anomaly_score REAL,
                predicted_attack_type TEXT,
                class_probability REAL,
                top_features_json TEXT NOT NULL,
                drift_metadata_json TEXT,
                alert_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS alerts_event_time_idx "
            "ON alerts(event_timestamp)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS alerts_attack_type_idx "
            "ON alerts(predicted_attack_type)"
        )
        self._connection.commit()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, separators=(",", ":"), allow_nan=False)

    def add(self, alert: Mapping[str, Any]) -> dict[str, Any]:
        """Insert an alert or reject a duplicate identifier."""
        record = dict(alert)
        required = {"alert_id", "window_index", "created_at"}
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError("Alert is missing fields: " + ", ".join(missing))
        if not isinstance(record["alert_id"], str) or not record["alert_id"]:
            raise ValueError("alert_id must be a non-empty string.")
        features = record.get("top_features", [])
        if not isinstance(features, list):
            raise ValueError("top_features must be a list.")
        values = (
            record["alert_id"],
            int(record["window_index"]),
            record.get("event_timestamp"),
            record.get("layer1_score"),
            record.get("predicted_class"),
            record.get("class_probability"),
            self._json(features),
            (
                None
                if record.get("drift_metadata") is None
                else self._json(record["drift_metadata"])
            ),
            self._json(record),
            record["created_at"],
        )
        with self._lock:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO alerts (
                        alert_id, window_index, event_timestamp, anomaly_score,
                        predicted_attack_type, class_probability, top_features_json,
                        drift_metadata_json, alert_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
        return record

    def get(self, alert_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT alert_json FROM alerts WHERE alert_id = ?",
                (alert_id,),
            ).fetchone()
        return None if row is None else json.loads(row["alert_json"])

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        attack_type: str | None = None,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000.")
        if offset < 0:
            raise ValueError("offset cannot be negative.")
        clauses = []
        parameters: list[Any] = []
        if attack_type is not None:
            clauses.append("predicted_attack_type = ?")
            parameters.append(attack_type)
        if since is not None:
            clauses.append("event_timestamp >= ?")
            parameters.append(float(since))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = (
            "SELECT alert_json FROM alerts"
            + where
            + " ORDER BY window_index DESC, alert_id DESC LIMIT ? OFFSET ?"
        )
        parameters.extend((limit, offset))
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [json.loads(row["alert_json"]) for row in rows]

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) FROM alerts").fetchone()
        return int(row[0])

    def clear(self) -> None:
        with self._lock:
            with self._connection:
                self._connection.execute("DELETE FROM alerts")

    def close(self) -> None:
        with self._lock:
            self._connection.close()
