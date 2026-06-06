from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional


def resolve_database_path(project_root: Path) -> Path:
    project_root = Path(project_root).resolve()
    configured_path = str(os.environ.get("APP_DB_PATH") or "").strip()
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = project_root / path
        return path.resolve()
    return (project_root / "data" / "app_data.sqlite3").resolve()


def connect_database(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def ensure_column(connection: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
    columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def initialize_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect_database(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS google_form_responses (
                form_hash TEXT PRIMARY KEY,
                form_dir_name TEXT NOT NULL,
                form_title TEXT NOT NULL DEFAULT '',
                form_id TEXT NOT NULL DEFAULT '',
                response_id TEXT NOT NULL DEFAULT '',
                submitted_at TEXT NOT NULL,
                received_at TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                age TEXT NOT NULL DEFAULT '',
                date_prefix TEXT NOT NULL,
                fields_json TEXT NOT NULL,
                phq8_json TEXT NOT NULL,
                remote_addr TEXT,
                user_agent TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_google_form_responses_received_at
                ON google_form_responses(received_at DESC, form_dir_name DESC);

            CREATE TABLE IF NOT EXISTS realtime_sessions (
                session_hash TEXT PRIMARY KEY,
                session_dir_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                remote_addr TEXT,
                user_agent TEXT,
                metadata_json TEXT,
                transcript_json TEXT,
                transcript_text TEXT,
                saved_file_names_json TEXT NOT NULL DEFAULT '[]',
                saved_file_paths_json TEXT NOT NULL DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_realtime_sessions_updated_at
                ON realtime_sessions(updated_at DESC);

            CREATE TABLE IF NOT EXISTS realtime_session_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_hash TEXT NOT NULL,
                run_id TEXT NOT NULL,
                session_dir_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                uploaded_at TEXT NOT NULL,
                remote_addr TEXT,
                user_agent TEXT,
                metadata_json TEXT,
                transcript_json TEXT,
                transcript_text TEXT,
                saved_file_names_json TEXT NOT NULL DEFAULT '[]',
                saved_file_paths_json TEXT NOT NULL DEFAULT '[]',
                memory_status TEXT NOT NULL DEFAULT '',
                memory_operation_id TEXT,
                memory_error TEXT,
                memory_queued_at TEXT,
                UNIQUE(session_hash, run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_realtime_session_runs_hash_started
                ON realtime_session_runs(session_hash, started_at DESC);

            CREATE INDEX IF NOT EXISTS idx_realtime_session_runs_uploaded_at
                ON realtime_session_runs(uploaded_at DESC);

            CREATE TABLE IF NOT EXISTS feedback_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_hash TEXT NOT NULL,
                item_id TEXT,
                response_text TEXT,
                feedback_text TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'web_realtime_feedback',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                remote_addr TEXT,
                user_agent TEXT,
                UNIQUE(session_hash, item_id)
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_records_session_hash
                ON feedback_records(session_hash, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_feedback_records_response_text
                ON feedback_records(session_hash, response_text);
            """
        )
        ensure_column(
            connection,
            "realtime_sessions",
            "saved_file_paths_json",
            "TEXT NOT NULL DEFAULT '[]'",
        )
        ensure_column(
            connection,
            "realtime_session_runs",
            "memory_status",
            "TEXT NOT NULL DEFAULT ''",
        )
        ensure_column(connection, "realtime_session_runs", "memory_operation_id", "TEXT")
        ensure_column(connection, "realtime_session_runs", "memory_error", "TEXT")
        ensure_column(connection, "realtime_session_runs", "memory_queued_at", "TEXT")


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_json(value: Optional[str], default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def insert_google_form_response(db_path: Path, record: dict[str, Any]) -> None:
    with connect_database(db_path) as connection:
        connection.execute(
            """
            INSERT INTO google_form_responses (
                form_hash,
                form_dir_name,
                form_title,
                form_id,
                response_id,
                submitted_at,
                received_at,
                name,
                age,
                date_prefix,
                fields_json,
                phq8_json,
                remote_addr,
                user_agent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("form_hash") or "",
                record.get("form_dir_name") or "",
                record.get("form_title") or "",
                record.get("form_id") or "",
                record.get("response_id") or "",
                record.get("submitted_at") or "",
                record.get("received_at") or "",
                record.get("name") or "",
                record.get("age") or "",
                record.get("date_prefix") or "",
                dump_json(record.get("fields") or {}),
                dump_json(record.get("phq8") or {}),
                record.get("remote_addr"),
                record.get("user_agent"),
            ),
        )


def list_google_form_response_summaries(db_path: Path) -> list[dict[str, Any]]:
    with connect_database(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                form_hash,
                form_dir_name,
                name,
                age,
                submitted_at,
                received_at,
                phq8_json
            FROM google_form_responses
            ORDER BY received_at DESC, form_dir_name DESC
            """
        ).fetchall()

    summaries = []
    for row in rows:
        phq8 = load_json(row["phq8_json"], {})
        summaries.append(
            {
                "form_hash": row["form_hash"] or "",
                "form_dir_name": row["form_dir_name"] or "",
                "name": row["name"] or "",
                "age": row["age"] or "",
                "submitted_at": row["submitted_at"] or "",
                "received_at": row["received_at"] or "",
                "phq8_score": phq8.get("total_score") if isinstance(phq8, dict) else None,
                "phq8_answered_count": phq8.get("answered_count") if isinstance(phq8, dict) else None,
                "phq8": phq8 if isinstance(phq8, dict) else {},
                "google_form_response_file": "",
                "saved_to": "sqlite",
            }
        )
    return summaries


def upsert_realtime_session(db_path: Path, record: dict[str, Any]) -> None:
    saved_file_names = record.get("saved_file_names")
    if not isinstance(saved_file_names, list):
        saved_file_names = []
    saved_file_paths = record.get("saved_file_paths")
    if not isinstance(saved_file_paths, list):
        saved_file_paths = []

    with connect_database(db_path) as connection:
        connection.execute(
            """
            INSERT INTO realtime_sessions (
                session_hash,
                session_dir_name,
                created_at,
                updated_at,
                remote_addr,
                user_agent,
                metadata_json,
                transcript_json,
                transcript_text,
                saved_file_names_json,
                saved_file_paths_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_hash) DO UPDATE SET
                session_dir_name = excluded.session_dir_name,
                updated_at = excluded.updated_at,
                remote_addr = excluded.remote_addr,
                user_agent = excluded.user_agent,
                metadata_json = COALESCE(excluded.metadata_json, realtime_sessions.metadata_json),
                transcript_json = COALESCE(excluded.transcript_json, realtime_sessions.transcript_json),
                transcript_text = COALESCE(excluded.transcript_text, realtime_sessions.transcript_text),
                saved_file_names_json = CASE
                    WHEN excluded.saved_file_names_json IS NOT NULL
                     AND excluded.saved_file_names_json != '[]'
                    THEN excluded.saved_file_names_json
                    ELSE realtime_sessions.saved_file_names_json
                END,
                saved_file_paths_json = CASE
                    WHEN excluded.saved_file_paths_json IS NOT NULL
                     AND excluded.saved_file_paths_json != '[]'
                    THEN excluded.saved_file_paths_json
                    ELSE realtime_sessions.saved_file_paths_json
                END
            """,
            (
                record.get("session_hash") or "",
                record.get("session_dir_name") or "",
                record.get("created_at") or record.get("updated_at") or "",
                record.get("updated_at") or "",
                record.get("remote_addr"),
                record.get("user_agent"),
                dump_json(record.get("metadata")) if record.get("metadata") is not None else None,
                dump_json(record.get("transcript")) if record.get("transcript") is not None else None,
                record.get("transcript_text"),
                dump_json(saved_file_names),
                dump_json(saved_file_paths),
            ),
        )


def upsert_realtime_session_run(db_path: Path, record: dict[str, Any]) -> None:
    saved_file_names = record.get("saved_file_names")
    if not isinstance(saved_file_names, list):
        saved_file_names = []
    saved_file_paths = record.get("saved_file_paths")
    if not isinstance(saved_file_paths, list):
        saved_file_paths = []

    with connect_database(db_path) as connection:
        connection.execute(
            """
            INSERT INTO realtime_session_runs (
                session_hash,
                run_id,
                session_dir_name,
                started_at,
                ended_at,
                uploaded_at,
                remote_addr,
                user_agent,
                metadata_json,
                transcript_json,
                transcript_text,
                saved_file_names_json,
                saved_file_paths_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_hash, run_id) DO UPDATE SET
                session_dir_name = excluded.session_dir_name,
                started_at = excluded.started_at,
                ended_at = COALESCE(excluded.ended_at, realtime_session_runs.ended_at),
                uploaded_at = excluded.uploaded_at,
                remote_addr = COALESCE(excluded.remote_addr, realtime_session_runs.remote_addr),
                user_agent = COALESCE(excluded.user_agent, realtime_session_runs.user_agent),
                metadata_json = COALESCE(excluded.metadata_json, realtime_session_runs.metadata_json),
                transcript_json = COALESCE(excluded.transcript_json, realtime_session_runs.transcript_json),
                transcript_text = COALESCE(excluded.transcript_text, realtime_session_runs.transcript_text),
                saved_file_names_json = CASE
                    WHEN excluded.saved_file_names_json IS NOT NULL
                     AND excluded.saved_file_names_json != '[]'
                    THEN excluded.saved_file_names_json
                    ELSE realtime_session_runs.saved_file_names_json
                END,
                saved_file_paths_json = CASE
                    WHEN excluded.saved_file_paths_json IS NOT NULL
                     AND excluded.saved_file_paths_json != '[]'
                    THEN excluded.saved_file_paths_json
                    ELSE realtime_session_runs.saved_file_paths_json
                END
            """,
            (
                record.get("session_hash") or "",
                record.get("run_id") or "",
                record.get("session_dir_name") or "",
                record.get("started_at") or record.get("uploaded_at") or "",
                record.get("ended_at"),
                record.get("uploaded_at") or "",
                record.get("remote_addr"),
                record.get("user_agent"),
                dump_json(record.get("metadata")) if record.get("metadata") is not None else None,
                dump_json(record.get("transcript")) if record.get("transcript") is not None else None,
                record.get("transcript_text"),
                dump_json(saved_file_names),
                dump_json(saved_file_paths),
            ),
        )


def get_realtime_session_run(
    db_path: Path,
    session_hash: str,
    run_id: str,
) -> Optional[dict[str, Any]]:
    with connect_database(db_path) as connection:
        row = connection.execute(
            """
            SELECT
                session_hash,
                run_id,
                session_dir_name,
                started_at,
                ended_at,
                uploaded_at,
                remote_addr,
                user_agent,
                metadata_json,
                transcript_json,
                transcript_text,
                saved_file_names_json,
                saved_file_paths_json,
                memory_status,
                memory_operation_id,
                memory_error,
                memory_queued_at
            FROM realtime_session_runs
            WHERE session_hash = ? AND run_id = ?
            LIMIT 1
            """,
            (session_hash, run_id),
        ).fetchone()
    if row is None:
        return None
    return {
        "session_hash": row["session_hash"] or "",
        "run_id": row["run_id"] or "",
        "session_dir_name": row["session_dir_name"] or "",
        "started_at": row["started_at"] or "",
        "ended_at": row["ended_at"],
        "uploaded_at": row["uploaded_at"] or "",
        "remote_addr": row["remote_addr"],
        "user_agent": row["user_agent"],
        "metadata": load_json(row["metadata_json"], None),
        "transcript": load_json(row["transcript_json"], None),
        "transcript_text": row["transcript_text"],
        "saved_file_names": load_json(row["saved_file_names_json"], []),
        "saved_file_paths": load_json(row["saved_file_paths_json"], []),
        "memory_status": row["memory_status"] or "",
        "memory_operation_id": row["memory_operation_id"],
        "memory_error": row["memory_error"],
        "memory_queued_at": row["memory_queued_at"],
    }


def update_realtime_session_run_memory(
    db_path: Path,
    session_hash: str,
    run_id: str,
    memory_update: dict[str, Any],
) -> None:
    with connect_database(db_path) as connection:
        connection.execute(
            """
            UPDATE realtime_session_runs
            SET memory_status = ?,
                memory_operation_id = ?,
                memory_error = ?,
                memory_queued_at = ?
            WHERE session_hash = ? AND run_id = ?
            """,
            (
                memory_update.get("status") or "",
                memory_update.get("operation_id"),
                memory_update.get("message"),
                memory_update.get("queued_at"),
                session_hash,
                run_id,
            ),
        )


def get_realtime_session(db_path: Path, session_hash: str) -> Optional[dict[str, Any]]:
    with connect_database(db_path) as connection:
        row = connection.execute(
            """
            SELECT
                session_hash,
                session_dir_name,
                created_at,
                updated_at,
                remote_addr,
                user_agent,
                metadata_json,
                transcript_json,
                transcript_text,
                saved_file_names_json,
                saved_file_paths_json
            FROM realtime_sessions
            WHERE session_hash = ?
            LIMIT 1
            """,
            (session_hash,),
        ).fetchone()
    if row is None:
        return None
    return {
        "session_hash": row["session_hash"] or "",
        "session_dir_name": row["session_dir_name"] or "",
        "created_at": row["created_at"] or "",
        "updated_at": row["updated_at"] or "",
        "remote_addr": row["remote_addr"],
        "user_agent": row["user_agent"],
        "metadata": load_json(row["metadata_json"], None),
        "transcript": load_json(row["transcript_json"], None),
        "transcript_text": row["transcript_text"],
        "saved_file_names": load_json(row["saved_file_names_json"], []),
        "saved_file_paths": load_json(row["saved_file_paths_json"], []),
    }


def ensure_realtime_session(
    db_path: Path,
    session_hash: str,
    session_dir_name: str,
    now: str,
    remote_addr: Optional[str],
    user_agent: Optional[str],
) -> None:
    upsert_realtime_session(
        db_path,
        {
            "session_hash": session_hash,
            "session_dir_name": session_dir_name,
            "created_at": now,
            "updated_at": now,
            "remote_addr": remote_addr,
            "user_agent": user_agent,
            "saved_file_names": [],
        },
    )


def _find_feedback_record_id(
    connection: sqlite3.Connection,
    session_hash: str,
    item_id: str,
    response_text: str,
) -> Optional[int]:
    if item_id:
        row = connection.execute(
            """
            SELECT id
            FROM feedback_records
            WHERE session_hash = ? AND item_id = ?
            LIMIT 1
            """,
            (session_hash, item_id),
        ).fetchone()
        return int(row["id"]) if row else None

    if response_text:
        row = connection.execute(
            """
            SELECT id
            FROM feedback_records
            WHERE session_hash = ?
              AND (item_id IS NULL OR item_id = '')
              AND response_text = ?
            LIMIT 1
            """,
            (session_hash, response_text),
        ).fetchone()
        return int(row["id"]) if row else None

    return None


def upsert_feedback_record(db_path: Path, record: dict[str, Any]) -> None:
    upsert_feedback_records(db_path, [record])


def upsert_feedback_records(db_path: Path, records: Iterable[dict[str, Any]]) -> None:
    with connect_database(db_path) as connection:
        for raw_record in records:
            if not isinstance(raw_record, dict):
                continue

            session_hash = str(raw_record.get("session_hash") or "").strip()
            feedback_text = str(raw_record.get("feedback_text") or "").strip()
            if not session_hash or not feedback_text:
                continue

            item_id = str(raw_record.get("item_id") or "").strip()
            response_text = str(raw_record.get("response_text") or "").strip()
            updated_at = str(raw_record.get("updated_at") or raw_record.get("created_at") or "").strip()
            created_at = str(raw_record.get("created_at") or updated_at).strip()
            existing_id = _find_feedback_record_id(connection, session_hash, item_id, response_text)

            if existing_id is not None:
                connection.execute(
                    """
                    UPDATE feedback_records
                    SET item_id = ?,
                        response_text = ?,
                        feedback_text = ?,
                        source = ?,
                        updated_at = ?,
                        remote_addr = ?,
                        user_agent = ?
                    WHERE id = ?
                    """,
                    (
                        item_id or None,
                        response_text or None,
                        feedback_text,
                        raw_record.get("source") or "web_realtime_feedback",
                        updated_at,
                        raw_record.get("remote_addr"),
                        raw_record.get("user_agent"),
                        existing_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO feedback_records (
                        session_hash,
                        item_id,
                        response_text,
                        feedback_text,
                        source,
                        created_at,
                        updated_at,
                        remote_addr,
                        user_agent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_hash,
                        item_id or None,
                        response_text or None,
                        feedback_text,
                        raw_record.get("source") or "web_realtime_feedback",
                        created_at,
                        updated_at,
                        raw_record.get("remote_addr"),
                        raw_record.get("user_agent"),
                    ),
                )


def count_feedback_records(db_path: Path, session_hash: str) -> int:
    with connect_database(db_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM feedback_records WHERE session_hash = ?",
            (session_hash,),
        ).fetchone()
    return int(row["count"]) if row else 0
