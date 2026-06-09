from __future__ import annotations

import json
import os
import sqlite3
import time
import hashlib
from datetime import datetime
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


def normalize_respondent_email(value: Any) -> str:
    return str(value or "").strip().lower()


def google_form_account_hash(respondent_email: Any) -> str:
    email = normalize_respondent_email(respondent_email)
    if not email:
        return ""
    digest = hashlib.sha256(email.encode("utf-8")).hexdigest()[:20]
    return f"acct_{digest}"


def google_form_account_key(respondent_email: Any, fallback_form_hash: Any = "") -> str:
    return google_form_account_hash(respondent_email) or str(fallback_form_hash or "").strip()


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
                respondent_email TEXT NOT NULL DEFAULT '',
                account_hash TEXT NOT NULL DEFAULT '',
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
                depression_status TEXT NOT NULL DEFAULT '',
                depression_total_score REAL,
                depression_binary INTEGER,
                depression_result_json TEXT,
                depression_error TEXT,
                depression_completed_at TEXT,
                depression_queued_at TEXT,
                depression_started_at TEXT,
                depression_worker_id TEXT,
                archive_manifest_json TEXT,
                ground_truth_total_score REAL,
                ground_truth_binary INTEGER,
                UNIQUE(session_hash, run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_realtime_session_runs_hash_started
                ON realtime_session_runs(session_hash, started_at DESC);

            CREATE INDEX IF NOT EXISTS idx_realtime_session_runs_uploaded_at
                ON realtime_session_runs(uploaded_at DESC);

            CREATE TABLE IF NOT EXISTS depression_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_hash TEXT NOT NULL,
                run_id TEXT NOT NULL,
                session_dir TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                priority INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                queued_at TEXT NOT NULL,
                available_at_epoch REAL NOT NULL,
                claimed_at TEXT,
                heartbeat_at TEXT,
                lease_expires_at_epoch REAL,
                completed_at TEXT,
                worker_id TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(session_hash, run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_depression_jobs_claim
                ON depression_jobs(status, available_at_epoch, priority DESC, id);

            CREATE INDEX IF NOT EXISTS idx_depression_jobs_lease
                ON depression_jobs(status, lease_expires_at_epoch);

            CREATE TABLE IF NOT EXISTS depression_workers (
                worker_id TEXT PRIMARY KEY,
                gpu_id TEXT NOT NULL,
                pid INTEGER NOT NULL,
                hostname TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL,
                details_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_depression_workers_heartbeat
                ON depression_workers(status, heartbeat_at DESC);

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
        ensure_column(
            connection,
            "realtime_session_runs",
            "depression_status",
            "TEXT NOT NULL DEFAULT ''",
        )
        ensure_column(connection, "realtime_session_runs", "depression_total_score", "REAL")
        ensure_column(connection, "realtime_session_runs", "depression_binary", "INTEGER")
        ensure_column(connection, "realtime_session_runs", "depression_result_json", "TEXT")
        ensure_column(connection, "realtime_session_runs", "depression_error", "TEXT")
        ensure_column(connection, "realtime_session_runs", "depression_completed_at", "TEXT")
        ensure_column(connection, "realtime_session_runs", "depression_queued_at", "TEXT")
        ensure_column(connection, "realtime_session_runs", "depression_started_at", "TEXT")
        ensure_column(connection, "realtime_session_runs", "depression_worker_id", "TEXT")
        ensure_column(connection, "realtime_session_runs", "archive_manifest_json", "TEXT")
        ensure_column(connection, "realtime_session_runs", "ground_truth_total_score", "REAL")
        ensure_column(connection, "realtime_session_runs", "ground_truth_binary", "INTEGER")
        ensure_column(
            connection,
            "google_form_responses",
            "respondent_email",
            "TEXT NOT NULL DEFAULT ''",
        )
        ensure_column(
            connection,
            "google_form_responses",
            "account_hash",
            "TEXT NOT NULL DEFAULT ''",
        )
        connection.execute(
            """
            DELETE FROM google_form_responses
            WHERE respondent_email IS NULL OR trim(respondent_email) = ''
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_google_form_responses_account_received_at
                ON google_form_responses(account_hash, received_at DESC, submitted_at DESC)
            """
        )
        rows = connection.execute(
            """
            SELECT form_hash, respondent_email, account_hash
            FROM google_form_responses
            """
        ).fetchall()
        for row in rows:
            account_hash = google_form_account_key(
                row["respondent_email"],
                row["form_hash"],
            )
            if account_hash == str(row["account_hash"] or ""):
                continue
            connection.execute(
                """
                UPDATE google_form_responses
                SET account_hash = ?
                WHERE form_hash = ?
                """,
                (account_hash, row["form_hash"]),
            )


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
    form_hash = str(record.get("form_hash") or "").strip()
    respondent_email = normalize_respondent_email(record.get("respondent_email"))
    if not respondent_email:
        raise ValueError("respondent_email is required for google form responses")
    account_hash = google_form_account_key(
        respondent_email,
        str(record.get("account_hash") or "").strip() or form_hash,
    )
    with connect_database(db_path) as connection:
        connection.execute(
            """
            INSERT INTO google_form_responses (
                form_hash,
                form_dir_name,
                form_title,
                form_id,
                response_id,
                respondent_email,
                account_hash,
                submitted_at,
                received_at,
                name,
                age,
                date_prefix,
                fields_json,
                phq8_json,
                remote_addr,
                user_agent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(form_hash) DO UPDATE SET
                form_dir_name = excluded.form_dir_name,
                form_title = excluded.form_title,
                form_id = excluded.form_id,
                response_id = excluded.response_id,
                respondent_email = excluded.respondent_email,
                account_hash = excluded.account_hash,
                submitted_at = excluded.submitted_at,
                received_at = excluded.received_at,
                name = excluded.name,
                age = excluded.age,
                date_prefix = excluded.date_prefix,
                fields_json = excluded.fields_json,
                phq8_json = excluded.phq8_json,
                remote_addr = excluded.remote_addr,
                user_agent = excluded.user_agent
            """,
            (
                form_hash,
                record.get("form_dir_name") or "",
                record.get("form_title") or "",
                record.get("form_id") or "",
                record.get("response_id") or "",
                respondent_email,
                account_hash,
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


def _google_form_response_from_row(row: sqlite3.Row) -> dict[str, Any]:
    phq8 = load_json(row["phq8_json"], {})
    respondent_email = normalize_respondent_email(row["respondent_email"])
    form_hash = row["form_hash"] or ""
    account_hash = google_form_account_key(
        respondent_email,
        row["account_hash"] or form_hash,
    )
    return {
        "user_key": account_hash,
        "account_hash": account_hash,
        "session_hash": account_hash,
        "respondent_email": respondent_email,
        "form_hash": form_hash,
        "latest_form_hash": form_hash,
        "form_dir_name": row["form_dir_name"] or "",
        "form_title": row["form_title"] or "",
        "form_id": row["form_id"] or "",
        "response_id": row["response_id"] or "",
        "name": row["name"] or "",
        "age": row["age"] or "",
        "submitted_at": row["submitted_at"] or "",
        "received_at": row["received_at"] or "",
        "date_prefix": row["date_prefix"] or "",
        "fields": load_json(row["fields_json"], {}),
        "phq8_score": phq8.get("total_score") if isinstance(phq8, dict) else None,
        "phq8_answered_count": phq8.get("answered_count") if isinstance(phq8, dict) else None,
        "phq8": phq8 if isinstance(phq8, dict) else {},
        "google_form_response_file": "",
        "saved_to": "sqlite",
    }


def get_latest_google_form_response_for_account(
    db_path: Path,
    *,
    account_hash: Any = "",
    respondent_email: Any = "",
    form_hash: Any = "",
) -> Optional[dict[str, Any]]:
    normalized_email = normalize_respondent_email(respondent_email)
    candidate_account_hashes = []
    for candidate in (
        account_hash,
        google_form_account_hash(normalized_email),
        form_hash,
    ):
        key = str(candidate or "").strip()
        if key and key not in candidate_account_hashes:
            candidate_account_hashes.append(key)

    with connect_database(db_path) as connection:
        row = None
        for candidate in candidate_account_hashes:
            row = connection.execute(
                """
                SELECT
                    form_hash,
                    form_dir_name,
                    form_title,
                    form_id,
                    response_id,
                    respondent_email,
                    account_hash,
                    submitted_at,
                    received_at,
                    name,
                    age,
                    date_prefix,
                    fields_json,
                    phq8_json
                FROM google_form_responses
                WHERE account_hash = ?
                ORDER BY submitted_at DESC, received_at DESC, form_dir_name DESC
                LIMIT 1
                """,
                (candidate,),
            ).fetchone()
            if row is not None:
                return _google_form_response_from_row(row)

        if normalized_email:
            row = connection.execute(
                """
                SELECT
                    form_hash,
                    form_dir_name,
                    form_title,
                    form_id,
                    response_id,
                    respondent_email,
                    account_hash,
                    submitted_at,
                    received_at,
                    name,
                    age,
                    date_prefix,
                    fields_json,
                    phq8_json
                FROM google_form_responses
                WHERE lower(respondent_email) = ?
                ORDER BY submitted_at DESC, received_at DESC, form_dir_name DESC
                LIMIT 1
                """,
                (normalized_email,),
            ).fetchone()
            if row is not None:
                return _google_form_response_from_row(row)

        form_hash_key = str(form_hash or "").strip()
        if form_hash_key:
            row = connection.execute(
                """
                SELECT
                    form_hash,
                    form_dir_name,
                    form_title,
                    form_id,
                    response_id,
                    respondent_email,
                    account_hash,
                    submitted_at,
                    received_at,
                    name,
                    age,
                    date_prefix,
                    fields_json,
                    phq8_json
                FROM google_form_responses
                WHERE form_hash = ?
                ORDER BY submitted_at DESC, received_at DESC, form_dir_name DESC
                LIMIT 1
                """,
                (form_hash_key,),
            ).fetchone()
            if row is not None:
                return _google_form_response_from_row(row)
    return None


def google_form_response_to_selected_user(response: dict[str, Any]) -> dict[str, Any]:
    account_hash = str(response.get("account_hash") or response.get("user_key") or "").strip()
    form_hash = str(response.get("form_hash") or "").strip()
    phq8 = response.get("phq8") if isinstance(response.get("phq8"), dict) else {}
    return {
        "user_key": account_hash or form_hash,
        "account_hash": account_hash or form_hash,
        "session_hash": account_hash or form_hash,
        "respondent_email": normalize_respondent_email(response.get("respondent_email")),
        "form_hash": form_hash,
        "latest_form_hash": str(response.get("latest_form_hash") or form_hash),
        "form_dir_name": response.get("form_dir_name") or "",
        "form_title": response.get("form_title") or "",
        "form_id": response.get("form_id") or "",
        "response_id": response.get("response_id") or "",
        "name": response.get("name") or "",
        "age": response.get("age") or "",
        "submitted_at": response.get("submitted_at") or "",
        "received_at": response.get("received_at") or "",
        "phq8_score": response.get("phq8_score"),
        "phq8_answered_count": response.get("phq8_answered_count"),
        "phq8": phq8,
    }


def latest_google_form_response_for_metadata(
    db_path: Path,
    *,
    session_hash: Any = "",
    metadata: Any = None,
) -> Optional[dict[str, Any]]:
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    selected_user = metadata_dict.get("selected_user")
    selected_user = selected_user if isinstance(selected_user, dict) else {}

    account_candidates = [
        selected_user.get("account_hash"),
        selected_user.get("user_key"),
        selected_user.get("session_hash"),
        metadata_dict.get("account_hash"),
        session_hash,
    ]
    email = (
        selected_user.get("respondent_email")
        or selected_user.get("email")
        or metadata_dict.get("respondent_email")
        or metadata_dict.get("email")
    )
    form_hash = selected_user.get("latest_form_hash") or selected_user.get("form_hash")

    seen: set[str] = set()
    for account_hash in account_candidates:
        key = str(account_hash or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        response = get_latest_google_form_response_for_account(
            db_path,
            account_hash=key,
            respondent_email=email,
            form_hash=form_hash,
        )
        if response is not None:
            return response

    return get_latest_google_form_response_for_account(
        db_path,
        respondent_email=email,
        form_hash=form_hash,
    )


def attach_latest_google_form_response_to_metadata(
    db_path: Path,
    *,
    session_hash: Any = "",
    metadata: Any = None,
) -> tuple[dict[str, Any], bool]:
    metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}
    response = latest_google_form_response_for_metadata(
        db_path,
        session_hash=session_hash,
        metadata=metadata_dict,
    )
    if response is None:
        return metadata_dict, False

    selected_user = metadata_dict.get("selected_user")
    selected_user = dict(selected_user) if isinstance(selected_user, dict) else {}
    latest_user = google_form_response_to_selected_user(response)
    updated_user = {**selected_user, **latest_user}
    changed = updated_user != selected_user
    metadata_dict["selected_user"] = updated_user
    metadata_dict["ground_truth_source"] = "google_form_responses.latest_by_account.phq8"
    return metadata_dict, changed


def list_google_form_response_summaries(db_path: Path) -> list[dict[str, Any]]:
    with connect_database(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                form_hash,
                form_dir_name,
                form_title,
                form_id,
                response_id,
                respondent_email,
                account_hash,
                name,
                age,
                submitted_at,
                received_at,
                date_prefix,
                fields_json,
                phq8_json
            FROM google_form_responses
            ORDER BY submitted_at DESC, received_at DESC, form_dir_name DESC
            """
        ).fetchall()

    summaries_by_account: dict[str, dict[str, Any]] = {}
    for row in rows:
        summary = _google_form_response_from_row(row)
        account_hash = summary["account_hash"] or summary["form_hash"]
        if account_hash not in summaries_by_account:
            summary["form_count"] = 0
            summaries_by_account[account_hash] = summary
        summaries_by_account[account_hash]["form_count"] += 1
    return list(summaries_by_account.values())


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
                saved_file_paths_json,
                archive_manifest_json,
                ground_truth_total_score,
                ground_truth_binary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                archive_manifest_json = COALESCE(
                    excluded.archive_manifest_json,
                    realtime_session_runs.archive_manifest_json
                ),
                ground_truth_total_score = COALESCE(
                    excluded.ground_truth_total_score,
                    realtime_session_runs.ground_truth_total_score
                ),
                ground_truth_binary = COALESCE(
                    excluded.ground_truth_binary,
                    realtime_session_runs.ground_truth_binary
                ),
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
                (
                    dump_json(record.get("archive_manifest"))
                    if record.get("archive_manifest") is not None
                    else None
                ),
                record.get("ground_truth_total_score"),
                (
                    None
                    if record.get("ground_truth_binary") is None
                    else int(bool(record.get("ground_truth_binary")))
                ),
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
                memory_queued_at,
                depression_status,
                depression_total_score,
                depression_binary,
                depression_result_json,
                depression_error,
                depression_completed_at,
                depression_queued_at,
                depression_started_at,
                depression_worker_id,
                archive_manifest_json,
                ground_truth_total_score,
                ground_truth_binary
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
        "depression_status": row["depression_status"] or "",
        "depression_total_score": row["depression_total_score"],
        "depression_binary": (
            None if row["depression_binary"] is None else bool(row["depression_binary"])
        ),
        "depression_result": load_json(row["depression_result_json"], None),
        "depression_error": row["depression_error"],
        "depression_completed_at": row["depression_completed_at"],
        "depression_queued_at": row["depression_queued_at"],
        "depression_started_at": row["depression_started_at"],
        "depression_worker_id": row["depression_worker_id"],
        "archive_manifest": load_json(row["archive_manifest_json"], None),
        "ground_truth_total_score": row["ground_truth_total_score"],
        "ground_truth_binary": (
            None
            if row["ground_truth_binary"] is None
            else bool(row["ground_truth_binary"])
        ),
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


def update_realtime_session_run_depression(
    db_path: Path,
    session_hash: str,
    run_id: str,
    depression_update: dict[str, Any],
) -> None:
    result = depression_update.get("result")
    if result is None:
        result_json = depression_update.get("result_json")
    else:
        result_json = dump_json(result)
    binary = depression_update.get("binary")
    if binary is None:
        binary = depression_update.get("depression_binary")

    with connect_database(db_path) as connection:
        connection.execute(
            """
            UPDATE realtime_session_runs
            SET depression_status = ?,
                depression_total_score = ?,
                depression_binary = ?,
                depression_result_json = ?,
                depression_error = ?,
                depression_completed_at = ?,
                ground_truth_total_score = COALESCE(?, ground_truth_total_score),
                ground_truth_binary = COALESCE(?, ground_truth_binary)
            WHERE session_hash = ? AND run_id = ?
            """,
            (
                depression_update.get("status") or "",
                depression_update.get("total_score"),
                None if binary is None else int(bool(binary)),
                result_json,
                depression_update.get("error") or depression_update.get("message"),
                depression_update.get("completed_at"),
                depression_update.get("ground_truth_total_score"),
                (
                    None
                    if depression_update.get("ground_truth_binary") is None
                    else int(bool(depression_update.get("ground_truth_binary")))
                ),
                session_hash,
                run_id,
            ),
        )


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def enqueue_depression_job(
    db_path: Path,
    session_hash: str,
    run_id: str,
    session_dir: Path,
    *,
    max_attempts: int = 3,
    priority: int = 0,
) -> dict[str, Any]:
    queued_at = _now_iso()
    available_at_epoch = time.time()
    max_attempts = max(1, int(max_attempts))
    priority = int(priority)
    with connect_database(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            """
            SELECT id, status, attempts, max_attempts, queued_at
            FROM depression_jobs
            WHERE session_hash = ? AND run_id = ?
            """,
            (session_hash, run_id),
        ).fetchone()
        if existing is None:
            cursor = connection.execute(
                """
                INSERT INTO depression_jobs (
                    session_hash,
                    run_id,
                    session_dir,
                    status,
                    priority,
                    attempts,
                    max_attempts,
                    queued_at,
                    available_at_epoch,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, 'queued', ?, 0, ?, ?, ?, ?, ?)
                """,
                (
                    session_hash,
                    run_id,
                    str(Path(session_dir).resolve()),
                    priority,
                    max_attempts,
                    queued_at,
                    available_at_epoch,
                    queued_at,
                    queued_at,
                ),
            )
            job_id = int(cursor.lastrowid)
            attempts = 0
            status = "queued"
        elif existing["status"] in {"queued", "running", "completed"}:
            job_id = int(existing["id"])
            attempts = int(existing["attempts"])
            max_attempts = int(existing["max_attempts"])
            queued_at = str(existing["queued_at"])
            status = str(existing["status"])
        else:
            job_id = int(existing["id"])
            attempts = 0
            status = "queued"
            connection.execute(
                """
                UPDATE depression_jobs
                SET session_dir = ?,
                    status = 'queued',
                    priority = ?,
                    attempts = 0,
                    max_attempts = ?,
                    queued_at = ?,
                    available_at_epoch = ?,
                    claimed_at = NULL,
                    heartbeat_at = NULL,
                    lease_expires_at_epoch = NULL,
                    completed_at = NULL,
                    worker_id = NULL,
                    last_error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    str(Path(session_dir).resolve()),
                    priority,
                    max_attempts,
                    queued_at,
                    available_at_epoch,
                    queued_at,
                    job_id,
                ),
            )
        if status == "queued":
            connection.execute(
                """
                UPDATE realtime_session_runs
                SET depression_status = 'queued',
                    depression_total_score = NULL,
                    depression_binary = NULL,
                    depression_result_json = NULL,
                    depression_error = NULL,
                    depression_completed_at = NULL,
                    depression_queued_at = ?,
                    depression_started_at = NULL,
                    depression_worker_id = NULL
                WHERE session_hash = ? AND run_id = ?
                """,
                (queued_at, session_hash, run_id),
            )
    return {
        "id": job_id,
        "status": status,
        "attempts": attempts,
        "max_attempts": max_attempts,
        "queued_at": queued_at,
    }


def _recover_expired_depression_jobs(
    connection: sqlite3.Connection,
    *,
    now_epoch: float,
    now_iso: str,
) -> tuple[int, int]:
    expired = connection.execute(
        """
        SELECT id, session_hash, run_id, attempts, max_attempts, worker_id
        FROM depression_jobs
        WHERE status = 'running'
          AND lease_expires_at_epoch IS NOT NULL
          AND lease_expires_at_epoch < ?
        """,
        (now_epoch,),
    ).fetchall()
    requeued = 0
    failed = 0
    for row in expired:
        message = (
            "Depression worker lease expired"
            + (f" ({row['worker_id']})" if row["worker_id"] else "")
            + "."
        )
        if int(row["attempts"]) >= int(row["max_attempts"]):
            connection.execute(
                """
                UPDATE depression_jobs
                SET status = 'error',
                    completed_at = ?,
                    heartbeat_at = NULL,
                    lease_expires_at_epoch = NULL,
                    last_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now_iso, message, now_iso, row["id"]),
            )
            connection.execute(
                """
                UPDATE realtime_session_runs
                SET depression_status = 'error',
                    depression_error = ?,
                    depression_completed_at = ?
                WHERE session_hash = ? AND run_id = ?
                """,
                (message, now_iso, row["session_hash"], row["run_id"]),
            )
            failed += 1
        else:
            connection.execute(
                """
                UPDATE depression_jobs
                SET status = 'queued',
                    available_at_epoch = ?,
                    claimed_at = NULL,
                    heartbeat_at = NULL,
                    lease_expires_at_epoch = NULL,
                    worker_id = NULL,
                    last_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now_epoch, message, now_iso, row["id"]),
            )
            connection.execute(
                """
                UPDATE realtime_session_runs
                SET depression_status = 'queued',
                    depression_error = ?,
                    depression_started_at = NULL,
                    depression_worker_id = NULL
                WHERE session_hash = ? AND run_id = ?
                """,
                (message, row["session_hash"], row["run_id"]),
            )
            requeued += 1
    return requeued, failed


def claim_next_depression_job(
    db_path: Path,
    *,
    worker_id: str,
    lease_seconds: float = 300.0,
) -> Optional[dict[str, Any]]:
    now_epoch = time.time()
    now_iso = _now_iso()
    lease_seconds = max(30.0, float(lease_seconds))
    with connect_database(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        _recover_expired_depression_jobs(
            connection,
            now_epoch=now_epoch,
            now_iso=now_iso,
        )
        row = connection.execute(
            """
            SELECT *
            FROM depression_jobs
            WHERE status = 'queued'
              AND attempts < max_attempts
              AND available_at_epoch <= ?
            ORDER BY priority DESC, id ASC
            LIMIT 1
            """,
            (now_epoch,),
        ).fetchone()
        if row is None:
            return None
        lease_expires_at_epoch = now_epoch + lease_seconds
        cursor = connection.execute(
            """
            UPDATE depression_jobs
            SET status = 'running',
                attempts = attempts + 1,
                claimed_at = ?,
                heartbeat_at = ?,
                lease_expires_at_epoch = ?,
                completed_at = NULL,
                worker_id = ?,
                updated_at = ?
            WHERE id = ? AND status = 'queued'
            """,
            (
                now_iso,
                now_iso,
                lease_expires_at_epoch,
                worker_id,
                now_iso,
                row["id"],
            ),
        )
        if cursor.rowcount != 1:
            return None
        connection.execute(
            """
            UPDATE realtime_session_runs
            SET depression_status = 'running',
                depression_error = NULL,
                depression_started_at = ?,
                depression_worker_id = ?
            WHERE session_hash = ? AND run_id = ?
            """,
            (now_iso, worker_id, row["session_hash"], row["run_id"]),
        )
        claimed = dict(row)
        claimed.update(
            {
                "status": "running",
                "attempts": int(row["attempts"]) + 1,
                "claimed_at": now_iso,
                "heartbeat_at": now_iso,
                "lease_expires_at_epoch": lease_expires_at_epoch,
                "worker_id": worker_id,
            }
        )
        return claimed


def heartbeat_depression_job(
    db_path: Path,
    job_id: int,
    *,
    worker_id: str,
    lease_seconds: float = 300.0,
) -> bool:
    now_iso = _now_iso()
    lease_expires_at_epoch = time.time() + max(30.0, float(lease_seconds))
    with connect_database(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE depression_jobs
            SET heartbeat_at = ?,
                lease_expires_at_epoch = ?,
                updated_at = ?
            WHERE id = ? AND status = 'running' AND worker_id = ?
            """,
            (now_iso, lease_expires_at_epoch, now_iso, int(job_id), worker_id),
        )
        return cursor.rowcount == 1


def finish_depression_job(
    db_path: Path,
    job_id: int,
    *,
    worker_id: str,
    status: str,
    error: str | None = None,
) -> bool:
    if status not in {"completed", "error"}:
        raise ValueError("Depression job status must be completed or error.")
    completed_at = _now_iso()
    with connect_database(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE depression_jobs
            SET status = ?,
                heartbeat_at = NULL,
                lease_expires_at_epoch = NULL,
                completed_at = ?,
                last_error = ?,
                updated_at = ?
            WHERE id = ? AND status = 'running' AND worker_id = ?
            """,
            (
                status,
                completed_at,
                error,
                completed_at,
                int(job_id),
                worker_id,
            ),
        )
        return cursor.rowcount == 1


def get_depression_job(
    db_path: Path,
    session_hash: str,
    run_id: str,
) -> Optional[dict[str, Any]]:
    with connect_database(db_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM depression_jobs
            WHERE session_hash = ? AND run_id = ?
            LIMIT 1
            """,
            (session_hash, run_id),
        ).fetchone()
    return dict(row) if row is not None else None


def depression_queue_counts(db_path: Path) -> dict[str, int]:
    with connect_database(db_path) as connection:
        rows = connection.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM depression_jobs
            GROUP BY status
            """
        ).fetchall()
    return {str(row["status"]): int(row["count"]) for row in rows}


def update_depression_worker(
    db_path: Path,
    *,
    worker_id: str,
    gpu_id: str,
    pid: int,
    hostname: str,
    status: str,
    started_at: str,
    details: dict[str, Any] | None = None,
) -> None:
    heartbeat_at = _now_iso()
    with connect_database(db_path) as connection:
        connection.execute(
            """
            INSERT INTO depression_workers (
                worker_id,
                gpu_id,
                pid,
                hostname,
                status,
                started_at,
                heartbeat_at,
                details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                status = excluded.status,
                heartbeat_at = excluded.heartbeat_at,
                details_json = COALESCE(
                    excluded.details_json,
                    depression_workers.details_json
                )
            """,
            (
                worker_id,
                gpu_id,
                int(pid),
                hostname,
                status,
                started_at,
                heartbeat_at,
                dump_json(details) if details is not None else None,
            ),
        )


def list_depression_workers(db_path: Path) -> list[dict[str, Any]]:
    with connect_database(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                worker_id,
                gpu_id,
                pid,
                hostname,
                status,
                started_at,
                heartbeat_at,
                details_json
            FROM depression_workers
            ORDER BY heartbeat_at DESC
            """
        ).fetchall()
    return [
        {
            "worker_id": row["worker_id"],
            "gpu_id": row["gpu_id"],
            "pid": row["pid"],
            "hostname": row["hostname"],
            "status": row["status"],
            "started_at": row["started_at"],
            "heartbeat_at": row["heartbeat_at"],
            "details": load_json(row["details_json"], None),
        }
        for row in rows
    ]


def heartbeat_depression_worker(
    db_path: Path,
    *,
    worker_id: str,
    status: str | None = None,
) -> bool:
    heartbeat_at = _now_iso()
    with connect_database(db_path) as connection:
        if status is None:
            cursor = connection.execute(
                """
                UPDATE depression_workers
                SET heartbeat_at = ?
                WHERE worker_id = ?
                """,
                (heartbeat_at, worker_id),
            )
        else:
            cursor = connection.execute(
                """
                UPDATE depression_workers
                SET status = ?,
                    heartbeat_at = ?
                WHERE worker_id = ?
                """,
                (status, heartbeat_at, worker_id),
            )
        return cursor.rowcount == 1


def update_realtime_session_run_artifacts(
    db_path: Path,
    session_hash: str,
    run_id: str,
    *,
    saved_file_names: list[str],
    saved_file_paths: list[dict[str, Any]],
    archive_manifest: dict[str, Any] | None = None,
) -> None:
    with connect_database(db_path) as connection:
        connection.execute(
            """
            UPDATE realtime_session_runs
            SET saved_file_names_json = ?,
                saved_file_paths_json = ?,
                archive_manifest_json = COALESCE(?, archive_manifest_json)
            WHERE session_hash = ? AND run_id = ?
            """,
            (
                dump_json(saved_file_names),
                dump_json(saved_file_paths),
                dump_json(archive_manifest) if archive_manifest is not None else None,
                session_hash,
                run_id,
            ),
        )


def mark_stale_depression_runs_interrupted(
    db_path: Path,
    *,
    completed_at: str,
    message: str,
) -> int:
    result_json = dump_json(
        {
            "status": "interrupted",
            "completed_at": completed_at,
            "error": message,
        }
    )
    with connect_database(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE realtime_session_runs
            SET depression_status = 'interrupted',
                depression_result_json = ?,
                depression_error = ?,
                depression_completed_at = ?
            WHERE depression_status IN ('queued', 'running')
            """,
            (result_json, message, completed_at),
        )
        return int(cursor.rowcount or 0)


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
