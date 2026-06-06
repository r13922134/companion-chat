import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import server_realtime
from app.storage import initialize_database, upsert_realtime_session


def json_upload(payload, filename):
    return (
        io.BytesIO(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
        filename,
    )


class RealtimeSessionRunTests(unittest.TestCase):
    def post_session(self, client, session_hash, started_at, transcript_text):
        metadata = {
            "session_hash": session_hash,
            "started_at_iso": started_at,
            "ended_at_iso": "2026-06-06T10:05:00.000Z",
        }
        transcript = {
            "session_hash": session_hash,
            "events": [{"speaker": "user", "text": transcript_text}],
            "plain_text": f"USER: {transcript_text}",
        }
        with patch.object(server_realtime, "is_hindsight_enabled", return_value=False):
            return client.post(
                "/api/realtime-session",
                data={
                    "session_hash": session_hash,
                    "metadata_file": json_upload(metadata, "metadata.json"),
                    "transcript_file": json_upload(transcript, "transcript.json"),
                    "transcript_text_file": (
                        io.BytesIO(f"USER: {transcript_text}".encode("utf-8")),
                        "transcript.txt",
                    ),
                },
                content_type="multipart/form-data",
            )

    def test_same_hash_creates_multiple_runs_and_retry_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "app.sqlite3"
            upload_root = root / "realtime"
            initialize_database(database_path)

            with (
                patch.object(server_realtime, "DATABASE_PATH", database_path),
                patch.object(server_realtime, "UPLOAD_ROOT", upload_root),
            ):
                client = server_realtime.app.test_client()
                first = self.post_session(
                    client,
                    "same-hash",
                    "2026-06-06T10:00:00.123Z",
                    "第一輪",
                )
                second = self.post_session(
                    client,
                    "same-hash",
                    "2026-06-06T11:00:00.456Z",
                    "第二輪",
                )
                retry = self.post_session(
                    client,
                    "same-hash",
                    "2026-06-06T10:00:00.123Z",
                    "第一輪重送",
                )

            first_payload = first.get_json()
            second_payload = second.get_json()
            retry_payload = retry.get_json()
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(retry.status_code, 200)
            self.assertEqual(first_payload["run_id"], retry_payload["run_id"])
            self.assertNotEqual(first_payload["run_id"], second_payload["run_id"])
            self.assertEqual(
                first_payload["session_dir_name"],
                f"same-hash/{first_payload['run_id']}",
            )

            run_dirs = sorted(path.name for path in (upload_root / "same-hash").iterdir())
            self.assertEqual(run_dirs, sorted([first_payload["run_id"], second_payload["run_id"]]))
            first_transcript = (
                upload_root / "same-hash" / first_payload["run_id"] / "transcript.txt"
            ).read_text(encoding="utf-8")
            self.assertEqual(first_transcript, "USER: 第一輪重送")

            with sqlite3.connect(database_path) as connection:
                rows = connection.execute(
                    """
                    SELECT session_hash, run_id, session_dir_name, metadata_json, transcript_text
                    FROM realtime_session_runs
                    WHERE session_hash = ?
                    ORDER BY run_id
                    """,
                    ("same-hash",),
                ).fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][0], "same-hash")
            self.assertTrue(rows[0][2].startswith("same-hash/"))
            self.assertEqual(json.loads(rows[0][3])["session_hash"], "same-hash")
            self.assertIn("USER:", rows[0][4])

    def test_run_id_collision_gets_incrementing_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "app.sqlite3"
            upload_root = root / "realtime"
            initialize_database(database_path)

            with (
                patch.object(server_realtime, "DATABASE_PATH", database_path),
                patch.object(server_realtime, "UPLOAD_ROOT", upload_root),
            ):
                client = server_realtime.app.test_client()
                first = self.post_session(
                    client,
                    "collision-hash",
                    "2026-06-06T10:00:00.1231Z",
                    "第一輪",
                ).get_json()
                second = self.post_session(
                    client,
                    "collision-hash",
                    "2026-06-06T10:00:00.1239Z",
                    "第二輪",
                ).get_json()

            self.assertEqual(second["run_id"], f"{first['run_id']}_01")

    def test_legacy_directory_is_migrated_once_and_indexed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "app.sqlite3"
            upload_root = root / "realtime"
            feedback_root = root / "feedback"
            legacy_dir = upload_root / "20260606_legacy-hash"
            legacy_dir.mkdir(parents=True)
            feedback_root.mkdir()
            (feedback_root / "keep.txt").write_text("untouched", encoding="utf-8")
            metadata = {
                "session_hash": "legacy-hash",
                "started_at_iso": "2026-06-06T10:01:22.086Z",
                "ended_at_iso": "2026-06-06T10:01:27.896Z",
            }
            (legacy_dir / "metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            (legacy_dir / "transcript.json").write_text(
                json.dumps({"events": []}),
                encoding="utf-8",
            )
            (legacy_dir / "transcript.txt").write_text("legacy", encoding="utf-8")

            initialize_database(database_path)
            upsert_realtime_session(
                database_path,
                {
                    "session_hash": "legacy-hash",
                    "session_dir_name": legacy_dir.name,
                    "created_at": metadata["started_at_iso"],
                    "updated_at": "2026-06-06T18:01:28+08:00",
                    "remote_addr": "127.0.0.1",
                    "user_agent": "test-agent",
                    "saved_file_names": ["metadata.json", "transcript.json", "transcript.txt"],
                },
            )

            server_realtime.migrate_legacy_realtime_sessions(database_path, upload_root)
            server_realtime.migrate_legacy_realtime_sessions(database_path, upload_root)

            expected_run_id = server_realtime.format_session_run_id(
                server_realtime.parse_session_datetime(metadata["started_at_iso"])
            )
            migrated_dir = upload_root / "legacy-hash" / expected_run_id
            self.assertFalse(legacy_dir.exists())
            self.assertTrue((migrated_dir / "metadata.json").is_file())
            self.assertEqual(
                (feedback_root / "keep.txt").read_text(encoding="utf-8"),
                "untouched",
            )

            with sqlite3.connect(database_path) as connection:
                rows = connection.execute(
                    """
                    SELECT session_hash, run_id, session_dir_name, remote_addr, user_agent
                    FROM realtime_session_runs
                    WHERE session_hash = ?
                    """,
                    ("legacy-hash",),
                ).fetchall()
            self.assertEqual(
                rows,
                [
                    (
                        "legacy-hash",
                        expected_run_id,
                        f"legacy-hash/{expected_run_id}",
                        "127.0.0.1",
                        "test-agent",
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
