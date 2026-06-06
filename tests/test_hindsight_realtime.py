import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import server_realtime
from app.storage import get_realtime_session_run, initialize_database


def json_upload(payload, filename):
    return (
        io.BytesIO(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
        filename,
    )


class HindsightRecallTests(unittest.TestCase):
    def setUp(self):
        self.client = server_realtime.app.test_client()

    def test_recall_adds_personal_context_to_realtime_instructions(self):
        recall_response = {
            "results": [
                {
                    "text": "使用者曾說他比較喜歡先被傾聽。",
                    "occurred_start": "2026-06-01T10:00:00Z",
                }
            ]
        }
        with (
            patch.object(server_realtime, "is_hindsight_enabled", return_value=True),
            patch.object(
                server_realtime,
                "post_hindsight_json",
                return_value=recall_response,
            ) as post_hindsight,
        ):
            response = self.client.post(
                "/api/realtime-response-instructions",
                json={
                    "kind": "default",
                    "conversation_mode": "listening",
                    "session_hash": "person-hash",
                    "user_transcript": "我今天很累",
                    "recall_memory": True,
                },
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["memory_status"], "recalled")
        self.assertIn("使用者曾說他比較喜歡先被傾聽", payload["memory_context"])
        self.assertIn("# Long-term User Context", payload["instructions"])
        self.assertIn("NEVER 把記憶當成診斷", payload["instructions"])
        path, request_payload = post_hindsight.call_args.args[:2]
        self.assertEqual(
            path,
            "/v1/default/banks/person-hash/memories/recall",
        )
        self.assertEqual(request_payload["query"], "我今天很累")
        self.assertEqual(request_payload["max_tokens"], 1200)

    def test_tool_followup_reuses_context_without_recalling(self):
        with patch.object(server_realtime, "post_hindsight_json") as post_hindsight:
            response = self.client.post(
                "/api/realtime-response-instructions",
                json={
                    "kind": "medical_qa_assistant",
                    "conversation_mode": "listening",
                    "session_hash": "person-hash",
                    "user_transcript": "治療後很累正常嗎",
                    "memory_context": "- 使用者希望回答簡短。",
                    "memory_status": "recalled",
                    "recall_memory": False,
                },
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["memory_status"], "recalled")
        self.assertIn("使用者希望回答簡短", payload["instructions"])
        post_hindsight.assert_not_called()

    def test_recall_failure_is_fail_open(self):
        with (
            patch.object(server_realtime, "is_hindsight_enabled", return_value=True),
            patch.object(
                server_realtime,
                "post_hindsight_json",
                side_effect=RuntimeError("service unavailable"),
            ),
        ):
            response = self.client.post(
                "/api/realtime-response-instructions",
                json={
                    "kind": "default",
                    "conversation_mode": "listening",
                    "session_hash": "person-hash",
                    "user_transcript": "我睡不著",
                    "recall_memory": True,
                },
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["memory_status"], "error")
        self.assertEqual(payload["memory_context"], "")
        self.assertIn("# Role & Objective", payload["instructions"])
        self.assertNotIn("# Long-term User Context", payload["instructions"])


class HindsightRetainTests(unittest.TestCase):
    def test_session_end_retains_text_only_and_records_operation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "app.sqlite3"
            upload_root = root / "realtime"
            initialize_database(database_path)
            metadata = {
                "session_hash": "person-hash",
                "started_at_iso": "2026-06-06T10:00:00.123Z",
                "ended_at_iso": "2026-06-06T10:05:00.000Z",
            }
            transcript = {
                "session_hash": "person-hash",
                "events": [
                    {
                        "speaker": "user",
                        "text": "我今天很累",
                        "timestamp_ms": 1780711200123,
                    },
                    {
                        "speaker": "assistant",
                        "text": "聽起來今天真的撐得很辛苦。",
                        "timestamp_ms": 1780711201123,
                    },
                ],
                "plain_text": "USER: 不應使用 fallback",
            }

            with (
                patch.object(server_realtime, "DATABASE_PATH", database_path),
                patch.object(server_realtime, "UPLOAD_ROOT", upload_root),
                patch.object(server_realtime, "is_hindsight_enabled", return_value=True),
                patch.object(
                    server_realtime,
                    "post_hindsight_json",
                    return_value={"operation_id": "op-123"},
                ) as post_hindsight,
            ):
                response = server_realtime.app.test_client().post(
                    "/api/realtime-session",
                    data={
                        "session_hash": "person-hash",
                        "metadata_file": json_upload(metadata, "metadata.json"),
                        "transcript_file": json_upload(transcript, "transcript.json"),
                        "transcript_text_file": (
                            io.BytesIO(b"USER: fallback transcript"),
                            "transcript.txt",
                        ),
                        "user_audio_file": (
                            io.BytesIO(b"SECRET-WAV-BYTES"),
                            "user_audio.wav",
                        ),
                        "video_frames_file": (
                            io.BytesIO(b"SECRET-VIDEO-BYTES"),
                            "video_frames.zip",
                        ),
                    },
                    content_type="multipart/form-data",
                )

            payload = response.get_json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(payload["memory_update"]["status"], "queued")
            self.assertEqual(payload["memory_update"]["operation_id"], "op-123")
            run_id = payload["run_id"]

            path, request_payload = post_hindsight.call_args.args[:2]
            self.assertEqual(
                path,
                "/v1/default/banks/person-hash/memories",
            )
            self.assertTrue(request_payload["async"])
            item = request_payload["items"][0]
            self.assertEqual(item["document_id"], f"realtime-session:{run_id}")
            self.assertIn("USER: 我今天很累", item["content"])
            self.assertIn("ASSISTANT: 聽起來今天真的撐得很辛苦。", item["content"])
            self.assertNotIn("SECRET-WAV-BYTES", item["content"])
            self.assertNotIn("SECRET-VIDEO-BYTES", item["content"])
            self.assertNotIn("user_audio.wav", json.dumps(request_payload))
            self.assertNotIn("video_frames.zip", json.dumps(request_payload))

            run = get_realtime_session_run(database_path, "person-hash", run_id)
            self.assertEqual(run["memory_status"], "queued")
            self.assertEqual(run["memory_operation_id"], "op-123")
            self.assertTrue(run["memory_queued_at"])

    def test_disabled_hindsight_does_not_call_service(self):
        with (
            patch.object(server_realtime, "is_hindsight_enabled", return_value=False),
            patch.object(server_realtime, "post_hindsight_json") as post_hindsight,
        ):
            result = server_realtime.retain_realtime_session_memory(
                "person-hash",
                "run-id",
                "2026-06-06T10:00:00Z",
                {"events": [{"speaker": "user", "text": "test"}]},
            )

        self.assertEqual(result["status"], "disabled")
        post_hindsight.assert_not_called()

    def test_retain_failure_does_not_fail_session_upload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "app.sqlite3"
            upload_root = root / "realtime"
            initialize_database(database_path)
            metadata = {
                "session_hash": "person-hash",
                "started_at_iso": "2026-06-06T10:00:00.123Z",
            }
            transcript = {
                "events": [{"speaker": "user", "text": "測試對話"}],
            }

            with (
                patch.object(server_realtime, "DATABASE_PATH", database_path),
                patch.object(server_realtime, "UPLOAD_ROOT", upload_root),
                patch.object(server_realtime, "is_hindsight_enabled", return_value=True),
                patch.object(
                    server_realtime,
                    "post_hindsight_json",
                    side_effect=RuntimeError("retain unavailable"),
                ),
            ):
                response = server_realtime.app.test_client().post(
                    "/api/realtime-session",
                    data={
                        "session_hash": "person-hash",
                        "metadata_file": json_upload(metadata, "metadata.json"),
                        "transcript_file": json_upload(transcript, "transcript.json"),
                    },
                    content_type="multipart/form-data",
                )

            payload = response.get_json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["memory_update"]["status"], "error")
            self.assertIn("retain unavailable", payload["memory_update"]["message"])
            run = get_realtime_session_run(
                database_path,
                "person-hash",
                payload["run_id"],
            )
            self.assertEqual(run["memory_status"], "error")
            self.assertIn("retain unavailable", run["memory_error"])


if __name__ == "__main__":
    unittest.main()
