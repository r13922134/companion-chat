import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import server_feedback


class FakeVectorStoreFiles:
    def __init__(self):
        self.calls = []

    def upload_and_poll(self, **kwargs):
        uploaded_file = kwargs["file"]
        self.calls.append(
            {
                "vector_store_id": kwargs["vector_store_id"],
                "filename": uploaded_file.name,
                "content": uploaded_file.read().decode("utf-8"),
                "attributes": kwargs["attributes"],
            }
        )
        return SimpleNamespace(id="file_feedback_123", status="completed", last_error=None)


class FeedbackVectorStoreTests(unittest.TestCase):
    def test_attach_previous_user_utterance_uses_assistant_item_position(self):
        records = [
            {
                "item_id": "assistant-2",
                "response_text": "第二個回答",
                "feedback_text": "修正後答案",
            }
        ]
        transcript = {
            "events": [
                {"speaker": "user", "item_id": "user-1", "text": "第一個問題"},
                {"speaker": "assistant", "item_id": "assistant-1", "text": "第一個回答"},
                {"speaker": "user", "item_id": "user-2", "text": "第二個問題"},
                {"speaker": "assistant", "item_id": "assistant-2", "text": "第二個回答"},
                {"speaker": "user", "item_id": "user-3", "text": "後來才說的話"},
            ]
        }

        server_feedback.attach_previous_user_utterances(records, transcript)

        self.assertEqual(records[0]["user_utterance"], "第二個問題")

    def test_upload_feedback_qa_pair_uses_existing_vector_store(self):
        fake_files = FakeVectorStoreFiles()
        fake_client = SimpleNamespace(
            vector_stores=SimpleNamespace(files=fake_files),
        )
        record = {
            "session_hash": "session-123",
            "item_id": "assistant-123",
            "user_utterance": "我最近吃不下怎麼辦？",
            "feedback_text": "可以先少量多餐，並告訴照護團隊。",
        }

        with (
            patch.object(
                server_feedback,
                "get_medical_qa_vector_store_id",
                return_value="vs_existing",
            ),
            patch.object(server_feedback, "initialize_openai_client"),
            patch.object(server_feedback, "openai_client", fake_client),
        ):
            result = server_feedback.upload_feedback_qa_pair(record)

        self.assertEqual(result["status"], "uploaded")
        self.assertEqual(result["vector_store_id"], "vs_existing")
        self.assertEqual(result["file_id"], "file_feedback_123")
        self.assertEqual(len(fake_files.calls), 1)
        self.assertEqual(fake_files.calls[0]["vector_store_id"], "vs_existing")
        self.assertIn(
            "### QA FEEDBACK_session-123_assistant-123",
            fake_files.calls[0]["content"],
        )
        self.assertIn("問題：我最近吃不下怎麼辦？", fake_files.calls[0]["content"])
        self.assertIn("答覆：可以先少量多餐，並告訴照護團隊。", fake_files.calls[0]["content"])

    def test_session_upload_derives_question_and_records_vector_update(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "app.sqlite3"
            upload_root = root / "uploads"
            server_feedback.initialize_database(database_path)
            captured_records = []

            def fake_upload(records):
                captured_records.extend(records)
                results = []
                for record in records:
                    result = {
                        "status": "uploaded",
                        "vector_store_id": "vs_existing",
                        "file_id": "file_feedback_456",
                    }
                    record["vector_store_update"] = result
                    results.append(result)
                return results

            transcript = {
                "events": [
                    {"speaker": "user", "item_id": "user-1", "text": "治療後很累怎麼辦？"},
                    {"speaker": "assistant", "item_id": "assistant-1", "text": "原本回答"},
                ]
            }
            feedback = {
                "feedback_records": [
                    {
                        "item_id": "assistant-1",
                        "response_text": "原本回答",
                        "feedback_text": "先確認疲累程度，再給簡短建議。",
                    }
                ]
            }

            with (
                patch.object(server_feedback, "DATABASE_PATH", database_path),
                patch.object(server_feedback, "UPLOAD_ROOT", upload_root),
                patch.object(server_feedback, "upload_feedback_qa_records", side_effect=fake_upload),
            ):
                client = server_feedback.app.test_client()
                response = client.post(
                    "/api/realtime-session",
                    data={
                        "session_hash": "session-archive",
                        "metadata_file": (
                            io.BytesIO(json.dumps({}).encode("utf-8")),
                            "metadata.json",
                        ),
                        "transcript_file": (
                            io.BytesIO(json.dumps(transcript, ensure_ascii=False).encode("utf-8")),
                            "transcript.json",
                        ),
                        "feedback_file": (
                            io.BytesIO(json.dumps(feedback, ensure_ascii=False).encode("utf-8")),
                            "feedback.json",
                        ),
                    },
                    content_type="multipart/form-data",
                )

            payload = response.get_json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(payload["vector_store_uploaded_count"], 1)
            self.assertEqual(captured_records[0]["user_utterance"], "治療後很累怎麼辦？")

            saved_feedback_path = Path(
                next(
                    item["path"]
                    for item in payload["saved_file_paths"]
                    if item["field_name"] == "feedback_file"
                )
            )
            saved_feedback = json.loads(saved_feedback_path.read_text(encoding="utf-8"))
            saved_record = saved_feedback["feedback_records"][0]
            self.assertEqual(saved_record["user_utterance"], "治療後很累怎麼辦？")
            self.assertEqual(saved_record["vector_store_update"]["status"], "uploaded")


if __name__ == "__main__":
    unittest.main()
