from __future__ import annotations

import ast
import json
import sqlite3
import wave
from pathlib import Path

import pytest

from app.depression_detector import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DepressionDetectionError,
    PARTICIPANT_TRANSCRIPT_FILENAME,
    PARTICIPANT_UTTERANCES_FILENAME,
    RealtimeDepressionDetector,
    RealtimeRunInput,
    VENDOR_ROOT,
    _ensure_vendor_import_path,
    _load_checkpoint_or_fail,
    build_modality_quality_flags,
    build_user_only_dialogue,
    effective_realtime_participant_interval_rows,
    missing_modality_feature,
    phq8_ground_truth,
    queue_realtime_depression_detection,
    realtime_participant_interval_rows,
    write_participant_interval_transcript,
    write_participant_utterances_jsonl,
)
from app.depression_aspect_retrieval import (
    AspectRetrievalConfig,
    LoadedLlmHiddenBackend,
    TranscriptUtterance,
    build_aspect_retrieval_records,
)
from app.depression_preprocessing import (
    PREPROCESSING_FILENAME,
    prepare_depression_translation_artifacts,
)
from app.realtime_analytics import build_realtime_analytics_snapshot
from app.session_artifacts import artifact_relative_path
from app.storage import (
    attach_latest_google_form_response_to_metadata,
    claim_next_depression_job,
    depression_queue_counts,
    enqueue_depression_job,
    finish_depression_job,
    get_depression_job,
    get_realtime_session_run,
    google_form_account_hash,
    heartbeat_depression_job,
    initialize_database,
    insert_google_form_response,
    list_google_form_response_summaries,
    update_depression_worker,
    update_realtime_session_run_depression,
    upsert_realtime_session_run,
)


def test_build_user_only_dialogue_ignores_assistant() -> None:
    transcript = {
        "events": [
            {"speaker": "assistant", "text": "你好"},
            {"speaker": "user", "text": "  我最近睡不好  "},
            {"speaker": "assistant", "text": "聽起來很累"},
            {"speaker": "user", "text": "也比較沒有精神"},
        ]
    }

    assert build_user_only_dialogue(transcript) == (
        "Utterance 1: 我最近睡不好\n"
        "Utterance 2: 也比較沒有精神"
    )


def test_default_detector_paths_are_project_local() -> None:
    project_root = Path(__file__).resolve().parents[1]

    assert DEFAULT_CONFIG_PATH == project_root / "configs" / "qwen3_4b_hubert_resnet_stage2.yaml"
    assert DEFAULT_CHECKPOINT_PATH == (
        project_root / "checkpoints" / "dynrag_filtered_epoch_10" / "epoch_10"
    )
    assert VENDOR_ROOT == project_root / "vendor"


class KeywordBackend:
    name = "keyword_l2"
    model_name = "test-keyword"

    keywords = [
        ("interest", "painting"),
        ("mood", "sad"),
        ("sleep", "sleep"),
        ("energy", "tired"),
        ("appetite", "appetite"),
        ("failure", "guilty"),
        ("focus", "focus"),
        ("psychomotor", "slow"),
    ]

    def generate_profile(self, dialogue: str) -> str:
        return "profile"

    def generate_query(
        self,
        *,
        profile: str,
        aspect: str,
        aspect_description: str,
        query_prompt_version: str = "aspect_evidence_v3",
    ) -> str:
        return f"{aspect} {aspect_description}"

    def embed_texts(self, texts: list[str]):
        import numpy as np

        vectors = np.zeros((len(texts), len(self.keywords)), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = text.lower()
            for col, terms in enumerate(self.keywords):
                if any(term in lowered for term in terms):
                    vectors[row, col] = 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def token_count(self, text: str) -> int:
        return max(1, len(text.split()))


def test_aspect_retrieval_builds_branch_specific_dialogues() -> None:
    utterances = [
        TranscriptUtterance(1, 0.0, 1.0, "I stopped painting."),
        TranscriptUtterance(2, 1.0, 2.0, "I cannot sleep at night."),
        TranscriptUtterance(3, 2.0, 3.0, "I feel tired most days."),
    ]
    output = build_aspect_retrieval_records(
        utterances=utterances,
        backend=KeywordBackend(),
        config=AspectRetrievalConfig(
            min_utterances=1,
            max_utterances=1,
            max_dialogue_tokens=20,
            min_score=-999,
            relative_score_margin=999,
            candidate_filter="none",
        ),
    )

    by_aspect = {row["aspect_index"]: row for row in output.records}

    assert by_aspect[2]["dialogue"] == "Utterance 2: I cannot sleep at night."
    assert by_aspect[3]["dialogue"] == "Utterance 3: I feel tired most days."
    assert by_aspect[2]["dialogue"] != by_aspect[3]["dialogue"]
    assert by_aspect[2]["aspect_description"] == (
        "Trouble falling or staying asleep, or sleeping too much"
    )


def test_realtime_participant_interval_csv_uses_user_speech_times(tmp_path: Path) -> None:
    transcript = {
        "events": [
            {"speaker": "assistant", "text": "prompt", "audio_start_seconds": 0.0, "audio_end_seconds": 1.0},
            {
                "speaker": "user",
                "text": "I cannot sleep",
                "audio_start_seconds": 1.25,
                "audio_end_seconds": 2.75,
            },
            {"speaker": "user", "text": "missing interval"},
        ]
    }

    rows = realtime_participant_interval_rows(transcript)
    path = write_participant_interval_transcript(
        transcript,
        tmp_path / PARTICIPANT_TRANSCRIPT_FILENAME,
    )

    assert rows == [(1.25, 2.75, "I cannot sleep")]
    assert path is not None
    assert path.read_text(encoding="utf-8").splitlines() == [
        "utterance_index,Start_Time,End_Time,Text",
        "1,1.250,2.750,I cannot sleep",
    ]


def test_participant_utterance_artifact_is_structured(tmp_path: Path) -> None:
    path = write_participant_utterances_jsonl(
        tmp_path / PARTICIPANT_UTTERANCES_FILENAME,
        [
            TranscriptUtterance(1, 1.25, 2.75, "I cannot sleep"),
            TranscriptUtterance(2, 3.0, 4.0, "I feel tired"),
        ],
    )

    assert path is not None
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {
            "utterance_index": 1,
            "speaker": "participant",
            "source_speaker": "user",
            "Start_Time": 1.25,
            "End_Time": 2.75,
            "Text": "I cannot sleep",
        },
        {
            "utterance_index": 2,
            "speaker": "participant",
            "source_speaker": "user",
            "Start_Time": 3.0,
            "End_Time": 4.0,
            "Text": "I feel tired",
        },
    ]


def test_effective_participant_intervals_filter_to_wav_duration(tmp_path: Path) -> None:
    audio_path = tmp_path / "user_audio.wav"
    sample_rate = 16000
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * sample_rate)

    transcript = {
        "events": [
            {
                "speaker": "user",
                "text": "valid interval",
                "audio_start_seconds": 0.1,
                "audio_end_seconds": 0.4,
            },
            {
                "speaker": "user",
                "text": "too short",
                "audio_start_seconds": 0.5,
                "audio_end_seconds": 0.55,
            },
            {
                "speaker": "user",
                "text": "outside audio",
                "audio_start_seconds": 2.0,
                "audio_end_seconds": 3.0,
            },
        ]
    }

    assert effective_realtime_participant_interval_rows(transcript, audio_path) == [
        (0.1, 0.4, "valid interval")
    ]


def test_missing_audio_and_video_are_valid_zero_fill_inputs(tmp_path: Path) -> None:
    run_input = RealtimeRunInput(
        session_hash="abc123",
        run_id="run-1",
        session_dir=tmp_path,
        transcript={"events": [{"speaker": "user", "text": "I feel tired."}]},
        transcript_source="transcript.json",
        preprocessing={},
        metadata={},
        user_audio_path=tmp_path / "missing-user-audio.wav",
        video_frames_path=tmp_path / "missing-video-frames.zip",
    )

    RealtimeDepressionDetector()._validate_formal_realtime_inputs(run_input)

    torch = pytest.importorskip("torch")
    audio = missing_modality_feature(768)
    video = missing_modality_feature(2048)
    assert tuple(audio.shape) == (1, 768)
    assert tuple(video.shape) == (1, 2048)
    assert int(torch.count_nonzero(audio).item()) == 0
    assert int(torch.count_nonzero(video).item()) == 0


def test_modality_quality_flags_mark_incomplete_predictions() -> None:
    flags = build_modality_quality_flags(
        audio_zero_fill_reason="missing_user_audio_file",
        video_zero_fill_reason=None,
        audio_frame_count=0,
        audio_frame_count_before_sampling=0,
        audio_downsampled=False,
        video_frame_count=8192,
        video_frame_count_before_sampling=9000,
        video_downsampled=True,
        participant_speech_seconds=310.5,
        speech_interval_count=85,
        text_utterance_count=120,
        candidate_text_utterance_count=44,
        video_capture_frame_count=9000,
    )

    assert flags["audio_missing"] is True
    assert flags["video_missing"] is False
    assert flags["audio_frame_count"] == 0
    assert flags["video_frame_count"] == 8192
    assert flags["video_downsampled"] is True
    assert flags["participant_speech_seconds"] == 310.5
    assert flags["speech_interval_count"] == 85
    assert flags["text_utterance_count"] == 120
    assert flags["candidate_text_utterance_count"] == 44
    assert flags["incomplete_modalities"] == ["audio"]
    assert flags["prediction_is_based_on_incomplete_modalities"] is True
    assert (
        flags["incomplete_modalities_message"]
        == "prediction is based on incomplete modalities"
    )


def test_hubert_extractor_is_reused_across_realtime_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_vendor_import_path()
    import mllm_dr.data

    created = []

    class FakeFeatureExtractor:
        def __init__(self, **kwargs) -> None:
            self.cache_dir = Path(kwargs["cache_dir"])
            created.append(self)

    monkeypatch.setattr(mllm_dr.data, "FeatureExtractor", FakeFeatureExtractor)
    detector = RealtimeDepressionDetector()
    features_cfg = {
        "hubert_model_name": "test-hubert",
        "hubert_chunk_seconds": 30.0,
        "max_interval_seconds": 30.0,
    }

    first = detector._get_audio_feature_extractor(
        cache_dir=tmp_path / "run-1",
        features_cfg=features_cfg,
        device="cpu",
    )
    second = detector._get_audio_feature_extractor(
        cache_dir=tmp_path / "run-2",
        features_cfg=features_cfg,
        device="cpu",
    )

    assert first is second
    assert len(created) == 1
    assert second.cache_dir == tmp_path / "run-2"


def test_shared_retrieval_disables_lora_adapter() -> None:
    events = []

    class FakeTokenizer:
        pad_token = "pad"
        eos_token = "eos"

    class AdapterContext:
        def __enter__(self):
            events.append("disabled")

        def __exit__(self, exc_type, exc, traceback):
            events.append("restored")

    class FakeLlm:
        peft_config = {"default": object()}

        def disable_adapter(self):
            return AdapterContext()

    backend = LoadedLlmHiddenBackend(
        tokenizer=FakeTokenizer(),
        llm=FakeLlm(),
        model_name="test-qwen",
    )

    with backend._base_model_context():
        events.append("retrieval")

    assert events == ["disabled", "retrieval", "restored"]


def test_missing_user_text_still_rejects_prediction(tmp_path: Path) -> None:
    run_input = RealtimeRunInput(
        session_hash="abc123",
        run_id="run-1",
        session_dir=tmp_path,
        transcript={"events": [{"speaker": "assistant", "text": "How are you?"}]},
        transcript_source="transcript.json",
        preprocessing={},
        metadata={},
        user_audio_path=tmp_path / "missing-user-audio.wav",
        video_frames_path=tmp_path / "missing-video-frames.zip",
    )

    with pytest.raises(DepressionDetectionError, match="No user transcript"):
        RealtimeDepressionDetector()._validate_formal_realtime_inputs(run_input)


def test_phq8_ground_truth_maps_form_items_to_aspects() -> None:
    metadata = {
        "selected_user": {
            "phq8": {
                "total_score": 8,
                "item_scores": {
                    str(index): {"score": score}
                    for index, score in enumerate([2, 2, 2, 2, 0, 0, 0, 0], start=1)
                },
            }
        }
    }

    assert phq8_ground_truth(metadata) == {
        "scale": "PHQ-8",
        "aspect_scores": [2, 2, 2, 2, 0, 0, 0, 0],
        "total_score": 8,
        "binary_depression": False,
        "complete": True,
    }


def test_english_translation_preprocessing_runs_in_worker_without_network(
    tmp_path: Path,
) -> None:
    preprocessing, files = prepare_depression_translation_artifacts(
        tmp_path,
        {
            "events": [
                {"speaker": "user", "text": "I have felt tired recently."},
            ]
        },
    )

    assert preprocessing["translation"]["status"] == "skipped_already_english"
    expected_path = tmp_path / artifact_relative_path(PREPROCESSING_FILENAME)
    assert expected_path.is_file()
    assert [row["filename"] for row in files] == [PREPROCESSING_FILENAME]
    assert [row["relative_path"] for row in files] == [
        artifact_relative_path(PREPROCESSING_FILENAME).as_posix()
    ]


def test_feedback_server_does_not_queue_depression_prediction() -> None:
    source_path = Path(__file__).resolve().parents[1] / "app" / "server_feedback.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }

    assert "queue_realtime_depression_detection" not in referenced_names
    assert "run_depression_detection_job" not in referenced_names


def test_depression_columns_update_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "app.sqlite3"
    initialize_database(db_path)
    upsert_realtime_session_run(
        db_path,
        {
            "session_hash": "abc123",
            "run_id": "20260609_010203_000",
            "session_dir_name": "abc123/20260609_010203_000",
            "started_at": "2026-06-09T01:02:03+08:00",
            "uploaded_at": "2026-06-09T01:03:03+08:00",
            "archive_manifest": {"schema_version": "companion-thesis-realtime-v1"},
            "ground_truth_total_score": 8,
            "ground_truth_binary": False,
        },
    )

    update_realtime_session_run_depression(
        db_path,
        "abc123",
        "20260609_010203_000",
        {
            "status": "ok",
            "total_score": 7.5,
            "binary": False,
            "result": {"status": "ok", "total_score": 7.5},
            "completed_at": "2026-06-09T01:04:03+08:00",
        },
    )

    row = get_realtime_session_run(db_path, "abc123", "20260609_010203_000")

    assert row is not None
    assert row["depression_status"] == "ok"
    assert row["depression_total_score"] == 7.5
    assert row["depression_binary"] is False
    assert row["depression_result"] == {"status": "ok", "total_score": 7.5}
    assert row["depression_error"] is None
    assert row["depression_completed_at"] == "2026-06-09T01:04:03+08:00"
    assert row["archive_manifest"] == {
        "account_id": "abc123",
        "session_hash": "abc123",
        "schema_version": "companion-thesis-realtime-v1"
    }
    assert row["ground_truth_total_score"] == 8
    assert row["ground_truth_binary"] is False


def test_realtime_analytics_snapshot_includes_phq_and_predictions(tmp_path: Path) -> None:
    db_path = tmp_path / "app.sqlite3"
    initialize_database(db_path)
    email = "test@example.com"
    account_hash = google_form_account_hash(email)
    phq8 = {
        "scale": "PHQ-8",
        "total_score": 6,
        "answered_count": 8,
        "max_score": 24,
        "item_scores": {
            "1": {"question": "1. interest", "answer": "好幾天", "score": 1},
            "2": {"question": "2. mood", "answer": "超過一半以上的天數", "score": 2},
            "3": {"question": "3. sleep", "answer": "完全沒有", "score": 0},
            "4": {"question": "4. energy", "answer": "幾乎每天", "score": 3},
            "5": {"question": "5. appetite", "answer": "完全沒有", "score": 0},
            "6": {"question": "6. failure", "answer": "完全沒有", "score": 0},
            "7": {"question": "7. focus", "answer": "完全沒有", "score": 0},
            "8": {"question": "8. movement", "answer": "完全沒有", "score": 0},
        },
    }
    insert_google_form_response(
        db_path,
        {
            "form_hash": "form123",
            "form_dir_name": "20260609_form123",
            "form_title": "PHQ-8",
            "respondent_email": email,
            "submitted_at": "2026-06-09T01:00:00+08:00",
            "received_at": "2026-06-09T01:00:10+08:00",
            "name": "Test User",
            "age": "35",
            "date_prefix": "2026-06-09",
            "fields": {},
            "phq8": phq8,
        },
    )
    upsert_realtime_session_run(
        db_path,
        {
            "session_hash": account_hash,
            "run_id": "20260609_010203_000",
            "session_dir_name": f"{account_hash}/20260609_010203_000",
            "started_at": "2026-06-09T01:02:03+08:00",
            "uploaded_at": "2026-06-09T01:03:03+08:00",
            "metadata": {
                "selected_user": {
                    "account_hash": account_hash,
                    "session_hash": account_hash,
                    "respondent_email": email,
                    "form_hash": "form123",
                    "name": "Test User",
                    "age": "35",
                    "phq8": phq8,
                }
            },
            "ground_truth_total_score": 6,
            "ground_truth_binary": False,
        },
    )
    update_realtime_session_run_depression(
        db_path,
        account_hash,
        "20260609_010203_000",
        {
            "status": "ok",
            "total_score": 8.25,
            "binary": False,
            "result": {
                "status": "ok",
                "total_score": 8.25,
                "aspects": [
                    {
                        "aspect_index": 0,
                        "clinical_description": "Little interest",
                        "prediction": 1.5,
                        "ground_truth": 1,
                        "prediction_source": "test",
                    }
                ],
            },
            "completed_at": "2026-06-09T01:04:03+08:00",
        },
    )

    snapshot = build_realtime_analytics_snapshot(db_path)

    assert snapshot["summary"]["user_count"] == 1
    assert snapshot["summary"]["run_count"] == 1
    assert snapshot["summary"]["mean_phq8_score"] == 6
    assert snapshot["summary"]["mean_prediction_score"] == 8.25
    assert snapshot["summary"]["mean_abs_error"] == 2.25
    assert snapshot["users"][0]["phq8"]["items"][0] == {
        "index": 1,
        "question": "1. interest",
        "answer": "好幾天",
        "score": 1,
    }
    assert snapshot["users"][0]["account_id"] == account_hash
    assert "respondent_email" not in snapshot["users"][0]
    assert "respondent_email" not in snapshot["runs"][0]
    assert snapshot["runs"][0]["prediction"] == 8.25
    assert snapshot["aspect_summary"][0]["mean_abs_error"] == 0.5


def test_google_form_responses_group_same_email_as_one_account(tmp_path: Path) -> None:
    db_path = tmp_path / "app.sqlite3"
    initialize_database(db_path)
    email = "Student@Example.COM"
    account_hash = google_form_account_hash(email)

    older_phq8 = {
        "scale": "PHQ-8",
        "total_score": 4,
        "answered_count": 8,
        "max_score": 24,
        "item_scores": {},
    }
    latest_phq8 = {
        "scale": "PHQ-8",
        "total_score": 12,
        "answered_count": 8,
        "max_score": 24,
        "item_scores": {},
    }
    insert_google_form_response(
        db_path,
        {
            "form_hash": "older_form",
            "form_dir_name": "20260608_older_form",
            "respondent_email": email,
            "submitted_at": "2026-06-08T10:00:00+08:00",
            "received_at": "2026-06-08T10:00:05+08:00",
            "name": "Student",
            "age": "20",
            "date_prefix": "20260608",
            "fields": {},
            "phq8": older_phq8,
        },
    )
    insert_google_form_response(
        db_path,
        {
            "form_hash": "latest_form",
            "form_dir_name": "20260610_latest_form",
            "respondent_email": email,
            "submitted_at": "2026-06-10T10:00:00+08:00",
            "received_at": "2026-06-10T10:00:05+08:00",
            "name": "Student",
            "age": "21",
            "date_prefix": "20260610",
            "fields": {},
            "phq8": latest_phq8,
        },
    )

    users = list_google_form_response_summaries(db_path)

    assert len(users) == 1
    assert users[0]["account_id"] == account_hash
    assert users[0]["account_hash"] == account_hash
    assert users[0]["session_hash"] == account_hash
    assert "respondent_email" not in users[0]
    assert users[0]["form_hash"] == "latest_form"
    assert users[0]["form_count"] == 2
    assert users[0]["age"] == "21"
    assert users[0]["phq8_score"] == 12


def test_google_form_response_requires_email(tmp_path: Path) -> None:
    db_path = tmp_path / "app.sqlite3"
    initialize_database(db_path)

    with pytest.raises(ValueError, match="respondent_email is required"):
        insert_google_form_response(
            db_path,
            {
                "form_hash": "form_without_email",
                "form_dir_name": "20260610_form_without_email",
                "submitted_at": "2026-06-10T10:00:00+08:00",
                "received_at": "2026-06-10T10:00:05+08:00",
                "date_prefix": "20260610",
                "fields": {},
                "phq8": {},
            },
        )


def test_initialize_database_deletes_legacy_google_forms_without_email(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE google_form_responses (
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
            )
            """
        )
        connection.execute(
            """
            INSERT INTO google_form_responses (
                form_hash,
                form_dir_name,
                submitted_at,
                received_at,
                date_prefix,
                fields_json,
                phq8_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy_form",
                "20260608_legacy_form",
                "2026-06-08T10:00:00+08:00",
                "2026-06-08T10:00:05+08:00",
                "20260608",
                "{}",
                "{}",
            ),
        )

    initialize_database(db_path)

    with sqlite3.connect(db_path) as connection:
        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM google_form_responses
            """
        ).fetchone()[0]

    assert count == 0


def test_initialize_database_rekeys_stale_account_hashes_by_email(tmp_path: Path) -> None:
    db_path = tmp_path / "stale.sqlite3"
    initialize_database(db_path)
    email = "student@example.com"
    expected_account_hash = google_form_account_hash(email)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO google_form_responses (
                form_hash,
                form_dir_name,
                respondent_email,
                account_hash,
                submitted_at,
                received_at,
                date_prefix,
                fields_json,
                phq8_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "older_form",
                "20260608_older_form",
                email,
                "older_form",
                "2026-06-08T10:00:00+08:00",
                "2026-06-08T10:00:05+08:00",
                "20260608",
                "{}",
                '{"total_score":4}',
            ),
        )
        connection.execute(
            """
            INSERT INTO google_form_responses (
                form_hash,
                form_dir_name,
                respondent_email,
                account_hash,
                submitted_at,
                received_at,
                date_prefix,
                fields_json,
                phq8_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "latest_form",
                "20260610_latest_form",
                email,
                expected_account_hash,
                "2026-06-10T10:00:00+08:00",
                "2026-06-10T10:00:05+08:00",
                "20260610",
                "{}",
                '{"total_score":12}',
            ),
        )

    initialize_database(db_path)
    users = list_google_form_response_summaries(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT account_id, account_hash
            FROM google_form_responses
            ORDER BY form_hash
            """
        ).fetchall()

    assert len(users) == 1
    assert users[0]["account_id"] == expected_account_hash
    assert users[0]["account_hash"] == expected_account_hash
    assert users[0]["form_hash"] == "latest_form"
    assert users[0]["form_count"] == 2
    assert {row["account_id"] for row in rows} == {expected_account_hash}
    assert {row["account_hash"] for row in rows} == {expected_account_hash}


def test_initialize_database_migrates_account_id_and_redacts_identity_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy_identity.sqlite3"
    initialize_database(db_path)
    metadata = {
        "respondent_email": "student@example.com",
        "selected_user": {
            "respondent_email": "student@example.com",
            "account_hash": "legacy123",
            "session_hash": "legacy123",
            "name": "Student",
        },
    }
    archive_manifest = {
        "schema_version": "companion-thesis-realtime-v1",
        "session_hash": "legacy123",
    }
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO realtime_sessions (
                session_hash,
                session_dir_name,
                created_at,
                updated_at,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "legacy123",
                "legacy123",
                "2026-06-08T10:00:00+08:00",
                "2026-06-08T10:00:00+08:00",
                json.dumps(metadata),
            ),
        )
        connection.execute(
            """
            INSERT INTO realtime_session_runs (
                session_hash,
                run_id,
                session_dir_name,
                started_at,
                uploaded_at,
                metadata_json,
                archive_manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy123",
                "run-1",
                "legacy123/run-1",
                "2026-06-08T10:00:00+08:00",
                "2026-06-08T10:01:00+08:00",
                json.dumps(metadata),
                json.dumps(archive_manifest),
            ),
        )

    initialize_database(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        session_row = connection.execute(
            """
            SELECT account_id, metadata_json
            FROM realtime_sessions
            WHERE session_hash = ?
            """,
            ("legacy123",),
        ).fetchone()
        run_row = connection.execute(
            """
            SELECT account_id, metadata_json, archive_manifest_json
            FROM realtime_session_runs
            WHERE session_hash = ? AND run_id = ?
            """,
            ("legacy123", "run-1"),
        ).fetchone()

    assert session_row["account_id"] == "legacy123"
    assert run_row["account_id"] == "legacy123"
    session_metadata = json.loads(session_row["metadata_json"])
    run_metadata = json.loads(run_row["metadata_json"])
    run_manifest = json.loads(run_row["archive_manifest_json"])
    assert "respondent_email" not in session_metadata
    assert "respondent_email" not in session_metadata["selected_user"]
    assert session_metadata["selected_user"]["account_id"] == "legacy123"
    assert "respondent_email" not in run_metadata
    assert "respondent_email" not in run_metadata["selected_user"]
    assert run_metadata["selected_user"]["account_id"] == "legacy123"
    assert run_manifest["account_id"] == "legacy123"
    assert run_manifest["session_hash"] == "legacy123"


def test_latest_form_phq8_is_used_for_account_ground_truth(tmp_path: Path) -> None:
    db_path = tmp_path / "app.sqlite3"
    initialize_database(db_path)
    email = "student@example.com"
    account_hash = google_form_account_hash(email)
    older_phq8 = {
        "scale": "PHQ-8",
        "total_score": 4,
        "answered_count": 8,
        "max_score": 24,
        "item_scores": {
            "1": {"question": "1. interest", "answer": "完全沒有", "score": 0},
        },
    }
    latest_phq8 = {
        "scale": "PHQ-8",
        "total_score": 12,
        "answered_count": 8,
        "max_score": 24,
        "item_scores": {
            "1": {"question": "1. interest", "answer": "幾乎每天", "score": 3},
        },
    }
    insert_google_form_response(
        db_path,
        {
            "form_hash": "older_form",
            "form_dir_name": "20260608_older_form",
            "respondent_email": email,
            "submitted_at": "2026-06-08T10:00:00+08:00",
            "received_at": "2026-06-08T10:00:05+08:00",
            "name": "Student",
            "age": "20",
            "date_prefix": "20260608",
            "fields": {},
            "phq8": older_phq8,
        },
    )
    insert_google_form_response(
        db_path,
        {
            "form_hash": "latest_form",
            "form_dir_name": "20260610_latest_form",
            "respondent_email": email,
            "submitted_at": "2026-06-10T10:00:00+08:00",
            "received_at": "2026-06-10T10:00:05+08:00",
            "name": "Student",
            "age": "21",
            "date_prefix": "20260610",
            "fields": {},
            "phq8": latest_phq8,
        },
    )

    metadata, changed = attach_latest_google_form_response_to_metadata(
        db_path,
        session_hash=account_hash,
        metadata={
            "selected_user": {
                "respondent_email": email,
                "form_hash": "older_form",
                "phq8": older_phq8,
            }
        },
    )

    assert changed is True
    assert metadata["selected_user"]["form_hash"] == "latest_form"
    assert metadata["selected_user"]["account_id"] == account_hash
    assert metadata["selected_user"]["session_hash"] == account_hash
    assert "respondent_email" not in metadata
    assert "respondent_email" not in metadata["selected_user"]
    assert metadata["selected_user"]["phq8"]["total_score"] == 12

    upsert_realtime_session_run(
        db_path,
        {
            "session_hash": account_hash,
            "run_id": "20260610_101500_000",
            "session_dir_name": f"{account_hash}/20260610_101500_000",
            "started_at": "2026-06-10T10:15:00+08:00",
            "uploaded_at": "2026-06-10T10:16:00+08:00",
            "metadata": metadata,
            "ground_truth_total_score": 12,
            "ground_truth_binary": True,
        },
    )
    update_realtime_session_run_depression(
        db_path,
        account_hash,
        "20260610_101500_000",
        {
            "status": "ok",
            "total_score": 8,
            "binary": False,
            "result": {
                "status": "ok",
                "total_score": 8,
                "aspects": [
                    {
                        "aspect_index": 0,
                        "clinical_description": "Little interest",
                        "prediction": 1,
                        "ground_truth": 3,
                    }
                ],
            },
            "completed_at": "2026-06-10T10:17:00+08:00",
        },
    )

    snapshot = build_realtime_analytics_snapshot(db_path)

    assert snapshot["summary"]["user_count"] == 1
    assert snapshot["users"][0]["form_count"] == 2
    assert snapshot["users"][0]["phq8"]["total_score"] == 12
    assert snapshot["users"][0]["account_id"] == account_hash
    assert "respondent_email" not in snapshot["users"][0]
    assert "respondent_email" not in snapshot["runs"][0]
    assert snapshot["runs"][0]["account_id"] == account_hash
    assert snapshot["runs"][0]["user_key"] == account_hash
    assert snapshot["runs"][0]["form_hash"] == "latest_form"
    assert snapshot["runs"][0]["ground_truth"] == 12
    assert snapshot["runs"][0]["abs_error"] == 4
    assert snapshot["runs"][0]["aspect_predictions"][0]["ground_truth"] == 3
    assert snapshot["runs"][0]["aspect_predictions"][0]["abs_error"] == 2


def test_analytics_keeps_run_time_ground_truth_snapshot_after_new_form(tmp_path: Path) -> None:
    db_path = tmp_path / "app.sqlite3"
    initialize_database(db_path)
    email = "student@example.com"
    account_hash = google_form_account_hash(email)
    older_phq8 = {
        "scale": "PHQ-8",
        "total_score": 4,
        "answered_count": 8,
        "max_score": 24,
        "item_scores": {
            "1": {"question": "1. interest", "answer": "完全沒有", "score": 0},
        },
    }
    latest_phq8 = {
        "scale": "PHQ-8",
        "total_score": 12,
        "answered_count": 8,
        "max_score": 24,
        "item_scores": {
            "1": {"question": "1. interest", "answer": "幾乎每天", "score": 3},
        },
    }
    insert_google_form_response(
        db_path,
        {
            "form_hash": "older_form",
            "form_dir_name": "20260608_older_form",
            "respondent_email": email,
            "submitted_at": "2026-06-08T10:00:00+08:00",
            "received_at": "2026-06-08T10:00:05+08:00",
            "name": "Student",
            "age": "20",
            "date_prefix": "20260608",
            "fields": {},
            "phq8": older_phq8,
        },
    )
    old_metadata, changed = attach_latest_google_form_response_to_metadata(
        db_path,
        session_hash=account_hash,
        metadata={
            "selected_user": {
                "respondent_email": email,
                "form_hash": "older_form",
                "phq8": older_phq8,
            }
        },
    )
    assert changed is True
    assert old_metadata["selected_user"]["form_hash"] == "older_form"
    assert old_metadata["selected_user"]["account_id"] == account_hash
    assert "respondent_email" not in old_metadata
    assert "respondent_email" not in old_metadata["selected_user"]

    upsert_realtime_session_run(
        db_path,
        {
            "session_hash": account_hash,
            "run_id": "20260608_101500_000",
            "session_dir_name": f"{account_hash}/20260608_101500_000",
            "started_at": "2026-06-08T10:15:00+08:00",
            "uploaded_at": "2026-06-08T10:16:00+08:00",
            "metadata": old_metadata,
            "ground_truth_total_score": 4,
            "ground_truth_binary": False,
        },
    )
    update_realtime_session_run_depression(
        db_path,
        account_hash,
        "20260608_101500_000",
        {
            "status": "ok",
            "total_score": 8,
            "binary": False,
            "result": {
                "status": "ok",
                "total_score": 8,
                "aspects": [
                    {
                        "aspect_index": 0,
                        "clinical_description": "Little interest",
                        "prediction": 1,
                        "ground_truth": 0,
                    }
                ],
            },
            "completed_at": "2026-06-08T10:17:00+08:00",
        },
    )
    insert_google_form_response(
        db_path,
        {
            "form_hash": "latest_form",
            "form_dir_name": "20260610_latest_form",
            "respondent_email": email,
            "submitted_at": "2026-06-10T10:00:00+08:00",
            "received_at": "2026-06-10T10:00:05+08:00",
            "name": "Student",
            "age": "21",
            "date_prefix": "20260610",
            "fields": {},
            "phq8": latest_phq8,
        },
    )

    snapshot = build_realtime_analytics_snapshot(db_path)

    assert snapshot["summary"]["user_count"] == 1
    assert snapshot["users"][0]["form_count"] == 2
    assert snapshot["users"][0]["phq8"]["total_score"] == 12
    assert snapshot["users"][0]["account_id"] == account_hash
    assert "respondent_email" not in snapshot["users"][0]
    assert "respondent_email" not in snapshot["runs"][0]
    assert snapshot["runs"][0]["account_id"] == account_hash
    assert snapshot["runs"][0]["user_key"] == account_hash
    assert snapshot["runs"][0]["form_hash"] == "older_form"
    assert snapshot["runs"][0]["latest_form_hash"] == "latest_form"
    assert snapshot["runs"][0]["phq8"]["total_score"] == 4
    assert snapshot["runs"][0]["ground_truth"] == 4
    assert snapshot["runs"][0]["abs_error"] == 4
    assert snapshot["runs"][0]["aspect_predictions"][0]["ground_truth"] == 0
    assert snapshot["runs"][0]["aspect_predictions"][0]["abs_error"] == 1


def test_sqlite_depression_queue_claims_each_job_once(tmp_path: Path) -> None:
    db_path = tmp_path / "app.sqlite3"
    initialize_database(db_path)
    run_id = "20260609_010203_000"
    session_dir = tmp_path / "uploads" / "abc123" / run_id
    session_dir.mkdir(parents=True)
    upsert_realtime_session_run(
        db_path,
        {
            "session_hash": "abc123",
            "run_id": run_id,
            "session_dir_name": f"abc123/{run_id}",
            "started_at": "2026-06-09T01:02:03+08:00",
            "uploaded_at": "2026-06-09T01:03:03+08:00",
        },
    )

    queued = enqueue_depression_job(db_path, "abc123", run_id, session_dir)
    claimed = claim_next_depression_job(
        db_path,
        worker_id="worker-1",
        lease_seconds=60,
    )

    assert queued["status"] == "queued"
    assert claimed is not None
    assert claimed["id"] == queued["id"]
    assert claimed["worker_id"] == "worker-1"
    assert claimed["attempts"] == 1
    assert claim_next_depression_job(
        db_path,
        worker_id="worker-2",
        lease_seconds=60,
    ) is None
    assert heartbeat_depression_job(
        db_path,
        claimed["id"],
        worker_id="worker-1",
        lease_seconds=60,
    )
    assert finish_depression_job(
        db_path,
        claimed["id"],
        worker_id="worker-1",
        status="completed",
    )
    assert get_depression_job(db_path, "abc123", run_id)["status"] == "completed"
    assert depression_queue_counts(db_path) == {"completed": 1}

    row = get_realtime_session_run(db_path, "abc123", run_id)
    assert row is not None
    assert row["depression_status"] == "running"
    assert row["depression_worker_id"] == "worker-1"
    assert row["depression_queued_at"]
    assert row["depression_started_at"]


def _register_ready_depression_worker(db_path: Path) -> None:
    update_depression_worker(
        db_path,
        worker_id="test-worker:gpu-0",
        gpu_id="0",
        pid=1234,
        hostname="test-host",
        status="ready",
        started_at="2026-06-09T01:00:00+08:00",
    )


def test_realtime_queue_function_marks_unavailable_without_worker(tmp_path: Path) -> None:
    db_path = tmp_path / "app.sqlite3"
    initialize_database(db_path)
    run_id = "20260609_010203_000"
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    upsert_realtime_session_run(
        db_path,
        {
            "session_hash": "abc123",
            "run_id": run_id,
            "session_dir_name": f"abc123/{run_id}",
            "started_at": "2026-06-09T01:02:03+08:00",
            "uploaded_at": "2026-06-09T01:03:03+08:00",
        },
    )

    result = queue_realtime_depression_detection(
        db_path,
        "abc123",
        run_id,
        session_dir,
    )

    assert result["status"] == "unavailable"
    assert get_depression_job(db_path, "abc123", run_id) is None
    row = get_realtime_session_run(db_path, "abc123", run_id)
    assert row["depression_status"] == "unavailable"
    assert row["depression_error"] == "GPU depression worker is unavailable."


def test_realtime_queue_function_only_persists_job_with_active_worker(tmp_path: Path) -> None:
    db_path = tmp_path / "app.sqlite3"
    initialize_database(db_path)
    _register_ready_depression_worker(db_path)
    run_id = "20260609_010203_000"
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    upsert_realtime_session_run(
        db_path,
        {
            "session_hash": "abc123",
            "run_id": run_id,
            "session_dir_name": f"abc123/{run_id}",
            "started_at": "2026-06-09T01:02:03+08:00",
            "uploaded_at": "2026-06-09T01:03:03+08:00",
        },
    )

    result = queue_realtime_depression_detection(
        db_path,
        "abc123",
        run_id,
        session_dir,
    )

    assert result["status"] == "queued"
    assert result["job_id"]
    assert get_depression_job(db_path, "abc123", run_id)["status"] == "queued"
    assert get_realtime_session_run(
        db_path,
        "abc123",
        run_id,
    )["depression_status"] == "queued"


def test_expired_depression_job_is_reclaimed(tmp_path: Path) -> None:
    db_path = tmp_path / "app.sqlite3"
    initialize_database(db_path)
    run_id = "20260609_010203_000"
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    upsert_realtime_session_run(
        db_path,
        {
            "session_hash": "abc123",
            "run_id": run_id,
            "session_dir_name": f"abc123/{run_id}",
            "started_at": "2026-06-09T01:02:03+08:00",
            "uploaded_at": "2026-06-09T01:03:03+08:00",
        },
    )
    enqueue_depression_job(db_path, "abc123", run_id, session_dir)
    first = claim_next_depression_job(
        db_path,
        worker_id="worker-1",
        lease_seconds=60,
    )
    assert first is not None

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE depression_jobs SET lease_expires_at_epoch = 0 WHERE id = ?",
            (first["id"],),
        )

    second = claim_next_depression_job(
        db_path,
        worker_id="worker-2",
        lease_seconds=60,
    )

    assert second is not None
    assert second["id"] == first["id"]
    assert second["worker_id"] == "worker-2"
    assert second["attempts"] == 2


def test_stage2_checkpoint_without_lora_or_full_state_errors(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint_dir = tmp_path / "epoch_10"
    checkpoint_dir.mkdir()
    torch.save({"audio_lqformer": None, "video_lqformer": None}, checkpoint_dir / "mllm_dr.pt")

    class DummyModel:
        audio_lqformer = None
        video_lqformer = None

    with pytest.raises(DepressionDetectionError, match="missing llm_adapter"):
        _load_checkpoint_or_fail(DummyModel(), checkpoint_dir)
