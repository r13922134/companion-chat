from __future__ import annotations

import ast
import sqlite3
import wave
from pathlib import Path

import pytest

from app.depression_detector import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    DepressionDetectionError,
    PARTICIPANT_TRANSCRIPT_FILENAME,
    RealtimeDepressionDetector,
    RealtimeRunInput,
    VENDOR_ROOT,
    _ensure_vendor_import_path,
    _load_checkpoint_or_fail,
    build_user_only_dialogue,
    effective_realtime_participant_interval_rows,
    missing_modality_feature,
    phq8_ground_truth,
    queue_realtime_depression_detection,
    realtime_participant_interval_rows,
    write_participant_interval_transcript,
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
from app.storage import (
    claim_next_depression_job,
    depression_queue_counts,
    enqueue_depression_job,
    finish_depression_job,
    get_depression_job,
    get_realtime_session_run,
    heartbeat_depression_job,
    initialize_database,
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
        "Start_Time,End_Time,Text",
        "1.250,2.750,I cannot sleep",
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
    assert (tmp_path / PREPROCESSING_FILENAME).is_file()
    assert [row["filename"] for row in files] == [PREPROCESSING_FILENAME]


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
        "schema_version": "companion-thesis-realtime-v1"
    }
    assert row["ground_truth_total_score"] == 8
    assert row["ground_truth_binary"] is False


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


def test_realtime_queue_function_only_persists_job(tmp_path: Path) -> None:
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
