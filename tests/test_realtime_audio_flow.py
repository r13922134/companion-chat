import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_realtime_pages_keep_barge_in_with_separate_outbound_track() -> None:
    for relative_path in (
        "app/templates/realtime.html",
        "app/templates/feedback_realtime.html",
    ):
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

        assert "sourceAudioTrack.clone()" in source
        assert "function detectBargeIn(userVolume, remoteVolume)" in source
        assert "state.outboundAudioTrack.enabled" in source
        assert "response.cancel" in source
        assert "track.enabled = !state.microphoneMuted" in source


def test_realtime_servers_keep_response_interruption_enabled() -> None:
    for relative_path in (
        "app/server_realtime.py",
        "app/server_feedback.py",
    ):
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

        assert '"interrupt_response": True' in source


def test_realtime_audio_detection_is_sensitive_for_quiet_speech() -> None:
    for relative_path in (
        "app/templates/realtime.html",
        "app/templates/feedback_realtime.html",
    ):
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

        assert "const BARGE_IN_VOICE_THRESHOLD = 0.024;" in source
        assert "const BARGE_IN_REQUIRED_FRAMES = 5;" in source
        assert "state.assistantEchoFloor * 2.0" in source
        assert "remoteVolume * 0.12" in source

    for relative_path in (
        "app/server_realtime.py",
        "app/server_feedback.py",
    ):
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

        assert '"threshold": 0.22' in source


def test_realtime_server_does_not_enable_debug_mode_by_default() -> None:
    source = (PROJECT_ROOT / "app/server_realtime.py").read_text(encoding="utf-8")

    assert 'env_flag("FLASK_DEBUG", default=False)' in source
    assert "debug=True" not in source


def test_depression_inference_runs_only_in_dedicated_worker() -> None:
    detector_source = (
        PROJECT_ROOT / "app" / "depression_detector.py"
    ).read_text(encoding="utf-8")
    server_source = (
        PROJECT_ROOT / "app" / "server_realtime.py"
    ).read_text(encoding="utf-8")
    worker_source = (
        PROJECT_ROOT / "app" / "depression_worker.py"
    ).read_text(encoding="utf-8")

    assert "ThreadPoolExecutor" not in detector_source
    assert "_EXECUTOR" not in detector_source
    assert "run_depression_detection_job(" not in server_source
    assert "claim_next_depression_job(" in worker_source
    assert "acquire_gpu_process_lock(" in worker_source
    assert "get_detector().warm_up()" in worker_source

    server_tree = ast.parse(server_source)
    server_calls = {
        node.func.id
        for node in ast.walk(server_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    }
    assert "prepare_depression_translation_artifacts" not in server_calls


def test_realtime_frontend_has_prediction_loading_and_aspect_modal() -> None:
    source = (PROJECT_ROOT / "app/templates/realtime.html").read_text(encoding="utf-8")

    assert 'id="depressionResultBackdrop"' in source
    assert "function showDepressionLoading" in source
    assert "function renderDepressionResult" in source
    assert "depression-loading-animation" in source
    assert "assessment-loading-main" not in source
    assert "--assessment-close-x" in source
    assert "function setDepressionReturnTarget" in source
    assert "執行編號" not in source
    assert "assessment-warning" not in source
    assert "hard_warnings" not in source
    assert "個人化面向查詢" in source
    assert "檢索到的使用者語句" in source
    assert "標註總分" in source
    assert "function handleDepressionResultButton" in source
    assert 'state.depressionResultView === "retry"' in source
    assert 'id="recordAudioToggle" type="checkbox">' in source
    assert 'id="recordVideoToggle" type="checkbox">' in source


def test_realtime_index_requires_a_complete_success_result() -> None:
    source = (PROJECT_ROOT / "app/server_realtime.py").read_text(encoding="utf-8")

    assert "if depression_result_path.is_file()" in source
    assert 'depression_result.get("status") == "ok"' in source
    assert 'isinstance(depression_result.get("aspects"), list)' in source
    assert "elif depression_error is not None:" in source
