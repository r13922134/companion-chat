import ast
import json
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


def test_realtime_frontends_support_official_text_input_event_flow() -> None:
    for relative_path in (
        "app/templates/realtime.html",
        "app/templates/feedback_realtime.html",
    ):
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

        assert 'id="textInputForm"' in source
        assert 'id="textInput"' in source
        assert 'id="sendTextButton"' in source
        assert 'class="call-controls-row"' in source
        assert "dist/realtime-controls.iife.js" in source
        assert "CompanionRealtimeControls?.mount" in source
        assert "flex-direction: column;" in source
        assert ".liquid-control-glass" in source
        assert ".liquid-control-glass--cluster" in source
        assert ".liquid-control-glass--input .glass" in source
        assert ".liquid-control-glass--input .glass__warp" in source
        assert ".liquid-control-glass--input .text-input-form:focus-within" in source
        assert "control-cluster" in source
        assert "const variantIconColors = {" in source
        assert '<span class="status-icon-color" style="color: ${iconColor};">' in source
        assert 'connected: "#1c1c1e"' in source
        assert 'connected: "#34c759"' in source
        assert 'uploading: "#1c1c1e"' in source
        assert 'uploading: "#007aff"' in source
        assert "prefers-reduced-transparency" in source
        assert "prefers-contrast: more" in source
        assert "font-size: 16px;" in source
        assert ".text-input-form[aria-disabled] {\n      opacity: 1;" in source
        assert ".text-input-form::before" not in source
        assert "backdrop-filter: blur(22px) saturate(150%);" not in source
        assert "backdrop-filter: blur(18px) saturate(148%);" not in source
        assert "right-buttons-group" not in source
        assert ".liquid-control-glass--round" not in source
        assert ".liquid-control-glass--pill" not in source
        assert "speakerLabel" not in source
        assert ".sidebar-log .line small" not in source
        assert 'type: "conversation.item.create"' in source
        assert 'type: "input_text"' in source
        assert "requestAssistantAfterUserTranscript();" in source
        assert 'output_modalities: ["audio"]' in source
        assert (
            'setTextInputEnabled(phase === SESSION_PHASE.LIVE && state.dc?.readyState === "open")'
            in source
        )

    main_source = (PROJECT_ROOT / "app/templates/realtime.html").read_text(encoding="utf-8")
    assert 'recordTranscript("user", itemId, text, { attachAudioInterval: false })' in main_source
    assert "function recordTranscript(speaker, itemId, text, options = {})" in main_source
    assert 'speaker === "user" && options.attachAudioInterval !== false' in main_source


def test_realtime_liquid_glass_react_bundle_is_wired() -> None:
    package_json = json.loads((PROJECT_ROOT / "package.json").read_text(encoding="utf-8"))
    controls_source = (PROJECT_ROOT / "frontend/realtime-controls.jsx").read_text(encoding="utf-8")
    vite_config = (PROJECT_ROOT / "vite.config.js").read_text(encoding="utf-8")
    built_bundle = PROJECT_ROOT / "app/static/dist/realtime-controls.iife.js"

    assert package_json["dependencies"]["liquid-glass-react"] == "1.1.1"
    assert package_json["dependencies"]["react"].startswith("19.")
    assert package_json["dependencies"]["react-dom"].startswith("19.")
    assert package_json["scripts"]["build"] == "vite build"
    assert 'import LiquidGlass from "liquid-glass-react";' in controls_source
    assert "displacementScale: 340" in controls_source
    assert '<Glass kind="input" interactive>' in controls_source
    assert "cluster:" in controls_source
    assert 'className="control-cluster"' in controls_source
    assert 'kind="round"' not in controls_source
    assert 'kind="pill"' not in controls_source
    assert "flushSync" in controls_source
    assert "window.CompanionRealtimeControls" in controls_source
    assert 'fileName: () => "realtime-controls.iife.js"' in vite_config
    assert built_bundle.is_file()


def test_realtime_index_requires_a_complete_success_result() -> None:
    source = (PROJECT_ROOT / "app/server_realtime.py").read_text(encoding="utf-8")

    assert "if depression_result_path.is_file()" in source
    assert 'depression_result.get("status") == "ok"' in source
    assert 'isinstance(depression_result.get("aspects"), list)' in source
    assert "elif depression_error is not None:" in source


def test_hindsight_recall_failure_does_not_enter_prompt(monkeypatch) -> None:
    from app import server_realtime

    def fail_recall(session_hash: str, query: str) -> dict:
        raise RuntimeError("hindsight down")

    monkeypatch.setattr(server_realtime, "recall_hindsight_memory", fail_recall)

    response = server_realtime.app.test_client().post(
        "/api/realtime-response-instructions",
        json={
            "kind": "default",
            "session_hash": "abc123",
            "user_transcript": "我今天有點累",
            "recall_memory": True,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["memory_status"] == "unavailable"
    assert payload["memory_context"] == ""
    assert "# Long-term User Context" not in payload["instructions"]


def test_hindsight_recall_results_are_capped_to_five() -> None:
    from app import server_realtime

    results = [
        {"text": f"memory {index}", "timestamp": f"2026-07-0{index + 1}"}
        for index in range(7)
    ]

    context = server_realtime.format_hindsight_recall_results(results)

    assert context.count("\n") == 4
    assert "memory 0" in context
    assert "memory 4" in context
    assert "memory 5" not in context
    assert "memory 6" not in context
