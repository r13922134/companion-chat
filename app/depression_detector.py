from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import threading
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from .depression_preprocessing import prepare_depression_translation_artifacts
    from .storage import (
        enqueue_depression_job,
        update_realtime_session_run_artifacts,
        update_realtime_session_run_depression,
    )
except ImportError:
    from depression_preprocessing import prepare_depression_translation_artifacts
    from storage import (
        enqueue_depression_job,
        update_realtime_session_run_artifacts,
        update_realtime_session_run_depression,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "vendor"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "qwen3_4b_hubert_resnet_stage2.yaml"
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "dynrag_filtered_epoch_10" / "epoch_10"
)
RESULT_FILENAME = "depression_result.json"
ERROR_FILENAME = "depression_error.json"
PREPROCESSING_FILENAME = "depression_preprocessing.json"
TRANSLATED_TRANSCRIPT_FILENAME = "transcript_depression_english.json"
PARTICIPANT_TRANSCRIPT_FILENAME = "user_speech_intervals.csv"
ASPECT_RETRIEVAL_FILENAME = "depression_aspect_retrieval.jsonl"
ASPECT_PREDICTIONS_FILENAME = "depression_aspect_predictions.csv"
DEFAULT_ASPECT_RETRIEVAL_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DYNRAG_FILTERED_BRANCH = "qwen36_27b_teacher_qwen3_4b_student_hubert_dynrag_filtered"

_DETECTOR: "RealtimeDepressionDetector | None" = None
_DETECTOR_LOCK = threading.Lock()


class DepressionDetectionError(RuntimeError):
    """Raised for expected realtime depression detection failures."""


@dataclass(frozen=True)
class RealtimeRunInput:
    session_hash: str
    run_id: str
    session_dir: Path
    transcript: dict[str, Any]
    transcript_source: str
    preprocessing: dict[str, Any]
    metadata: dict[str, Any]
    user_audio_path: Path
    video_frames_path: Path


def is_depression_detection_enabled() -> bool:
    value = str(os.environ.get("DEPRESSION_DETECTION_ENABLED", "1")).strip().lower()
    return value not in {"0", "false", "no", "off", "disabled"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def read_json_file(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def write_json_file(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl_file(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def refresh_archive_manifest(
    session_dir: Path,
    *,
    prediction_status: str,
) -> dict[str, Any] | None:
    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    manifest_path = session_dir / "archive_manifest.json"
    manifest = read_json_file(manifest_path, default={})
    if not isinstance(manifest, dict) or not manifest:
        return None
    artifacts = []
    for path in sorted(item for item in session_dir.iterdir() if item.is_file()):
        if path.name == manifest_path.name:
            continue
        artifacts.append(
            {
                "field_name": _artifact_field_name(path.name),
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest["updated_at"] = now_iso()
    manifest["prediction_status"] = prediction_status
    manifest["artifacts"] = artifacts
    write_json_file(manifest_path, manifest)
    return manifest


def missing_modality_feature(feature_dim: int):
    import torch

    return torch.zeros(1, int(feature_dim), dtype=torch.float32)


def phq8_ground_truth(metadata: dict[str, Any]) -> dict[str, Any]:
    selected_user = metadata.get("selected_user") if isinstance(metadata, dict) else None
    phq8 = selected_user.get("phq8") if isinstance(selected_user, dict) else None
    item_scores = phq8.get("item_scores") if isinstance(phq8, dict) else None
    aspects: list[int | None] = []
    for item_no in range(1, 9):
        item = item_scores.get(str(item_no)) if isinstance(item_scores, dict) else None
        raw_score = item.get("score") if isinstance(item, dict) else None
        try:
            score = int(raw_score)
        except (TypeError, ValueError):
            score = None
        aspects.append(score if score is not None and 0 <= score <= 3 else None)

    raw_total = phq8.get("total_score") if isinstance(phq8, dict) else None
    try:
        total_score = int(raw_total)
    except (TypeError, ValueError):
        total_score = None
    if total_score is None and all(score is not None for score in aspects):
        total_score = sum(int(score) for score in aspects if score is not None)
    return {
        "scale": "PHQ-8",
        "aspect_scores": aspects,
        "total_score": total_score,
        "binary_depression": None if total_score is None else total_score >= 10,
        "complete": all(score is not None for score in aspects),
    }


def attach_ground_truth(
    result: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    ground_truth = phq8_ground_truth(metadata)
    result["ground_truth"] = ground_truth
    aspects = result.get("aspects")
    if isinstance(aspects, list):
        scores = ground_truth["aspect_scores"]
        for index, aspect in enumerate(aspects[: len(scores)]):
            if isinstance(aspect, dict):
                aspect["ground_truth"] = scores[index]
    result_metadata = result.setdefault("metadata", {})
    if isinstance(result_metadata, dict):
        result_metadata["ground_truth_source"] = (
            "metadata.selected_user.phq8.item_scores"
        )
    return result


def build_user_only_dialogue(transcript: dict[str, Any]) -> str:
    events = transcript.get("events") if isinstance(transcript, dict) else None
    if not isinstance(events, list):
        return ""

    lines: list[str] = []
    utterance_index = 1
    for event in events:
        if not isinstance(event, dict):
            continue
        speaker = str(event.get("speaker") or "").strip().lower()
        if speaker != "user":
            continue
        text = " ".join(str(event.get("text") or "").split())
        if not text:
            continue
        lines.append(f"Utterance {utterance_index}: {text}")
        utterance_index += 1
    return "\n".join(lines)


def resolve_realtime_run_input(
    session_hash: str,
    run_id: str,
    session_dir: Path,
) -> RealtimeRunInput:
    session_dir = Path(session_dir)
    translated_transcript_path = session_dir / TRANSLATED_TRANSCRIPT_FILENAME
    transcript_path = translated_transcript_path if translated_transcript_path.is_file() else session_dir / "transcript.json"
    transcript = read_json_file(transcript_path, default={})
    metadata = read_json_file(session_dir / "metadata.json", default={})
    preprocessing = read_json_file(session_dir / PREPROCESSING_FILENAME, default={})
    if not isinstance(transcript, dict):
        transcript = {}
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(preprocessing, dict):
        preprocessing = {}
    return RealtimeRunInput(
        session_hash=session_hash,
        run_id=run_id,
        session_dir=session_dir,
        transcript=transcript,
        transcript_source=transcript_path.name,
        preprocessing=preprocessing,
        metadata=metadata,
        user_audio_path=session_dir / "user_audio.wav",
        video_frames_path=session_dir / "video_frames.zip",
    )


def _event_float(event: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = event.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            return number
    return None


def realtime_participant_interval_rows(
    transcript: dict[str, Any],
) -> list[tuple[float, float, str]]:
    events = transcript.get("events") if isinstance(transcript, dict) else None
    if not isinstance(events, list):
        return []
    rows: list[tuple[float, float, str]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        speaker = str(event.get("speaker") or "").strip().lower()
        if speaker != "user":
            continue
        text = " ".join(str(event.get("text") or "").split())
        if not text:
            continue
        start = _event_float(
            event,
            "audio_start_seconds",
            "start_seconds",
            "start_time",
        )
        end = _event_float(
            event,
            "audio_end_seconds",
            "end_seconds",
            "end_time",
        )
        if start is None or end is None or end <= start:
            continue
        rows.append((start, end, text))
    return rows


def _wav_duration_seconds(audio_path: Path) -> float | None:
    try:
        import wave

        with wave.open(str(audio_path), "rb") as handle:
            frame_rate = handle.getframerate()
            if frame_rate <= 0:
                return None
            return float(handle.getnframes()) / float(frame_rate)
    except Exception:
        return None


def effective_realtime_participant_interval_rows(
    transcript: dict[str, Any],
    audio_path: Path | None = None,
) -> list[tuple[float, float, str]]:
    min_seconds = 0.1
    try:
        min_seconds = float(
            os.environ.get("DEPRESSION_MIN_USER_SPEECH_INTERVAL_SECONDS") or min_seconds
        )
    except Exception:
        pass
    min_seconds = max(0.0, min_seconds)

    audio_duration = _wav_duration_seconds(audio_path) if audio_path is not None else None
    rows: list[tuple[float, float, str]] = []
    for start, end, text in realtime_participant_interval_rows(transcript):
        clipped_start = max(0.0, float(start))
        clipped_end = float(end)
        if audio_duration is not None:
            clipped_end = min(clipped_end, audio_duration)
        if clipped_end <= clipped_start:
            continue
        if clipped_end - clipped_start < min_seconds:
            continue
        rows.append((clipped_start, clipped_end, text))
    return rows


def write_participant_interval_transcript(
    transcript: dict[str, Any],
    output_path: Path,
    rows: list[tuple[float, float, str]] | None = None,
) -> Path | None:
    rows = rows if rows is not None else realtime_participant_interval_rows(transcript)
    if not rows:
        return None
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Start_Time", "End_Time", "Text"])
        writer.writeheader()
        for start, end, text in rows:
            writer.writerow(
                {
                    "Start_Time": f"{start:.3f}",
                    "End_Time": f"{end:.3f}",
                    "Text": text,
                }
            )
    return output_path


def summarize_depression_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status") or "",
        "total_score": result.get("total_score"),
        "binary": result.get("binary_depression"),
        "completed_at": result.get("completed_at"),
        "error": result.get("error"),
    }


def _ensure_vendor_import_path() -> None:
    vendor_root = str(VENDOR_ROOT)
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)


def _load_checkpoint_or_fail(model: Any, checkpoint: str | Path | None) -> list[str]:
    if not checkpoint:
        raise DepressionDetectionError("Missing depression checkpoint path.")

    import torch

    checkpoint_path = Path(checkpoint)
    checkpoint_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    model_path = checkpoint_path / "mllm_dr.pt" if checkpoint_path.is_dir() else checkpoint_path
    adapter_path = checkpoint_dir / "llm_adapter" / "adapter_model.safetensors"

    load_kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        ckpt = torch.load(model_path, mmap=True, **load_kwargs)
    except TypeError:
        ckpt = torch.load(model_path, **load_kwargs)

    has_adapter = adapter_path.exists()
    has_full_state = "full_state_dict" in ckpt
    if not has_adapter and not has_full_state:
        raise DepressionDetectionError(
            "Stage2 checkpoint is missing llm_adapter and full_state_dict; "
            "refusing degraded base-LLM detection."
        )

    loaded_parts: list[str] = []
    if ckpt.get("audio_lqformer") is not None and model.audio_lqformer is not None:
        model.audio_lqformer.load_state_dict(ckpt["audio_lqformer"], strict=False)
        loaded_parts.append("audio_lqformer")
    if ckpt.get("video_lqformer") is not None and model.video_lqformer is not None:
        model.video_lqformer.load_state_dict(ckpt["video_lqformer"], strict=False)
        loaded_parts.append("video_lqformer")

    if has_adapter:
        if not hasattr(model.llm, "peft_config"):
            raise DepressionDetectionError(
                "Checkpoint has a LoRA adapter, but the loaded model is not a PEFT model."
            )
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        adapter_state = load_file(adapter_path, device="cpu")
        _resize_llm_for_adapter_state(model, adapter_state)
        result = set_peft_model_state_dict(
            model.llm,
            adapter_state,
            adapter_name="default",
            low_cpu_mem_usage=True,
        )
        loaded_parts.append("llm_adapter")
        print(
            "[DEPRESSION] Loaded LoRA adapter "
            f"| missing={len(result.missing_keys)} "
            f"| unexpected={len(result.unexpected_keys)}",
            flush=True,
        )
    elif has_full_state:
        missing, unexpected = model.load_state_dict(ckpt["full_state_dict"], strict=False)
        loaded_parts.append("full_state_dict")
        print(
            "[DEPRESSION] Loaded full checkpoint "
            f"| missing={len(missing)} | unexpected={len(unexpected)}",
            flush=True,
        )

    metadata = ckpt.get("metadata") or {}
    if metadata:
        print(f"[DEPRESSION] Checkpoint metadata: {metadata}", flush=True)
    print(f"[DEPRESSION] Loaded checkpoint parts: {', '.join(loaded_parts)}", flush=True)
    return loaded_parts


def _resize_llm_for_adapter_state(
    model: Any,
    adapter_state: dict[str, Any],
) -> None:
    vocab_sizes = {
        int(tensor.shape[0])
        for key, tensor in adapter_state.items()
        if key.endswith(("embed_tokens.weight", "lm_head.weight")) and tensor.ndim == 2
    }
    if not vocab_sizes:
        return
    if len(vocab_sizes) != 1:
        raise DepressionDetectionError(
            f"Adapter has inconsistent vocab sizes: {sorted(vocab_sizes)}"
        )

    target_vocab_size = next(iter(vocab_sizes))
    current_vocab_size = int(model.llm.get_input_embeddings().weight.shape[0])
    tokenizer_vocab_size = len(model.tokenizer)
    if target_vocab_size < tokenizer_vocab_size:
        raise DepressionDetectionError(
            "Adapter vocab size is smaller than tokenizer vocab size: "
            f"adapter={target_vocab_size} tokenizer={tokenizer_vocab_size}"
        )
    if target_vocab_size != current_vocab_size:
        model.llm.resize_token_embeddings(target_vocab_size)
        print(
            "[DEPRESSION] Resized LLM token embeddings for adapter compatibility: "
            f"{current_vocab_size} -> {target_vocab_size}",
            flush=True,
        )


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_auxiliary_modules(model: Any, device: Any) -> None:
    for module in (
        getattr(model, "audio_lqformer", None),
        getattr(model, "video_lqformer", None),
    ):
        if module is not None:
            module.to(device)
    if hasattr(model, "aspect_embedding_table"):
        model.aspect_embedding_table = model.aspect_embedding_table.to(device)


def _configure_model_for_inference(model: Any) -> None:
    llm = getattr(model, "llm", None)
    disable_gradient_checkpointing = getattr(
        llm,
        "gradient_checkpointing_disable",
        None,
    )
    if callable(disable_gradient_checkpointing):
        disable_gradient_checkpointing()
    config = getattr(llm, "config", None)
    if config is not None:
        config.use_cache = True
    generation_config = getattr(llm, "generation_config", None)
    if generation_config is not None:
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
    model.requires_grad_(False)
    model.eval()


def _move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _build_generation_kwargs(generation_cfg: dict[str, Any]) -> dict[str, Any]:
    do_sample = bool(generation_cfg.get("do_sample", True))
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation_cfg.get("max_new_tokens", 128)),
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs["temperature"] = float(generation_cfg.get("temperature", 0.8))
        kwargs["top_k"] = int(generation_cfg.get("top_k", 10))
        if "top_p" in generation_cfg:
            kwargs["top_p"] = float(generation_cfg["top_p"])
    for key in ("repetition_penalty",):
        if key in generation_cfg:
            kwargs[key] = float(generation_cfg[key])
    for key in ("no_repeat_ngram_size", "num_beams"):
        if key in generation_cfg:
            kwargs[key] = int(generation_cfg[key])
    return kwargs


def _sanitize_csv_text(text: str) -> str:
    return " ".join((text or "").split())


def _aspect_probability_columns(model: Any, forward: Any, row_index: int) -> dict[str, Any]:
    columns = {f"tract_prob_{value}": None for value in range(4)}
    probs = getattr(forward, "tract_score_probs", None)
    values = getattr(model, "tract_score_values", None)
    if probs is None or values is None or probs.numel() == 0 or values.numel() == 0:
        return columns
    sample_probs = probs[row_index].detach().float().cpu().tolist()
    score_values = values.detach().cpu().tolist()
    for value, prob in zip(score_values, sample_probs, strict=False):
        key = f"tract_prob_{int(value)}"
        if key in columns:
            columns[key] = float(prob)
    return columns


class RealtimeDepressionDetector:
    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    ) -> None:
        self.config_path = Path(
            os.environ.get("DEPRESSION_CONFIG_PATH", str(config_path))
        ).expanduser()
        self.checkpoint_path = Path(
            os.environ.get("DEPRESSION_CHECKPOINT_PATH", str(checkpoint_path))
        ).expanduser()
        self._lock = threading.Lock()
        self._loaded = False
        self._model = None
        self._cfg: dict[str, Any] = {}
        self._loaded_checkpoint_parts: list[str] = []
        self._audio_feature_extractor: Any = None
        self._audio_feature_extractor_key: tuple[Any, ...] | None = None
        self._video_feature_extractor: ResNetFrameFeatureExtractor | None = None
        self._aspect_retrieval_backend: Any = None
        self._aspect_retrieval_backend_key: tuple[Any, ...] | None = None

    def detect(self, run_input: RealtimeRunInput) -> dict[str, Any]:
        self._validate_formal_realtime_inputs(run_input)
        with self._lock:
            self._lazy_load()
            return self._detect_loaded(run_input)

    def _validate_formal_realtime_inputs(self, run_input: RealtimeRunInput) -> None:
        if not build_user_only_dialogue(run_input.transcript):
            raise DepressionDetectionError(
                "No user transcript utterances for depression detection."
            )

    def _get_aspect_retrieval_backend(self) -> Any:
        _ensure_vendor_import_path()
        try:
            from .depression_aspect_retrieval import (
                LoadedLlmHiddenBackend,
                LocalHiddenBackend,
            )
        except ImportError:
            from depression_aspect_retrieval import (
                LoadedLlmHiddenBackend,
                LocalHiddenBackend,
            )

        requested_backend = str(
            os.environ.get("DEPRESSION_ASPECT_RETRIEVAL_BACKEND") or "hidden"
        ).strip().lower()
        if requested_backend not in {"hidden", "qwen_hidden", "local_hidden"}:
            raise DepressionDetectionError(
                "retrieval_failed: dynrag_filtered realtime detection requires "
                "local_hidden_l2 retrieval with the base Qwen model; lexical or "
                f"loaded detector backends are not valid. requested={requested_backend}"
            )

        model_name = str(
            os.environ.get("DEPRESSION_ASPECT_RETRIEVAL_MODEL")
            or DEFAULT_ASPECT_RETRIEVAL_MODEL
        ).strip()
        raw_device_map = str(
            os.environ.get("DEPRESSION_ASPECT_RETRIEVAL_DEVICE_MAP") or "auto"
        ).strip()
        device_map = (
            None
            if raw_device_map.lower() in {"", "0", "false", "no", "none", "off"}
            else raw_device_map
        )
        batch_size = int(os.environ.get("DEPRESSION_ASPECT_RETRIEVAL_BATCH_SIZE") or 16)
        max_embedding_length = int(
            os.environ.get("DEPRESSION_ASPECT_RETRIEVAL_MAX_EMBEDDING_LENGTH") or 256
        )
        max_profile_new_tokens = int(
            os.environ.get("DEPRESSION_ASPECT_RETRIEVAL_MAX_PROFILE_NEW_TOKENS") or 192
        )
        max_query_new_tokens = int(
            os.environ.get("DEPRESSION_ASPECT_RETRIEVAL_MAX_QUERY_NEW_TOKENS") or 64
        )
        share_detector_llm = str(
            os.environ.get("DEPRESSION_ASPECT_RETRIEVAL_SHARE_LLM", "1")
        ).strip().lower() not in {"0", "false", "no", "off", "disabled"}
        backend_key = (
            model_name,
            device_map,
            batch_size,
            max_embedding_length,
            max_profile_new_tokens,
            max_query_new_tokens,
            share_detector_llm,
        )
        if (
            self._aspect_retrieval_backend is not None
            and self._aspect_retrieval_backend_key == backend_key
        ):
            return self._aspect_retrieval_backend

        try:
            if share_detector_llm:
                if self._model is None:
                    raise DepressionDetectionError(
                        "Detector model must be loaded before shared retrieval."
                    )
                detector_model_name = str(
                    self._cfg.get("model", {}).get("base_model_name_or_path") or ""
                ).strip()
                if model_name != detector_model_name:
                    raise DepressionDetectionError(
                        "Shared retrieval requires the same base model as the "
                        f"detector: retrieval={model_name} detector={detector_model_name}. "
                        "Set DEPRESSION_ASPECT_RETRIEVAL_SHARE_LLM=0 to load a "
                        "separate retrieval model."
                    )
                backend = LoadedLlmHiddenBackend(
                    tokenizer=self._model.tokenizer,
                    llm=self._model.llm,
                    model_name=model_name,
                    batch_size=batch_size,
                    max_embedding_length=max_embedding_length,
                    max_profile_new_tokens=max_profile_new_tokens,
                    max_query_new_tokens=max_query_new_tokens,
                )
            else:
                backend = LocalHiddenBackend(
                    model_name,
                    device_map=device_map,
                    batch_size=batch_size,
                    max_embedding_length=max_embedding_length,
                    max_profile_new_tokens=max_profile_new_tokens,
                    max_query_new_tokens=max_query_new_tokens,
                )
        except Exception as exc:
            raise DepressionDetectionError(
                "retrieval_failed: unable to load base aspect retrieval model "
                f"{model_name}: {exc}"
            ) from exc
        self._aspect_retrieval_backend = backend
        self._aspect_retrieval_backend_key = backend_key
        return backend

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        _ensure_vendor_import_path()
        from mllm_dr.config import load_config, set_seed
        from mllm_dr.model import load_mllm_dr

        cfg = load_config(self.config_path)
        cfg.setdefault("training", {})
        if os.environ.get("DEPRESSION_DEVICE_MAP"):
            cfg["training"]["device_map"] = os.environ["DEPRESSION_DEVICE_MAP"]
        set_seed(int(cfg.get("seed", 42)))
        model = load_mllm_dr(cfg)
        loaded_parts = _load_checkpoint_or_fail(model, self.checkpoint_path)
        device = _device()
        if cfg.get("training", {}).get("device_map") is None:
            model.to(device)
        else:
            _move_auxiliary_modules(model, device)
        _configure_model_for_inference(model)
        self._cfg = cfg
        self._model = model
        self._loaded_checkpoint_parts = loaded_parts
        self._loaded = True

    def warm_up(self) -> dict[str, Any]:
        with self._lock:
            self._lazy_load()
            if self._model is None:
                raise DepressionDetectionError("Depression detector model is not loaded.")

            import numpy as np
            import torch
            from PIL import Image
            from mllm_dr.data.edaic import pad_sequence_features
            from mllm_dr.prompts import PHQ8_ASPECTS
            from mllm_dr.training.text import build_inference_encoding, pad_token_batch

            device = _device()
            model = self._model
            cfg = self._cfg
            data_cfg = cfg.get("dataset", {})
            model_cfg = cfg.get("model", {})
            features_cfg = cfg.get("features", {})

            retrieval = self._get_aspect_retrieval_backend()
            generate_one = getattr(retrieval, "_generate", None)
            if callable(generate_one):
                generate_one("Summarize: I have felt tired recently.", 1)
            retrieval.embed_texts(["I have felt tired recently."])

            warm_cache = PROJECT_ROOT / "data" / ".depression_warmup" / "hubert"
            audio_extractor = self._get_audio_feature_extractor(
                cache_dir=warm_cache,
                features_cfg=features_cfg,
                device=device,
            )
            audio_extractor._lazy_load()
            audio_extractor._extract_hubert_chunks(
                np.zeros(16000, dtype=np.float32)
            )

            if self._video_feature_extractor is None:
                self._video_feature_extractor = ResNetFrameFeatureExtractor(device=device)
            self._video_feature_extractor._lazy_load()
            blank_image = Image.new("RGB", (224, 224), color=(0, 0, 0))
            video_tensor = self._video_feature_extractor._preprocess(blank_image)
            self._video_feature_extractor._forward_batch([video_tensor])

            aspect_index = 0
            _, aspect_description = PHQ8_ASPECTS[aspect_index]
            encoding = build_inference_encoding(
                tokenizer=model.tokenizer,
                dialogue="Utterance 1: I have felt tired recently.",
                aspect=aspect_description,
                max_length=min(int(data_cfg.get("max_text_length", 2048)), 512),
                score_after_rationale=bool(model_cfg.get("score_after_rationale", False)),
                score_only_response=bool(model_cfg.get("score_only_response", False)),
                rationale_only_response=bool(
                    model_cfg.get("rationale_only_response", False)
                ),
                merge_system_into_user=bool(
                    model_cfg.get("merge_system_into_user", False)
                ),
                chat_template_kwargs=model_cfg.get("chat_template_kwargs"),
            )
            tokens = _move_batch(
                pad_token_batch([encoding], model.tokenizer.pad_token_id),
                device,
            )
            audio_dim = int(model_cfg.get("audio_input_dim", 768))
            video_dim = int(model_cfg.get("video_input_dim", 2048))
            audio, audio_mask = pad_sequence_features(
                [missing_modality_feature(audio_dim)],
                audio_dim,
            )
            video, video_mask = pad_sequence_features(
                [missing_modality_feature(video_dim)],
                video_dim,
            )
            aspect_ids = torch.tensor([aspect_index], dtype=torch.long, device=device)
            with torch.inference_mode():
                model.generate(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    aspect_ids=aspect_ids,
                    audio_features=audio.to(device),
                    audio_mask=audio_mask.to(device),
                    video_features=video.to(device),
                    video_mask=video_mask.to(device),
                    max_new_tokens=1,
                    do_sample=False,
                )
            return {
                "status": "ready",
                "device": str(device),
                "checkpoint_parts": list(self._loaded_checkpoint_parts),
                "retrieval_backend": retrieval.name,
                "retrieval_shared_llm": bool(
                    getattr(retrieval, "llm", None) is model.llm
                ),
                "audio_model": features_cfg.get(
                    "hubert_model_name",
                    "facebook/hubert-base-ls960",
                ),
                "video_model": "torchvision/resnet50",
            }

    def _get_audio_feature_extractor(
        self,
        *,
        cache_dir: Path,
        features_cfg: dict[str, Any],
        device: Any,
    ) -> Any:
        _ensure_vendor_import_path()
        from mllm_dr.data import FeatureExtractor

        model_name = features_cfg.get(
            "hubert_model_name",
            "facebook/hubert-base-ls960",
        )
        chunk_seconds = float(features_cfg.get("hubert_chunk_seconds", 30.0))
        max_interval_seconds = features_cfg.get("max_interval_seconds", 30.0)
        extractor_key = (
            model_name,
            str(device),
            chunk_seconds,
            max_interval_seconds,
        )
        if (
            self._audio_feature_extractor is None
            or self._audio_feature_extractor_key != extractor_key
        ):
            self._audio_feature_extractor = FeatureExtractor(
                cache_dir=cache_dir,
                hubert_model_name=model_name,
                device=device,
                chunk_seconds=chunk_seconds,
                participant_only=True,
                max_interval_seconds=max_interval_seconds,
            )
            self._audio_feature_extractor_key = extractor_key
        else:
            self._audio_feature_extractor.cache_dir = Path(cache_dir)
            self._audio_feature_extractor.cache_dir.mkdir(parents=True, exist_ok=True)
        return self._audio_feature_extractor

    def _detect_loaded(self, run_input: RealtimeRunInput) -> dict[str, Any]:
        import numpy as np
        import torch

        from mllm_dr.data.edaic import pad_sequence_features, temporal_uniform_sample
        from mllm_dr.metrics import parse_evaluation_result
        from mllm_dr.prompts import PHQ8_ASPECTS
        from mllm_dr.training.text import (
            build_inference_encoding,
            build_scoring_encoding,
            extract_rationale_text,
            pad_token_batch,
        )
        try:
            from .depression_aspect_retrieval import (
                build_aspect_retrieval_records,
                config_from_env,
                transcript_user_utterances,
            )
        except ImportError:
            from depression_aspect_retrieval import (
                build_aspect_retrieval_records,
                config_from_env,
                transcript_user_utterances,
            )

        if self._model is None:
            raise DepressionDetectionError("Depression detector model is not loaded.")

        utterances = transcript_user_utterances(run_input.transcript)
        if not utterances:
            raise DepressionDetectionError("No user transcript utterances for depression detection.")

        cfg = self._cfg
        model = self._model
        data_cfg = cfg.get("dataset", {})
        features_cfg = cfg.get("features", {})
        model_cfg = cfg.get("model", {})
        generation_cfg = cfg.get("generation", {})
        device = _device()

        retrieval_config = config_from_env()
        requested_retrieval_backend = str(
            os.environ.get("DEPRESSION_ASPECT_RETRIEVAL_BACKEND") or "hidden"
        ).strip().lower()
        retrieval_backend = self._get_aspect_retrieval_backend()
        try:
            retrieval_output = build_aspect_retrieval_records(
                utterances=utterances,
                backend=retrieval_backend,
                config=retrieval_config,
                split="realtime",
                participant_id=0,
            )
        except Exception as exc:
            raise DepressionDetectionError(
                "retrieval_failed: base Qwen aspect retrieval failed; refusing "
                f"lexical fallback for formal dynrag_filtered detection: {exc}"
            ) from exc
        aspect_retrieval_rows = {
            int(row["aspect_index"]): row for row in retrieval_output.records
        }
        write_jsonl_file(
            run_input.session_dir / ASPECT_RETRIEVAL_FILENAME,
            retrieval_output.records,
        )

        audio_feature_dim = int(model_cfg.get("audio_input_dim", 768))
        video_feature_dim = int(model_cfg.get("video_input_dim", 2048))
        participant_interval_rows: list[tuple[float, float, str]] = []
        participant_transcript_path: Path | None = None
        use_participant_only_audio = False
        audio_zero_fill_reason: str | None = None
        if not run_input.user_audio_path.is_file():
            audio_zero_fill_reason = "missing_user_audio_file"
            audio = missing_modality_feature(audio_feature_dim)
        else:
            participant_interval_rows = effective_realtime_participant_interval_rows(
                run_input.transcript,
                run_input.user_audio_path,
            )
            if not participant_interval_rows:
                audio_zero_fill_reason = "missing_valid_user_speech_intervals"
                audio = missing_modality_feature(audio_feature_dim)
            else:
                participant_transcript_path = write_participant_interval_transcript(
                    run_input.transcript,
                    run_input.session_dir / PARTICIPANT_TRANSCRIPT_FILENAME,
                    rows=participant_interval_rows,
                )
                if participant_transcript_path is None:
                    audio_zero_fill_reason = "failed_to_write_user_speech_intervals"
                    audio = missing_modality_feature(audio_feature_dim)
                else:
                    use_participant_only_audio = True
                    cache_dir = run_input.session_dir / ".depression_cache" / "hubert"
                    feature_extractor = self._get_audio_feature_extractor(
                        cache_dir=cache_dir,
                        features_cfg=features_cfg,
                        device=device,
                    )
                    audio = feature_extractor.load_or_extract_audio(
                        run_input.user_audio_path,
                        transcript_path=participant_transcript_path,
                    )
                    if audio.numel() == 0 or int(torch.count_nonzero(audio).item()) == 0:
                        audio_zero_fill_reason = "empty_hubert_features"
                        audio = missing_modality_feature(audio_feature_dim)
        if data_cfg.get("max_audio_frames"):
            audio = temporal_uniform_sample(audio, int(data_cfg["max_audio_frames"]))

        video_zero_fill_reason: str | None = None
        if not run_input.video_frames_path.is_file():
            video_zero_fill_reason = "missing_video_frames_file"
            video = missing_modality_feature(video_feature_dim)
        else:
            video = self._extract_video_features(run_input.video_frames_path, device)
            if video.numel() == 0 or int(torch.count_nonzero(video).item()) == 0:
                video_zero_fill_reason = "empty_video_features"
                video = missing_modality_feature(video_feature_dim)
        if data_cfg.get("max_video_frames"):
            video = temporal_uniform_sample(video, int(data_cfg["max_video_frames"]))

        audio_batch, audio_mask = pad_sequence_features([audio], audio_feature_dim)
        video_batch, video_mask = pad_sequence_features([video], video_feature_dim)
        audio_batch = audio_batch.to(device)
        audio_mask = audio_mask.to(device)
        video_batch = video_batch.to(device)
        video_mask = video_mask.to(device)

        use_generation_scoring = bool(model_cfg.get("use_tract_raft", False))
        score_source = cfg.get("prediction", {}).get("score_source", "tract")
        ground_truth = phq8_ground_truth(run_input.metadata)
        ground_truth_aspects = ground_truth["aspect_scores"]
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for aspect_index, (aspect_key, aspect) in enumerate(PHQ8_ASPECTS):
                retrieval_row = aspect_retrieval_rows.get(aspect_index, {})
                dialogue = str(retrieval_row.get("dialogue") or "").strip()
                if not dialogue:
                    raise DepressionDetectionError(
                        f"Aspect retrieval returned empty dialogue for aspect_index={aspect_index}."
                    )
                encoding = build_inference_encoding(
                    tokenizer=model.tokenizer,
                    dialogue=dialogue,
                    aspect=aspect,
                    max_length=int(data_cfg.get("max_text_length", 2048)),
                    score_after_rationale=bool(model_cfg.get("score_after_rationale", False)),
                    score_only_response=bool(model_cfg.get("score_only_response", False)),
                    rationale_only_response=bool(
                        model_cfg.get("rationale_only_response", False)
                    ),
                    merge_system_into_user=bool(
                        model_cfg.get("merge_system_into_user", False)
                    ),
                    chat_template_kwargs=model_cfg.get("chat_template_kwargs"),
                )
                tokens = pad_token_batch([encoding], model.tokenizer.pad_token_id)
                tokens = _move_batch(tokens, device)
                aspect_ids = torch.tensor([aspect_index], dtype=torch.long, device=device)
                generated = model.generate(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    aspect_ids=aspect_ids,
                    audio_features=audio_batch,
                    audio_mask=audio_mask,
                    video_features=video_batch,
                    video_mask=video_mask,
                    **_build_generation_kwargs(generation_cfg),
                )
                decoded = model.tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )
                generation_text = decoded[0] if decoded else ""
                if use_generation_scoring:
                    scoring_encoding = build_scoring_encoding(
                        tokenizer=model.tokenizer,
                        dialogue=dialogue,
                        aspect=aspect,
                        generation_text=generation_text,
                        max_length=int(data_cfg.get("max_text_length", 2048)),
                        include_score_token_mask=True,
                        score_after_rationale=bool(
                            model_cfg.get("score_after_rationale", False)
                        ),
                        score_only_response=bool(model_cfg.get("score_only_response", False)),
                        merge_system_into_user=bool(
                            model_cfg.get("merge_system_into_user", False)
                        ),
                        chat_template_kwargs=model_cfg.get("chat_template_kwargs"),
                    )
                    scoring_tokens = pad_token_batch(
                        [scoring_encoding],
                        model.tokenizer.pad_token_id,
                    )
                    scoring_tokens = _move_batch(scoring_tokens, device)
                    forward = model(
                        input_ids=scoring_tokens["input_ids"],
                        attention_mask=scoring_tokens["attention_mask"],
                        labels=None,
                        aspect_ids=aspect_ids,
                        audio_features=audio_batch,
                        audio_mask=audio_mask,
                        video_features=video_batch,
                        video_mask=video_mask,
                        score_token_mask=scoring_tokens.get("score_token_mask"),
                        regression_labels=None,
                        use_mse=False,
                    )
                else:
                    forward = model(
                        input_ids=tokens["input_ids"],
                        attention_mask=tokens["attention_mask"],
                        labels=None,
                        aspect_ids=aspect_ids,
                        audio_features=audio_batch,
                        audio_mask=audio_mask,
                        video_features=video_batch,
                        video_mask=video_mask,
                        score_token_mask=tokens.get("score_token_mask"),
                        regression_labels=None,
                        use_mse=False,
                    )

                parsed = parse_evaluation_result(generation_text)
                regression_score = (
                    float(forward.regression_scores[0].detach().cpu())
                    if forward.regression_scores is not None
                    else float("nan")
                )
                tract_score = (
                    float(forward.tract_scores[0].detach().cpu())
                    if forward.tract_scores is not None
                    else float("nan")
                )
                regression_clipped = float(np.clip(regression_score, 0.0, 3.0))
                tract_clipped = float(np.clip(tract_score, 0.0, 3.0))
                if score_source == "tract":
                    prediction = regression_clipped if np.isnan(tract_score) else tract_clipped
                    prediction_source = "missing_tract" if np.isnan(tract_score) else "tract"
                elif score_source == "tract_rounded":
                    prediction = (
                        int(np.clip(round(regression_score), 0, 3))
                        if np.isnan(tract_score)
                        else int(np.clip(round(tract_score), 0, 3))
                    )
                    prediction_source = (
                        "missing_tract_rounded"
                        if np.isnan(tract_score)
                        else "tract_rounded"
                    )
                elif score_source == "generation":
                    prediction = float("nan") if parsed is None else float(parsed)
                    prediction_source = "generation" if parsed is not None else "missing"
                else:
                    prediction = float(parsed) if parsed is not None else regression_clipped
                    prediction_source = "generation" if parsed is not None else "tract"

                rows.append(
                    {
                        "aspect_index": aspect_index,
                        "aspect_key": aspect_key,
                        "aspect": aspect,
                        "clinical_description": retrieval_row.get(
                            "aspect_description"
                        ),
                        "prediction": prediction,
                        "ground_truth": ground_truth_aspects[aspect_index],
                        "prediction_source": prediction_source,
                        "generation_score": parsed,
                        "tract_score": None if np.isnan(tract_score) else tract_clipped,
                        "regression_score": regression_score,
                        "tract_raw_score": None if np.isnan(tract_score) else tract_score,
                        "retrieval_query": _sanitize_csv_text(
                            str(retrieval_row.get("query") or "")
                        ),
                        "retrieved_utterance_indices": retrieval_row.get(
                            "utterance_indices",
                            [],
                        ),
                        "retrieval_scores": retrieval_row.get("scores", []),
                        "retrieved_utterances": retrieval_row.get("utterances", []),
                        "global_fallback_utterance_indices": retrieval_row.get(
                            "global_fallback_utterance_indices",
                            [],
                        ),
                        "global_fallback_utterance_rows": retrieval_row.get(
                            "global_fallback_utterance_rows",
                            [],
                        ),
                        "selected_dialogue_tokens": retrieval_row.get(
                            "selected_dialogue_tokens"
                        ),
                        **_aspect_probability_columns(model, forward, 0),
                        "generated_rationale": _sanitize_csv_text(
                            f"Evaluation Reason: {extract_rationale_text(generation_text)}"
                        ),
                        "generation": _sanitize_csv_text(generation_text),
                    }
                )

        numeric_predictions = [
            float(row["prediction"])
            for row in rows
            if row.get("prediction") is not None and np.isfinite(float(row["prediction"]))
        ]
        total_score = float(sum(numeric_predictions)) if len(numeric_predictions) == 8 else None
        completed_at = now_iso()
        translation_metadata = run_input.preprocessing.get("translation")
        translation_status = (
            str(translation_metadata.get("status") or "")
            if isinstance(translation_metadata, dict)
            else ""
        )
        text_language_shift_risk = None
        if (
            run_input.transcript_source != TRANSLATED_TRANSCRIPT_FILENAME
            and translation_status != "skipped_already_english"
        ):
            text_language_shift_risk = (
                "E-DAIC training transcripts are English; realtime text was "
                "not translated before depression detection."
            )
        audio_interval_total_seconds = float(
            sum(max(0.0, end - start) for start, end, _ in participant_interval_rows)
        )
        video_shift_warning = {
            "code": "video_feature_source_shift",
            "severity": "hard_warning",
            "message": (
                "Training used E-DAIC participant video ResNet .mat features; "
                "realtime inference uses browser-captured JPEG frames and an "
                "online torchvision ResNet-50 extractor."
            ),
            "score_is_formal_but_not_edaic_video_aligned": True,
        }
        hard_warnings = []
        if video_zero_fill_reason is None:
            hard_warnings.append(video_shift_warning)
        for modality, reason in (
            ("audio", audio_zero_fill_reason),
            ("video", video_zero_fill_reason),
        ):
            if reason:
                hard_warnings.append(
                    {
                        "code": f"{modality}_modality_zero_filled",
                        "severity": "hard_warning",
                        "message": (
                            f"{modality} modality was unavailable and replaced "
                            f"with a zero feature vector ({reason})."
                        ),
                        "modality": modality,
                        "reason": reason,
                    }
                )
        result = {
            "status": "ok",
            "session_hash": run_input.session_hash,
            "run_id": run_input.run_id,
            "completed_at": completed_at,
            "total_score": total_score,
            "binary_depression": None if total_score is None else bool(total_score >= 10.0),
            "threshold": 10.0,
            "ground_truth": ground_truth,
            "aspects": rows,
            "metadata": {
                "source": "web_realtime",
                "target_thesis_branch": DYNRAG_FILTERED_BRANCH,
                "method_alignment": {
                    "aspect_retrieval_requires_base_qwen_hidden": True,
                    "aspect_retrieval_lexical_fallback_allowed": False,
                    "audio_requires_user_speech_intervals_when_available": True,
                    "audio_full_mic_fallback_allowed": False,
                    "missing_audio_zero_fill_allowed": True,
                    "missing_video_zero_fill_allowed": True,
                    "video_aligned_to_edaic_mat_resnet": False,
                },
                "hard_warnings": hard_warnings,
                "dialogue_source": "transcript.events.user_only.aspect_retrieval",
                "transcript_source": run_input.transcript_source,
                "text_translation": translation_metadata,
                "text_language_shift_risk": text_language_shift_risk,
                "aspect_retrieval_source": ASPECT_RETRIEVAL_FILENAME,
                "aspect_retrieval_backend_requested": requested_retrieval_backend,
                "aspect_retrieval_backend": retrieval_output.backend_name,
                "aspect_retrieval_model": retrieval_output.backend_model_name,
                "aspect_retrieval_shared_detector_llm": bool(
                    getattr(retrieval_backend, "llm", None) is model.llm
                ),
                "aspect_retrieval_adapter_disabled": bool(
                    getattr(retrieval_backend, "llm", None) is model.llm
                    and callable(getattr(model.llm, "disable_adapter", None))
                ),
                "aspect_retrieval_fallback_allowed": False,
                "aspect_retrieval_expected_backend": "local_hidden_l2",
                "aspect_retrieval_expected_model": DEFAULT_ASPECT_RETRIEVAL_MODEL,
                "aspect_retrieval_config": {
                    "min_utterances": retrieval_config.min_utterances,
                    "max_utterances": retrieval_config.max_utterances,
                    "max_dialogue_tokens": retrieval_config.max_dialogue_tokens,
                    "min_score": retrieval_config.min_score,
                    "relative_score_margin": retrieval_config.relative_score_margin,
                    "candidate_filter": retrieval_config.candidate_filter,
                    "candidate_filter_min_words": (
                        retrieval_config.candidate_filter_min_words
                    ),
                    "min_candidate_utterances": (
                        retrieval_config.min_candidate_utterances
                    ),
                    "query_prompt_version": retrieval_config.query_prompt_version,
                    "context_window": retrieval_config.context_window,
                    "adaptive_reflect": retrieval_config.adaptive_reflect,
                    "evidence_rerank": retrieval_config.evidence_rerank,
                    "min_score_applies_to_min": (
                        retrieval_config.min_score_applies_to_min
                    ),
                    "global_fallback_mode": retrieval_config.global_fallback_mode,
                    "global_fallback_utterances": (
                        retrieval_config.global_fallback_utterances
                    ),
                    "global_fallback_max_dialogue_tokens": (
                        retrieval_config.global_fallback_max_dialogue_tokens
                    ),
                },
                "candidate_filter_summary": retrieval_output.candidate_filter_summary,
                "audio_source": (
                    str(run_input.user_audio_path)
                    if run_input.user_audio_path.is_file()
                    else None
                ),
                "audio_feature_source": (
                    "zero_fill"
                    if audio_zero_fill_reason
                    else "hubert_participant_only"
                ),
                "audio_zero_filled": audio_zero_fill_reason is not None,
                "audio_zero_fill_reason": audio_zero_fill_reason,
                "audio_participant_only": use_participant_only_audio,
                "audio_participant_interval_source": (
                    str(participant_transcript_path)
                    if participant_transcript_path is not None
                    else None
                ),
                "audio_participant_interval_count": len(participant_interval_rows),
                "audio_participant_interval_total_seconds": audio_interval_total_seconds,
                "audio_full_mic_fallback_used": False,
                "audio_distribution_shift_risk": (
                    "Audio modality was zero-filled."
                    if audio_zero_fill_reason
                    else None
                ),
                "audio_hubert_sample_rate": 16000,
                "audio_hubert_chunk_seconds": float(
                    features_cfg.get("hubert_chunk_seconds", 30.0)
                ),
                "video_source": (
                    str(run_input.video_frames_path)
                    if run_input.video_frames_path.is_file()
                    else None
                ),
                "video_feature_source": (
                    "zero_fill"
                    if video_zero_fill_reason
                    else "torchvision_resnet50_imagenet_pool"
                ),
                "video_zero_filled": video_zero_fill_reason is not None,
                "video_zero_fill_reason": video_zero_fill_reason,
                "video_training_feature_source": "edaic_participant_CNN_ResNet_mat",
                "video_distribution_shift_warning": (
                    None if video_zero_fill_reason else video_shift_warning
                ),
                "video_distribution_shift_risk": (
                    "Video modality was zero-filled."
                    if video_zero_fill_reason
                    else (
                        "Training used E-DAIC .mat ResNet features; realtime uses "
                        "JPEG frames extracted in-browser and torchvision ResNet-50."
                    )
                ),
                "video_capture_metadata": {
                    "archive_video_enabled": run_input.metadata.get(
                        "archive_video_enabled"
                    ),
                    "video_frame_source": run_input.metadata.get(
                        "video_frame_source"
                    ),
                    "video_frame_sampling_interval_ms": run_input.metadata.get(
                        "video_frame_sampling_interval_ms"
                    ),
                    "video_frame_target_width": run_input.metadata.get(
                        "video_frame_target_width"
                    ),
                    "video_frame_target_height": run_input.metadata.get(
                        "video_frame_target_height"
                    ),
                    "video_frame_jpeg_quality": run_input.metadata.get(
                        "video_frame_jpeg_quality"
                    ),
                    "video_frame_count": run_input.metadata.get("video_frame_count"),
                },
                "config_path": str(self.config_path),
                "checkpoint_path": str(self.checkpoint_path),
                "checkpoint_parts": self._loaded_checkpoint_parts,
                "score_source": score_source,
                "aspect_count": len(rows),
                "user_utterance_count": len(utterances),
                "ground_truth_source": "metadata.selected_user.phq8.item_scores",
                "audio_feature_shape": list(audio.shape),
                "video_feature_shape": list(video.shape),
            },
        }
        return result

    def _extract_video_features(self, video_frames_path: Path, device: Any):
        if self._video_feature_extractor is None:
            self._video_feature_extractor = ResNetFrameFeatureExtractor(device=device)
        return self._video_feature_extractor.extract_zip(video_frames_path)


class ResNetFrameFeatureExtractor:
    def __init__(self, device: Any = None, batch_size: int | None = None) -> None:
        self.device = device or _device()
        self.batch_size = int(
            batch_size or os.environ.get("DEPRESSION_VIDEO_BATCH_SIZE", "16")
        )
        self._model = None
        self._preprocess = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch
        from torch import nn
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = nn.Identity()
        model.to(self.device)
        model.eval()
        self._model = model
        self._preprocess = weights.transforms()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def extract_zip(self, video_frames_path: Path):
        import torch
        from PIL import Image

        self._lazy_load()
        if self._model is None or self._preprocess is None:
            raise DepressionDetectionError("ResNet video feature extractor failed to load.")

        names: list[str]
        with zipfile.ZipFile(video_frames_path, "r") as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            if not names:
                return torch.zeros(1, 2048, dtype=torch.float32)

            chunks: list[Any] = []
            batch: list[Any] = []
            for name in names:
                with archive.open(name) as handle:
                    image = Image.open(handle).convert("RGB")
                    batch.append(self._preprocess(image))
                if len(batch) >= self.batch_size:
                    chunks.append(self._forward_batch(batch))
                    batch = []
            if batch:
                chunks.append(self._forward_batch(batch))

        if not chunks:
            return torch.zeros(1, 2048, dtype=torch.float32)
        return torch.cat(chunks, dim=0).float().cpu()

    def _forward_batch(self, batch: list[Any]):
        import torch

        inputs = torch.stack(batch, dim=0).to(self.device)
        with torch.inference_mode():
            features = self._model(inputs)
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        out = features.detach().cpu().float()
        del inputs, features
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return out


def get_detector() -> RealtimeDepressionDetector:
    global _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            _DETECTOR = RealtimeDepressionDetector()
        return _DETECTOR


def run_depression_detection_job(
    database_path: Path,
    session_hash: str,
    run_id: str,
    session_dir: Path,
) -> dict[str, Any]:
    update_realtime_session_run_depression(
        database_path,
        session_hash,
        run_id,
        {"status": "running"},
    )
    try:
        source_transcript = read_json_file(
            Path(session_dir) / "transcript.json",
            default={},
        )
        prepare_depression_translation_artifacts(
            Path(session_dir),
            source_transcript if isinstance(source_transcript, dict) else None,
        )
        run_input = resolve_realtime_run_input(session_hash, run_id, session_dir)
        result = get_detector().detect(run_input)
        write_json_file(session_dir / RESULT_FILENAME, result)
        write_aspect_predictions_csv(
            result,
            session_dir / ASPECT_PREDICTIONS_FILENAME,
        )
        archive_manifest = refresh_archive_manifest(
            session_dir,
            prediction_status="ok",
        )
        artifact_paths = sorted(path for path in session_dir.iterdir() if path.is_file())
        update_realtime_session_run_artifacts(
            database_path,
            session_hash,
            run_id,
            saved_file_names=[path.name for path in artifact_paths],
            saved_file_paths=[
                {
                    "field_name": _artifact_field_name(path.name),
                    "filename": path.name,
                    "path": str(path),
                }
                for path in artifact_paths
            ],
            archive_manifest=archive_manifest,
        )
        update_realtime_session_run_depression(
            database_path,
            session_hash,
            run_id,
            {
                "status": "ok",
                "total_score": result.get("total_score"),
                "binary": result.get("binary_depression"),
                "result": result,
                "completed_at": result.get("completed_at") or now_iso(),
            },
        )
        print(
            f"[DEPRESSION] Done | session={session_hash} | run={run_id} "
            f"| total={result.get('total_score')}",
            flush=True,
        )
        return result
    except Exception as exc:
        completed_at = now_iso()
        error_result = {
            "status": "error",
            "session_hash": session_hash,
            "run_id": run_id,
            "completed_at": completed_at,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_file(session_dir / ERROR_FILENAME, error_result)
        archive_manifest = refresh_archive_manifest(
            session_dir,
            prediction_status="error",
        )
        artifact_paths = sorted(path for path in session_dir.iterdir() if path.is_file())
        update_realtime_session_run_artifacts(
            database_path,
            session_hash,
            run_id,
            saved_file_names=[path.name for path in artifact_paths],
            saved_file_paths=[
                {
                    "field_name": _artifact_field_name(path.name),
                    "filename": path.name,
                    "path": str(path),
                }
                for path in artifact_paths
            ],
            archive_manifest=archive_manifest,
        )
        update_realtime_session_run_depression(
            database_path,
            session_hash,
            run_id,
            {
                "status": "error",
                "result": error_result,
                "error": str(exc),
                "completed_at": completed_at,
            },
        )
        print(
            f"[DEPRESSION] Failed | session={session_hash} | run={run_id} "
            f"| error={exc}",
            flush=True,
        )
        traceback.print_exc()
        return error_result


def queue_realtime_depression_detection(
    database_path: Path,
    session_hash: str,
    run_id: str,
    session_dir: Path,
) -> dict[str, Any]:
    if not is_depression_detection_enabled():
        update_realtime_session_run_depression(
            database_path,
            session_hash,
            run_id,
            {"status": "disabled"},
        )
        return {"status": "disabled"}

    queue = enqueue_depression_job(
        database_path,
        session_hash,
        run_id,
        Path(session_dir),
        max_attempts=int(os.environ.get("DEPRESSION_JOB_MAX_ATTEMPTS") or 3),
    )
    return {
        "status": queue["status"],
        "job_id": queue["id"],
        "queued_at": queue["queued_at"],
    }


def write_aspect_predictions_csv(result: dict[str, Any], output_path: Path) -> None:
    aspects = result.get("aspects")
    if not isinstance(aspects, list) or not aspects:
        return
    keys: list[str] = []
    for row in aspects:
        if isinstance(row, dict):
            for key in row:
                if key not in keys:
                    keys.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in aspects:
            if isinstance(row, dict):
                writer.writerow(row)


def _artifact_field_name(filename: str) -> str:
    return {
        RESULT_FILENAME: "depression_result_file",
        ERROR_FILENAME: "depression_error_file",
        ASPECT_RETRIEVAL_FILENAME: "depression_aspect_retrieval_file",
        ASPECT_PREDICTIONS_FILENAME: "depression_aspect_predictions_file",
        PARTICIPANT_TRANSCRIPT_FILENAME: "participant_transcript_file",
        PREPROCESSING_FILENAME: "depression_preprocessing_file",
        TRANSLATED_TRANSCRIPT_FILENAME: "depression_transcript_file",
        "metadata.json": "metadata_file",
        "transcript.json": "transcript_file",
        "transcript.txt": "transcript_text_file",
        "user_audio.wav": "user_audio_file",
        "assistant_audio.wav": "assistant_audio_file",
        "video_frames.zip": "video_frames_file",
        "archive_manifest.json": "archive_manifest_file",
    }.get(filename, "saved_file")
