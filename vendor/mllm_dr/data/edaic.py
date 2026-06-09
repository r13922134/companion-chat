from __future__ import annotations

import tarfile
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mllm_dr.prompts import PHQ8_ASPECTS, build_dialogue

from .features import (
    FeatureExtractor,
    load_resnet_features,
    read_participant_intervals,
)

ALT_DETAIL_COLUMNS = {
    "PHQ_8NoInterest": "PHQ8_1_NoInterest",
    "PHQ_8Depressed": "PHQ8_2_Depressed",
    "PHQ_8Sleep": "PHQ8_3_Sleep",
    "PHQ_8Tired": "PHQ8_4_Tired",
    "PHQ_8Appetite": "PHQ8_5_Appetite",
    "PHQ_8Failure": "PHQ8_6_Failure",
    "PHQ_8Concentrating": "PHQ8_7_Concentration",
    "PHQ_8Moving": "PHQ8_8_Psychomotor",
}

DEFAULT_TRANSCRIPT_CLEANUP_CONFIG: dict[str, Any] = {
    "enabled": False,
    "min_confidence": 0.8,
    "max_low_confidence_words": 3,
    "max_filler_words": 3,
    "max_prompt_fragment_words": 10,
    "drop_short_fillers": True,
    "drop_low_confidence_fragments": True,
    "drop_prompt_fragments": True,
}

SHORT_FILLERS = {
    "ah",
    "alright",
    "bye",
    "goodbye",
    "hey",
    "hm",
    "hmm",
    "no",
    "nope",
    "now",
    "ok",
    "okay",
    "sure",
    "thanks",
    "thank you",
    "uh",
    "um",
    "yeah",
    "yep",
    "yes",
}

PROMPT_FRAGMENT_RE = re.compile(
    r"^\s*(?:"
    r"are\s+you\s+okay(?:\s+with\s+this)?|"
    r"can\s+you\s+tell\s+me|"
    r"could\s+you\s+tell\s+me|"
    r"do\s+you\s+|"
    r"have\s+you\s+ever|"
    r"how\s+(?:are|do|did|would|easy|long)\s+|"
    r"tell\s+me\s+about|"
    r"thanks?\s+for\s+sharing|"
    r"what(?:'s|\s+is|\s+are|\s+was)\s+|"
    r"when\s+(?:is|was|did)\s+|"
    r"where\s+(?:are|did|do|were)\s+|"
    r"would\s+you\s+"
    r")",
    re.IGNORECASE,
)

CONTENTFUL_SHORT_RE = re.compile(
    r"\b(?:"
    r"adhd|angry|anxious|appetite|asleep|blame|concentrat|confus|"
    r"depress|difficult|down|eat|edgy|energy|failure|fatigue|focus|"
    r"guilt|guilty|grumpy|hard|hopeless|irritable|joy|letharg|"
    r"lonely|negative|pain|ptsd|rage|regret|restless|sad|shy|sleep|"
    r"slow|stress|tired|weight|worth"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TranscriptUtterance:
    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class AspectRetrievalRecord:
    participant_id: int
    aspect_index: int
    dialogue: str
    split: str | None = None
    aspect: str | None = None
    query: str | None = None
    profile: str | None = None
    utterance_indices: tuple[int, ...] = ()
    scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class EDaicSample:
    participant_id: int
    split: str
    aspect_index: int
    aspect_key: str
    aspect: str
    label: int
    total_score: int
    binary_label: int
    dialogue: str
    transcript_path: Path
    audio_path: Path
    video_path: Path
    rationale: str | None = None
    retrieval_query: str | None = None
    retrieval_profile: str | None = None
    retrieved_utterance_indices: tuple[int, ...] = ()
    retrieval_scores: tuple[float, ...] = ()
    teacher_score_probs: tuple[float, ...] = ()


def _participant_dir(root: Path, participant_id: int) -> Path:
    return root / "data" / f"{int(participant_id)}_P"


def _archive_for_member_path(path: Path) -> tuple[Path, str] | None:
    participant_dir = path.parent
    if not participant_dir.name.endswith("_P"):
        return None
    archive_path = participant_dir.with_suffix(".tar.gz")
    return archive_path, f"{participant_dir.name}/{path.name}"


def _transcript_cleanup_config(value: bool | dict[str, Any] | None) -> dict[str, Any]:
    config = dict(DEFAULT_TRANSCRIPT_CLEANUP_CONFIG)
    if isinstance(value, bool):
        config["enabled"] = value
    elif isinstance(value, dict):
        config.update(value)
    return config


def _normalized_words(text: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9']+", " ", text.lower()).strip()
    return [word for word in normalized.split() if word]


def _is_short_filler(text: str, max_words: int) -> bool:
    words = _normalized_words(text)
    if not words or len(words) > max_words:
        return False
    normalized = " ".join(words)
    return normalized in SHORT_FILLERS or all(word in SHORT_FILLERS for word in words)


def _is_low_confidence_fragment(
    text: str,
    confidence: float | None,
    min_confidence: float,
    max_words: int,
) -> bool:
    if confidence is None or confidence >= min_confidence:
        return False
    words = _normalized_words(text)
    if len(words) > max_words:
        return False
    return not CONTENTFUL_SHORT_RE.search(text)


def _is_prompt_fragment(text: str, max_words: int) -> bool:
    words = _normalized_words(text)
    if len(words) > max_words:
        return False
    if CONTENTFUL_SHORT_RE.search(text):
        return False
    return PROMPT_FRAGMENT_RE.search(text) is not None


def _clean_transcript_rows(
    rows: list[tuple[int, float, float, str, float | None]],
    cleanup: bool | dict[str, Any] | None,
) -> list[tuple[int, float, float, str]]:
    config = _transcript_cleanup_config(cleanup)
    if not config.get("enabled", False):
        return [(index, start, end, text) for index, start, end, text, _ in rows]

    min_confidence = float(config.get("min_confidence", 0.8))
    max_low_confidence_words = int(config.get("max_low_confidence_words", 3))
    max_filler_words = int(config.get("max_filler_words", 3))
    max_prompt_fragment_words = int(config.get("max_prompt_fragment_words", 10))
    drop_regexes = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in config.get("drop_regexes", [])
    ]

    cleaned: list[tuple[int, float, float, str]] = []
    for index, start, end, text, confidence in rows:
        if (
            config.get("drop_short_fillers", True)
            and _is_short_filler(text, max_filler_words)
        ):
            continue
        if (
            config.get("drop_low_confidence_fragments", True)
            and _is_low_confidence_fragment(
                text,
                confidence,
                min_confidence,
                max_low_confidence_words,
            )
        ):
            continue
        if (
            config.get("drop_prompt_fragments", True)
            and _is_prompt_fragment(text, max_prompt_fragment_words)
        ):
            continue
        if any(pattern.search(text) for pattern in drop_regexes):
            continue
        cleaned.append((index, start, end, text))
    return cleaned


def read_transcript_utterances(
    path: Path,
    transcript_cleanup: bool | dict[str, Any] | None = None,
) -> list[TranscriptUtterance]:
    if path.exists():
        df = pd.read_csv(path)
    else:
        archive_info = _archive_for_member_path(path)
        if archive_info is None:
            return []
        archive_path, member = archive_info
        if not archive_path.exists():
            return []
        with tarfile.open(archive_path, "r:gz") as tar:
            extracted = tar.extractfile(member)
            if extracted is None:
                return []
            df = pd.read_csv(extracted)
    if df.empty:
        return []
    rows: list[tuple[int, float, float, str, float | None]] = []
    for index, row in enumerate(df.to_dict("records"), start=1):
        text = row.get("Text", "")
        if pd.isna(text):
            continue
        confidence = row.get("Confidence")
        if confidence is not None and not pd.isna(confidence):
            confidence = float(confidence)
        else:
            confidence = None
        rows.append(
            (
                index,
                float(row.get("Start_Time", 0.0)),
                float(row.get("End_Time", 0.0)),
                str(text),
                confidence,
            )
        )
    cleaned_rows = _clean_transcript_rows(rows, transcript_cleanup)
    return [
        TranscriptUtterance(index=index, start=start, end=end, text=text)
        for index, start, end, text in cleaned_rows
    ]


def _build_dialogue_from_utterances(utterances: list[TranscriptUtterance]) -> str:
    return build_dialogue(
        [(row.start, row.end, row.text) for row in utterances],
        utterance_indices=[row.index for row in utterances],
    )


def _read_transcript(
    path: Path,
    transcript_cleanup: bool | dict[str, Any] | None = None,
) -> str:
    return _build_dialogue_from_utterances(
        read_transcript_utterances(path, transcript_cleanup=transcript_cleanup)
    )


def _load_rationales(path: str | Path | None) -> dict[tuple[int, int], str]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    rationales: dict[tuple[int, int], str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            key = (int(item["participant_id"]), int(item["aspect_index"]))
            rationales[key] = str(item.get("rationale", ""))
    return rationales


def _load_teacher_score_probs(
    path: str | Path | None,
) -> dict[tuple[int, int], tuple[float, ...]]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    rows: dict[tuple[int, int], tuple[float, ...]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            values = item.get("teacher_score_probs")
            if values is None:
                values = item.get("score_probs")
            if values is None:
                continue
            probs = tuple(float(value) for value in values)
            if len(probs) != 4:
                continue
            key = (int(item["participant_id"]), int(item["aspect_index"]))
            rows[key] = probs
    return rows


def _aspect_retrieval_config(
    value: str | Path | dict[str, Any] | None,
) -> tuple[Path | None, bool]:
    if value is None:
        return None, False
    if isinstance(value, (str, Path)):
        return Path(value), False
    path = value.get("path")
    return Path(path) if path else None, bool(value.get("require", False))


def _load_aspect_retrieval(
    path: str | Path | None,
) -> dict[tuple[str, int, int], AspectRetrievalRecord]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    records: dict[tuple[str, int, int], AspectRetrievalRecord] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            participant_id = int(item["participant_id"])
            aspect_index = int(item["aspect_index"])
            split = item.get("split")
            dialogue = str(item.get("dialogue") or "")
            utterance_indices = tuple(
                int(index) for index in item.get("utterance_indices", [])
            )
            scores = tuple(float(score) for score in item.get("scores", []))
            record = AspectRetrievalRecord(
                participant_id=participant_id,
                aspect_index=aspect_index,
                split=str(split) if split else None,
                aspect=item.get("aspect"),
                dialogue=dialogue,
                query=item.get("query"),
                profile=item.get("profile"),
                utterance_indices=utterance_indices,
                scores=scores,
            )
            records[(str(split or ""), participant_id, aspect_index)] = record
    return records


def _lookup_aspect_retrieval(
    records: dict[tuple[str, int, int], AspectRetrievalRecord],
    split: str,
    participant_id: int,
    aspect_index: int,
) -> AspectRetrievalRecord | None:
    return records.get((split, participant_id, aspect_index)) or records.get(
        ("", participant_id, aspect_index)
    )


def _temporal_cue_summary_config(
    value: bool | dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"enabled": value, "segments": 3, "max_tokens": 256}
    if isinstance(value, dict):
        config = {"enabled": False, "segments": 3, "max_tokens": 256}
        config.update(value)
        return config
    return {"enabled": False, "segments": 3, "max_tokens": 256}


def _segment_ranges(length: int, segments: int) -> list[tuple[int, int]]:
    length = max(0, int(length))
    segments = max(1, int(segments))
    ranges: list[tuple[int, int]] = []
    for idx in range(segments):
        start = int(round(idx * length / segments))
        end = int(round((idx + 1) * length / segments))
        ranges.append((min(start, length), min(max(end, start + 1), length)))
    return ranges


def _mean_norm(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(tensor.float(), dim=-1).mean().item())


def _mean_delta_norm(tensor: torch.Tensor) -> float:
    if tensor.shape[0] < 2:
        return 0.0
    delta = tensor.float()[1:] - tensor.float()[:-1]
    return float(torch.linalg.vector_norm(delta, dim=-1).mean().item())


def temporal_cue_summary_text(
    *,
    sample: EDaicSample,
    audio: torch.Tensor,
    video: torch.Tensor,
    segments: int = 3,
    max_tokens: int = 256,
) -> str:
    """Summarize only HuBERT, transcript timing, and ResNet temporal statistics."""
    segments = max(1, int(segments))
    max_tokens = max(24, int(max_tokens))
    intervals = read_participant_intervals(sample.transcript_path)
    speech_seconds = sum(max(0.0, end - start) for start, end in intervals)
    session_end = max((end for _, end in intervals), default=0.0)
    speech_ratio = speech_seconds / session_end if session_end > 0 else 0.0
    names = ["early", "middle", "late"]
    if segments != 3:
        names = [f"segment {idx + 1}" for idx in range(segments)]
    audio_ranges = _segment_ranges(int(audio.shape[0]), segments)
    video_ranges = _segment_ranges(int(video.shape[0]), segments)
    parts = [
        (
            "Temporal cue summary from HuBERT participant-speech intervals and "
            f"ResNet frame features: participant speech covers {speech_seconds:.1f}s "
            f"across {len(intervals)} intervals, ratio {speech_ratio:.2f}."
        )
    ]
    for idx in range(segments):
        audio_start, audio_end = audio_ranges[idx]
        video_start, video_end = video_ranges[idx]
        audio_segment = audio[audio_start:audio_end]
        video_segment = video[video_start:video_end]
        name = names[idx] if idx < len(names) else f"segment {idx + 1}"
        parts.append(
            f"{name}: HuBERT activation {_mean_norm(audio_segment):.3f}, "
            f"ResNet activation {_mean_norm(video_segment):.3f}, "
            f"ResNet temporal delta {_mean_delta_norm(video_segment):.3f}."
        )
    words = " ".join(parts).split()
    if len(words) > max_tokens:
        words = words[:max_tokens]
    return " ".join(words)


def append_temporal_cue_summary(
    dialogue: str,
    *,
    sample: EDaicSample,
    audio: torch.Tensor,
    video: torch.Tensor,
    config: bool | dict[str, Any] | None,
) -> str:
    summary_cfg = _temporal_cue_summary_config(config)
    if not bool(summary_cfg.get("enabled", False)):
        return dialogue
    summary = temporal_cue_summary_text(
        sample=sample,
        audio=audio,
        video=video,
        segments=int(summary_cfg.get("segments", 3)),
        max_tokens=int(summary_cfg.get("max_tokens", 256)),
    )
    if not summary:
        return dialogue
    return f"{dialogue}\n\nTemporal cue summary:\n{summary}"


class EDaicDataset(Dataset[EDaicSample]):
    """E-DAIC-WOZ PHQ-8 aspect-level dataset.

    The paper treats each participant as eight aspect-level training samples.
    For E-DAIC, each aspect uses the full interview transcript because the
    questions do not map cleanly to PHQ-8 items.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        rationale_path: str | Path | None = None,
        max_participants: int | None = None,
        exclude_participants: list[int] | tuple[int, ...] | set[int] | None = None,
        transcript_cleanup: bool | dict[str, Any] | None = None,
        aspect_retrieval: str | Path | dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        split_file = self.root / "labels" / f"{split}_split.csv"
        detail_file = self.root / "labels" / "Detailed_PHQ8_Labels.csv"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        if not detail_file.exists():
            raise FileNotFoundError(f"Missing detailed PHQ-8 labels: {detail_file}")

        split_df = pd.read_csv(split_file)
        excluded = {int(pid) for pid in (exclude_participants or [])}
        if excluded:
            split_df = split_df[
                ~split_df["Participant_ID"].astype(int).isin(excluded)
            ]
        detail_df = pd.read_csv(detail_file).set_index("Participant_ID")
        alt_detail_file = self.root / "labels" / "detailed_lables.csv"
        alt_detail_df = None
        if alt_detail_file.exists():
            alt_detail_df = pd.read_csv(alt_detail_file).set_index("Participant")
        if max_participants:
            split_df = split_df.head(int(max_participants))
        rationales = _load_rationales(rationale_path)
        teacher_score_probs = _load_teacher_score_probs(rationale_path)
        aspect_retrieval_path, require_aspect_retrieval = _aspect_retrieval_config(
            aspect_retrieval
        )
        aspect_retrieval_records = _load_aspect_retrieval(aspect_retrieval_path)
        if require_aspect_retrieval and not aspect_retrieval_records:
            raise FileNotFoundError(
                f"Missing required aspect retrieval cache: {aspect_retrieval_path}"
            )

        samples: list[EDaicSample] = []
        for _, split_row in split_df.iterrows():
            pid = int(split_row["Participant_ID"])
            has_detail = pid in detail_df.index
            has_alt_detail = alt_detail_df is not None and pid in alt_detail_df.index
            if not has_detail and not has_alt_detail:
                continue
            detail = detail_df.loc[pid] if has_detail else alt_detail_df.loc[pid]
            pdir = _participant_dir(self.root, pid)
            transcript_path = pdir / f"{pid}_Transcript.csv"
            transcript = _read_transcript(
                transcript_path,
                transcript_cleanup=transcript_cleanup,
            )
            audio_path = pdir / f"{pid}_AUDIO.wav"
            video_path = pdir / "features" / f"{pid}_CNN_ResNet.mat"
            total_score = int(
                detail["PHQ_8Total"]
                if "PHQ_8Total" in detail.index
                else detail["Depression_severity"]
            )
            binary_label = int(split_row.get("PHQ_Binary", int(total_score >= 10)))
            for aspect_index, (aspect_key, aspect) in enumerate(PHQ8_ASPECTS):
                column = aspect_key if aspect_key in detail.index else ALT_DETAIL_COLUMNS[aspect_key]
                label = int(detail[column])
                retrieval = _lookup_aspect_retrieval(
                    aspect_retrieval_records,
                    split=split,
                    participant_id=pid,
                    aspect_index=aspect_index,
                )
                if require_aspect_retrieval and retrieval is None:
                    raise KeyError(
                        "Missing required aspect retrieval row for "
                        f"split={split!r}, participant_id={pid}, "
                        f"aspect_index={aspect_index}"
                    )
                samples.append(
                    EDaicSample(
                        participant_id=pid,
                        split=split,
                        aspect_index=aspect_index,
                        aspect_key=aspect_key,
                        aspect=aspect,
                        label=label,
                        total_score=total_score,
                        binary_label=binary_label,
                        dialogue=retrieval.dialogue if retrieval else transcript,
                        transcript_path=transcript_path,
                        audio_path=audio_path,
                        video_path=video_path,
                        rationale=rationales.get((pid, aspect_index)),
                        retrieval_query=retrieval.query if retrieval else None,
                        retrieval_profile=retrieval.profile if retrieval else None,
                        retrieved_utterance_indices=(
                            retrieval.utterance_indices if retrieval else ()
                        ),
                        retrieval_scores=retrieval.scores if retrieval else (),
                        teacher_score_probs=teacher_score_probs.get(
                            (pid, aspect_index),
                            (),
                        ),
                    )
                )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EDaicSample:
        return self.samples[index]


class EDaicFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset: EDaicDataset,
        feature_extractor: FeatureExtractor | None = None,
        max_audio_frames: int | None = None,
        max_video_frames: int | None = None,
        feature_backend: str = "hubert_resnet",
        audio_feature_dim: int = 768,
        video_feature_dim: int = 2048,
        cache_features_in_memory: bool = False,
        temporal_cue_summary: bool | dict[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.max_audio_frames = max_audio_frames
        self.max_video_frames = max_video_frames
        self.feature_backend = feature_backend
        self.audio_feature_dim = int(audio_feature_dim)
        self.video_feature_dim = int(video_feature_dim)
        self.cache_features_in_memory = cache_features_in_memory
        self.temporal_cue_summary = _temporal_cue_summary_config(temporal_cue_summary)
        self._feature_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        cached = self._feature_cache.get(sample.participant_id)
        if self.cache_features_in_memory and cached is not None:
            audio, video = cached
        else:
            if self.feature_backend != "hubert_resnet":
                raise ValueError(
                    f"Unsupported feature backend after cleanup: {self.feature_backend}"
                )
            if self.feature_extractor is None:
                audio = torch.zeros(1, self.audio_feature_dim, dtype=torch.float32)
            else:
                audio = self.feature_extractor.load_or_extract_audio(
                    sample.audio_path,
                    transcript_path=sample.transcript_path,
                )
            video = load_resnet_features(sample.video_path)
            if self.max_audio_frames:
                audio = temporal_uniform_sample(audio, self.max_audio_frames)
            if self.max_video_frames:
                video = temporal_uniform_sample(video, self.max_video_frames)
            if audio.numel() == 0:
                audio = torch.zeros(1, self.audio_feature_dim, dtype=torch.float32)
            if video.numel() == 0:
                video = torch.zeros(1, self.video_feature_dim, dtype=torch.float32)
            if self.cache_features_in_memory:
                self._feature_cache[sample.participant_id] = (audio, video)
        dialogue = append_temporal_cue_summary(
            sample.dialogue,
            sample=sample,
            audio=audio,
            video=video,
            config=self.temporal_cue_summary,
        )
        teacher_score_probs = (
            torch.tensor(sample.teacher_score_probs, dtype=torch.float32)
            if sample.teacher_score_probs
            else torch.zeros(4, dtype=torch.float32)
        )
        return {
            "sample": sample,
            "participant_id": sample.participant_id,
            "aspect_index": sample.aspect_index,
            "aspect": sample.aspect,
            "dialogue": dialogue,
            "label": torch.tensor(float(sample.label), dtype=torch.float32),
            "total_score": torch.tensor(float(sample.total_score), dtype=torch.float32),
            "binary_label": torch.tensor(int(sample.binary_label), dtype=torch.long),
            "rationale": sample.rationale,
            "retrieval_query": sample.retrieval_query,
            "retrieval_profile": sample.retrieval_profile,
            "retrieved_utterance_indices": sample.retrieved_utterance_indices,
            "retrieval_scores": sample.retrieval_scores,
            "teacher_score_probs": teacher_score_probs,
            "teacher_score_prob_mask": torch.tensor(
                bool(sample.teacher_score_probs),
                dtype=torch.bool,
            ),
            "audio_features": audio,
            "video_features": video,
        }


def temporal_uniform_sample(tensor: torch.Tensor, max_frames: int) -> torch.Tensor:
    """Keep coverage of the whole interview instead of only the first frames."""
    if tensor.shape[0] <= max_frames:
        return tensor
    indices = torch.linspace(0, tensor.shape[0] - 1, steps=max_frames).long()
    return tensor.index_select(0, indices)


def pad_sequence_features(
    tensors: list[torch.Tensor],
    feature_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(tensors)
    max_len = max([int(t.shape[0]) for t in tensors] + [1])
    out = torch.zeros(batch, max_len, feature_dim, dtype=torch.float32)
    mask = torch.zeros(batch, max_len, dtype=torch.bool)
    for i, tensor in enumerate(tensors):
        if tensor.numel() == 0:
            continue
        length = min(tensor.shape[0], max_len)
        width = min(tensor.shape[1], feature_dim)
        out[i, :length, :width] = tensor[:length, :width].float()
        mask[i, :length] = True
    return out, mask


def collate_feature_batch(
    batch: list[dict[str, Any]],
    audio_feature_dim: int = 768,
    video_feature_dim: int = 2048,
) -> dict[str, Any]:
    audio, audio_mask = pad_sequence_features(
        [item["audio_features"] for item in batch], feature_dim=int(audio_feature_dim)
    )
    video, video_mask = pad_sequence_features(
        [item["video_features"] for item in batch], feature_dim=int(video_feature_dim)
    )
    out = {
        "samples": [item["sample"] for item in batch],
        "participant_id": torch.tensor(
            [item["participant_id"] for item in batch], dtype=torch.long
        ),
        "aspect_index": torch.tensor(
            [item["aspect_index"] for item in batch], dtype=torch.long
        ),
        "aspects": [item["aspect"] for item in batch],
        "dialogues": [item["dialogue"] for item in batch],
        "rationales": [item["rationale"] for item in batch],
        "retrieval_queries": [item.get("retrieval_query") for item in batch],
        "retrieval_profiles": [item.get("retrieval_profile") for item in batch],
        "retrieved_utterance_indices": [
            item.get("retrieved_utterance_indices", ()) for item in batch
        ],
        "retrieval_scores": [item.get("retrieval_scores", ()) for item in batch],
        "teacher_score_probs": torch.stack(
            [
                item.get(
                    "teacher_score_probs",
                    torch.zeros(4, dtype=torch.float32),
                )
                for item in batch
            ]
        ),
        "teacher_score_prob_mask": torch.stack(
            [
                item.get(
                    "teacher_score_prob_mask",
                    torch.tensor(False, dtype=torch.bool),
                )
                for item in batch
            ]
        ),
        "score_labels": torch.stack([item["label"] for item in batch]),
        "total_scores": torch.stack([item["total_score"] for item in batch]),
        "binary_labels": torch.stack([item["binary_label"] for item in batch]),
        "audio_features": audio,
        "audio_mask": audio_mask,
        "video_features": video,
        "video_mask": video_mask,
    }
    return out


def group_aspect_predictions(
    participant_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gold: dict[int, float] = {}
    pred: dict[int, float] = {}
    for pid, y, yhat in zip(participant_ids, labels, predictions, strict=False):
        gold[int(pid)] = gold.get(int(pid), 0.0) + float(y)
        pred[int(pid)] = pred.get(int(pid), 0.0) + float(yhat)
    ordered = sorted(gold)
    return (
        np.array([gold[pid] for pid in ordered], dtype=np.float32),
        np.array([pred[pid] for pid in ordered], dtype=np.float32),
    )
