from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat


def _archive_member_for_path(path: Path) -> tuple[Path, str] | None:
    parts = path.parts
    if "data" not in parts:
        return None
    try:
        data_index = parts.index("data")
        participant_name = parts[data_index + 1]
    except IndexError:
        return None
    if not participant_name.endswith("_P"):
        return None
    archive_path = Path(*parts[: data_index + 1]) / f"{participant_name}.tar.gz"
    member_parts = parts[data_index + 1 :]
    return archive_path, "/".join(member_parts)


def _loadmat_from_archive(path: Path) -> dict | None:
    archive_info = _archive_member_for_path(path)
    if archive_info is None:
        return None
    archive_path, member = archive_info
    if not archive_path.exists():
        return None
    with tarfile.open(archive_path, "r:gz") as tar:
        extracted = tar.extractfile(member)
        if extracted is None:
            return None
        return loadmat(BytesIO(extracted.read()))


def _read_bytes_from_archive(path: Path) -> bytes | None:
    archive_info = _archive_member_for_path(path)
    if archive_info is None:
        return None
    archive_path, member = archive_info
    if not archive_path.exists():
        return None
    with tarfile.open(archive_path, "r:gz") as tar:
        extracted = tar.extractfile(member)
        if extracted is None:
            return None
        return extracted.read()


def _read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, **kwargs)
    data = _read_bytes_from_archive(path)
    if data is None:
        return None
    return pd.read_csv(BytesIO(data), **kwargs)


def read_participant_intervals(
    transcript_path: str | Path,
    *,
    max_interval_seconds: float | None = 30.0,
) -> list[tuple[float, float]]:
    """Read participant speech intervals from an E-DAIC transcript CSV."""
    df = _read_csv(Path(transcript_path))
    if df is None or df.empty:
        return []
    if "Start_Time" not in df.columns or "End_Time" not in df.columns:
        return []
    starts = pd.to_numeric(df["Start_Time"], errors="coerce")
    ends = pd.to_numeric(df["End_Time"], errors="coerce")
    intervals: list[tuple[float, float]] = []
    for start, end in zip(starts, ends, strict=False):
        if not np.isfinite(start) or not np.isfinite(end):
            continue
        start = float(start)
        end = float(end)
        if end <= start:
            continue
        if max_interval_seconds is not None and end - start > max_interval_seconds:
            continue
        intervals.append((start, end))
    return intervals


def load_resnet_features(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.exists():
        mat = loadmat(path)
    else:
        mat = _loadmat_from_archive(path)
        if mat is None:
            return torch.zeros(1, 2048, dtype=torch.float32)
    array = mat.get("feature")
    if array is None:
        numeric_keys = [k for k in mat if not k.startswith("__")]
        if not numeric_keys:
            return torch.zeros(1, 2048, dtype=torch.float32)
        array = mat[numeric_keys[0]]
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] != 2048 and tensor.shape[0] == 2048:
        tensor = tensor.transpose(0, 1)
    return tensor.contiguous()


class FeatureExtractor:
    """HuBERT audio feature extraction with an on-disk cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        hubert_model_name: str = "facebook/hubert-base-ls960",
        device: str | torch.device | None = None,
        sample_rate: int = 16000,
        chunk_seconds: float = 30.0,
        participant_only: bool = False,
        max_interval_seconds: float | None = 30.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hubert_model_name = hubert_model_name
        self.sample_rate = sample_rate
        self.chunk_seconds = float(chunk_seconds)
        self.participant_only = bool(participant_only)
        self.max_interval_seconds = max_interval_seconds
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoFeatureExtractor, AutoModel

        self._processor = AutoFeatureExtractor.from_pretrained(self.hubert_model_name)
        self._model = AutoModel.from_pretrained(self.hubert_model_name).to(self.device)
        self._model.eval()

    def _cache_path(self, audio_path: str | Path) -> Path:
        audio_path = Path(audio_path)
        suffix = "_participant" if self.participant_only else ""
        return self.cache_dir / f"{audio_path.stem}{suffix}.pt"

    def load_or_extract_audio(
        self,
        audio_path: str | Path,
        transcript_path: str | Path | None = None,
    ) -> torch.Tensor:
        audio_path = Path(audio_path)
        cache_path = self._cache_path(audio_path)
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu").float()
        wav = self._load_audio(audio_path)
        if wav is None:
            return torch.zeros(1, 768, dtype=torch.float32)
        if self.participant_only and transcript_path is not None:
            masked = self._participant_waveform(wav, transcript_path)
            if masked.size:
                wav = masked
        self._lazy_load()
        features = self._extract_hubert_chunks(wav)
        torch.save(features, cache_path)
        return features

    def _participant_waveform(
        self,
        wav: np.ndarray,
        transcript_path: str | Path,
    ) -> np.ndarray:
        intervals = read_participant_intervals(
            transcript_path,
            max_interval_seconds=self.max_interval_seconds,
        )
        if not intervals:
            return np.asarray([], dtype=np.float32)
        pieces: list[np.ndarray] = []
        total = int(wav.shape[0])
        for start, end in intervals:
            start_idx = max(0, min(total, int(round(start * self.sample_rate))))
            end_idx = max(start_idx, min(total, int(round(end * self.sample_rate))))
            if end_idx > start_idx:
                pieces.append(wav[start_idx:end_idx])
        if not pieces:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32, copy=False)

    def _extract_hubert_chunks(self, wav: np.ndarray) -> torch.Tensor:
        chunk_size = max(1, int(self.sample_rate * self.chunk_seconds))
        chunks: list[torch.Tensor] = []
        for start in range(0, len(wav), chunk_size):
            chunk = wav[start : start + chunk_size]
            if chunk.size == 0:
                continue
            inputs = self._processor(
                chunk,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = self._model(**inputs)
            chunks.append(outputs.last_hidden_state.squeeze(0).detach().cpu().float())
            del inputs, outputs
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        if not chunks:
            return torch.zeros(1, 768, dtype=torch.float32)
        return torch.cat(chunks, dim=0)

    def _load_audio(self, audio_path: Path) -> np.ndarray | None:
        audio_source: Path | BytesIO
        if audio_path.exists():
            audio_source = audio_path
        else:
            audio_bytes = _read_bytes_from_archive(audio_path)
            if audio_bytes is None:
                return None
            audio_source = BytesIO(audio_bytes)
        try:
            import soundfile as sf
            from scipy.signal import resample_poly

            wav, sr = sf.read(audio_source, dtype="float32", always_2d=False)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            if sr != self.sample_rate:
                gcd = np.gcd(sr, self.sample_rate)
                wav = resample_poly(wav, self.sample_rate // gcd, sr // gcd).astype(
                    np.float32
                )
            return np.asarray(wav, dtype=np.float32)
        except Exception:
            import librosa

            if isinstance(audio_source, BytesIO):
                audio_source.seek(0)
            wav, _ = librosa.load(audio_source, sr=self.sample_rate, mono=True)
            return np.asarray(wav, dtype=np.float32)
