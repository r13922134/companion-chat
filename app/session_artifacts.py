from __future__ import annotations

from pathlib import Path
from typing import Iterable


INPUT_DIR = "input"
MEDIA_DIR = "media"
ANALYSIS_DIR = "analysis"

ARTIFACT_DIRECTORIES = {
    "metadata.json": INPUT_DIR,
    "transcript.json": INPUT_DIR,
    "user_audio.wav": MEDIA_DIR,
    "assistant_audio.wav": MEDIA_DIR,
    "video_frames.zip": MEDIA_DIR,
    "archive_manifest.json": ANALYSIS_DIR,
    "depression_result.json": ANALYSIS_DIR,
    "depression_error.json": ANALYSIS_DIR,
    "depression_aspect_retrieval.jsonl": ANALYSIS_DIR,
    "depression_preprocessing.json": ANALYSIS_DIR,
    "transcript_depression_english.json": ANALYSIS_DIR,
    "user_speech_intervals.csv": ANALYSIS_DIR,
    "participant_utterances.jsonl": ANALYSIS_DIR,
}


def artifact_relative_path(filename: str) -> Path:
    directory = ARTIFACT_DIRECTORIES.get(filename)
    return Path(directory) / filename if directory else Path(filename)


def artifact_write_path(session_dir: Path, filename: str) -> Path:
    return Path(session_dir) / artifact_relative_path(filename)


def artifact_read_path(session_dir: Path, filename: str) -> Path:
    session_dir = Path(session_dir)
    organized_path = artifact_write_path(session_dir, filename)
    if organized_path.is_file():
        return organized_path
    legacy_path = session_dir / filename
    if legacy_path.is_file():
        return legacy_path
    return organized_path


def ensure_artifact_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def iter_artifact_files(session_dir: Path) -> Iterable[Path]:
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        return []
    files = []
    for path in session_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(session_dir)
        if any(part.startswith(".") for part in relative.parts):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(session_dir).as_posix())


def artifact_record(session_dir: Path, path: Path, field_name: str) -> dict:
    try:
        relative_path = Path(path).relative_to(session_dir).as_posix()
    except ValueError:
        relative_path = Path(path).name
    return {
        "field_name": field_name,
        "filename": Path(path).name,
        "relative_path": relative_path,
        "path": str(path),
    }
