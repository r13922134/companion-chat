from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


PREPROCESSING_FILENAME = "depression_preprocessing.json"
TRANSLATED_TRANSCRIPT_FILENAME = "transcript_depression_english.json"
TRANSLATED_TRANSCRIPT_TEXT_FILENAME = "transcript_depression_english.txt"
DEFAULT_TRANSLATION_MODEL = "gpt-5.4-mini"
CJK_RE = re.compile(r"[\u3400-\u9fff]")


def read_json_file(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def write_json_file(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def env_positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(str(os.environ.get(name) or default).strip()))
    except Exception:
        return default


def translation_model() -> str:
    return str(
        os.environ.get("OPENAI_DEPRESSION_TRANSLATION_MODEL")
        or os.environ.get("DEPRESSION_TRANSLATION_MODEL")
        or DEFAULT_TRANSLATION_MODEL
    ).strip()


def openai_key_missing_message() -> str:
    return "OpenAI API key not found for depression transcript translation."


def post_openai_json(api_path: str, payload: dict, timeout: int = 60) -> dict:
    api_key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(openai_key_missing_message())
    request_obj = urlrequest.Request(
        "https://api.openai.com/v1/" + api_path.lstrip("/"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(request_obj, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"OpenAI translation request failed: {exc.code} {body[:500]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"OpenAI translation request failed: {exc.reason}"
        ) from exc


def transcript_user_event_rows(transcript: dict) -> list[dict]:
    events = transcript.get("events") if isinstance(transcript, dict) else None
    if not isinstance(events, list):
        return []
    rows = []
    utterance_index = 1
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        if str(event.get("speaker") or "").strip().lower() != "user":
            continue
        text = " ".join(str(event.get("text") or "").split())
        if not text:
            continue
        rows.append(
            {
                "event_index": event_index,
                "utterance_index": utterance_index,
                "text": text,
            }
        )
        utterance_index += 1
    return rows


def parse_translation_json(content: str) -> dict:
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "", 1).replace("```", "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("translations"), list):
        raise RuntimeError("Translation response missing translations array.")
    return parsed


def translate_user_utterance_chunk(
    rows: list[dict],
    model: str,
) -> dict[int, str]:
    payload_rows = [
        {"index": int(row["utterance_index"]), "text": str(row["text"])}
        for row in rows
    ]
    user_prompt = (
        "Return a JSON object with this exact shape: "
        '{"translations":[{"index":1,"text":"..."}]}. '
        "Translate each item independently and keep the same index values.\n\n"
        f"{json.dumps({'utterances': payload_rows}, ensure_ascii=False)}"
    )
    response = post_openai_json(
        "/chat/completions",
        {
            "model": model,
            "max_completion_tokens": max(
                600,
                min(12000, len(user_prompt) // 2 + 800),
            ),
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Translate participant utterances from Traditional Chinese "
                        "or Mandarin into natural clinical English for a PHQ-8 "
                        "depression screening model. Preserve first-person meaning, "
                        "negation, uncertainty, frequency, duration, intensity, "
                        "temporal references, and symptom wording. Do not add "
                        "diagnoses, scores, interpretations, or advice. Return only JSON."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        },
    )
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("Translation response has no choices.")
    parsed = parse_translation_json(
        str((choices[0].get("message") or {}).get("content") or "")
    )
    translated: dict[int, str] = {}
    for item in parsed["translations"]:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        text = " ".join(str(item.get("text") or "").split())
        if text:
            translated[index] = text
    return translated


def build_depression_transcript_text(transcript: dict) -> str:
    lines = []
    for event in transcript.get("events") or []:
        if not isinstance(event, dict):
            continue
        speaker = str(event.get("speaker") or "").strip().lower()
        text = " ".join(str(event.get("text") or "").split())
        if speaker in {"user", "assistant"} and text:
            lines.append(f"{speaker.upper()}: {text}")
    return "\n".join(lines)


def prepare_depression_translation_artifacts(
    session_dir: Path,
    transcript: dict | None,
) -> tuple[dict, list[dict]]:
    session_dir = Path(session_dir)
    preprocessing_path = session_dir / PREPROCESSING_FILENAME
    translated_path = session_dir / TRANSLATED_TRANSCRIPT_FILENAME
    existing = read_json_file(preprocessing_path, default={})
    translation = existing.get("translation") if isinstance(existing, dict) else None
    if (
        translated_path.is_file()
        and isinstance(translation, dict)
        and translation.get("status") == "ok"
    ):
        return existing, []

    preprocessing = {
        "translation": {
            "status": "skipped",
            "enabled": env_flag("DEPRESSION_TRANSLATION_ENABLED", default=True),
            "model": translation_model(),
            "source_language": "zh",
            "target_language": "en",
            "artifact": None,
            "message": "",
        }
    }
    created_files: list[dict] = []

    def record(path: Path, field_name: str) -> None:
        created_files.append(
            {
                "field_name": field_name,
                "filename": path.name,
                "path": str(path),
            }
        )

    def finish(status: str, message: str = "") -> tuple[dict, list[dict]]:
        preprocessing["translation"]["status"] = status
        preprocessing["translation"]["message"] = message
        write_json_file(preprocessing_path, preprocessing)
        record(preprocessing_path, "depression_preprocessing_file")
        return preprocessing, created_files

    if not isinstance(transcript, dict):
        return finish("skipped_no_transcript", "No transcript JSON was uploaded.")
    rows = transcript_user_event_rows(transcript)
    if not rows:
        return finish("skipped_no_user_utterances", "No user utterances were available.")
    if not env_flag("DEPRESSION_TRANSLATION_ENABLED", default=True):
        return finish("disabled", "DEPRESSION_TRANSLATION_ENABLED is disabled.")
    if not any(CJK_RE.search(str(row["text"])) for row in rows):
        return finish("skipped_already_english", "No CJK user text detected.")
    if not str(os.environ.get("OPENAI_API_KEY") or "").strip():
        return finish("skipped_no_api_key", openai_key_missing_message())

    model = translation_model()
    try:
        translated_by_index: dict[int, str] = {}
        chunk_size = env_positive_int("DEPRESSION_TRANSLATION_CHUNK_SIZE", 40)
        for start in range(0, len(rows), chunk_size):
            translated_by_index.update(
                translate_user_utterance_chunk(rows[start : start + chunk_size], model)
            )
        missing = [
            int(row["utterance_index"])
            for row in rows
            if int(row["utterance_index"]) not in translated_by_index
        ]
        if missing:
            raise RuntimeError(f"Translation missing utterance indices: {missing[:10]}")

        translated_transcript = copy.deepcopy(transcript)
        events = translated_transcript.get("events")
        assert isinstance(events, list)
        for row in rows:
            event = events[int(row["event_index"])]
            if not isinstance(event, dict):
                continue
            event["text_original"] = str(row["text"])
            event["text"] = translated_by_index[int(row["utterance_index"])]
            event["translation_source_language"] = "zh"
            event["translation_target_language"] = "en"
            event["translation_model"] = model
        translated_transcript["translation"] = {
            "status": "ok",
            "model": model,
            "source_language": "zh",
            "target_language": "en",
            "user_utterance_count": len(rows),
        }
        write_json_file(translated_path, translated_transcript)
        record(translated_path, "depression_transcript_file")

        translated_text_path = session_dir / TRANSLATED_TRANSCRIPT_TEXT_FILENAME
        translated_text_path.write_text(
            build_depression_transcript_text(translated_transcript),
            encoding="utf-8",
        )
        record(translated_text_path, "depression_transcript_text_file")
        preprocessing["translation"].update(
            {
                "status": "ok",
                "artifact": TRANSLATED_TRANSCRIPT_FILENAME,
                "user_utterance_count": len(rows),
                "message": "Translated user utterances for depression detection.",
            }
        )
        write_json_file(preprocessing_path, preprocessing)
        record(preprocessing_path, "depression_preprocessing_file")
        return preprocessing, created_files
    except Exception as exc:
        return finish("error", str(exc))
