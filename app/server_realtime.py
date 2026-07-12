from pathlib import Path
from uuid import uuid4
from datetime import datetime
import hashlib
import json
import os
import re
import shutil
import time
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from typing import List, Dict, Optional, Tuple

import traceback
from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from werkzeug.utils import secure_filename
try:
    from .storage import (
        get_realtime_session,
        get_realtime_session_run,
        get_depression_job,
        attach_latest_google_form_response_to_metadata,
        google_form_account_key,
        initialize_database,
        insert_google_form_response,
        list_depression_workers,
        list_google_form_response_summaries,
        depression_queue_counts,
        resolve_database_path,
        update_realtime_session_run_depression,
        update_realtime_session_run_memory,
        upsert_realtime_session_run,
    )
    from .depression_detector import (
        PREPROCESSING_FILENAME,
        TRANSLATED_TRANSCRIPT_FILENAME,
        attach_ground_truth,
        phq8_ground_truth,
        queue_realtime_depression_detection,
    )
    from .session_artifacts import (
        artifact_read_path,
        artifact_record,
        artifact_relative_path,
        artifact_write_path,
        ensure_artifact_parent,
        iter_artifact_files,
    )
    from .realtime_analytics import build_realtime_analytics_snapshot
    from .prompts import (
        HINDSIGHT_RETAIN_CONTEXT,
        append_long_term_memory_instructions,
        append_tool_output_to_instructions,
        build_default_assistant_instructions,
        build_medical_qa_tool_spec,
        build_mood_assessment_context_prompt,
        build_mood_assessment_tool_spec,
        build_response_instructions,
        build_search_web_tool_spec,
        build_web_search_context_prompt,
        canonical_mood_aspect_key,
        mood_aspect_labels,
        normalize_mood_aspect_keys,
        normalize_mood_aspect_state,
    )
except ImportError:
    from storage import (
        get_realtime_session,
        get_realtime_session_run,
        get_depression_job,
        attach_latest_google_form_response_to_metadata,
        google_form_account_key,
        initialize_database,
        insert_google_form_response,
        list_depression_workers,
        list_google_form_response_summaries,
        depression_queue_counts,
        resolve_database_path,
        update_realtime_session_run_depression,
        update_realtime_session_run_memory,
        upsert_realtime_session_run,
    )
    from depression_detector import (
        PREPROCESSING_FILENAME,
        TRANSLATED_TRANSCRIPT_FILENAME,
        attach_ground_truth,
        phq8_ground_truth,
        queue_realtime_depression_detection,
    )
    from session_artifacts import (
        artifact_read_path,
        artifact_record,
        artifact_relative_path,
        artifact_write_path,
        ensure_artifact_parent,
        iter_artifact_files,
    )
    from realtime_analytics import build_realtime_analytics_snapshot
    from prompts import (
        HINDSIGHT_RETAIN_CONTEXT,
        append_long_term_memory_instructions,
        append_tool_output_to_instructions,
        build_default_assistant_instructions,
        build_medical_qa_tool_spec,
        build_mood_assessment_context_prompt,
        build_mood_assessment_tool_spec,
        build_response_instructions,
        build_search_web_tool_spec,
        build_web_search_context_prompt,
        canonical_mood_aspect_key,
        mood_aspect_labels,
        normalize_mood_aspect_keys,
        normalize_mood_aspect_state,
    )

try:
    from opencc import OpenCC
except Exception:
    OpenCC = None


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_dotenv_file(path: Path) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        name, value = line.split("=", 1)
        name = name.strip()
        if not name or name in os.environ:
            continue
        os.environ[name] = _strip_env_value(value)


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DOTENV_PATHS = [PROJECT_ROOT / ".env", APP_DIR / ".env"]

for dotenv_path in DOTENV_PATHS:
    load_dotenv_file(dotenv_path)

app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

UPLOAD_ROOT = (PROJECT_ROOT / "uploads" / "realtime").resolve()
FORM_UPLOAD_ROOT = UPLOAD_ROOT
DATABASE_PATH = resolve_database_path(PROJECT_ROOT)

WEB_SEARCH_MODEL = "gpt-5.4-mini"
MOOD_ASSESSMENT_MODEL = "gpt-5.4-mini"
REALTIME_MODEL = "gpt-realtime-2"
REALTIME_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
REALTIME_DEFAULT_VOICE = "coral"
openai_client = None
openai_client_error = None
traditional_converter = None
traditional_converter_checked = False
traditional_converter_config = ""
traditional_converter_error = ""


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


_JSON_DEFAULT_MISSING = object()


def read_json_file(path: Path, default=_JSON_DEFAULT_MISSING):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is _JSON_DEFAULT_MISSING else default


def write_json_file(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def openai_key_missing_message() -> str:
    dotenv_locations = ", ".join(str(path) for path in DOTENV_PATHS)
    return (
        "OpenAI API key not found. Set OPENAI_API_KEY in .env "
        f"({dotenv_locations}) or as an environment variable."
    )


def env_config_value(*names: str, default=None):
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return value
    return default


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



def post_openai_json(api_path: str, payload: dict, timeout: int = 30) -> dict:
    api_key = load_openai_api_key()
    if not api_key:
        raise RuntimeError(openai_key_missing_message())

    url = "https://api.openai.com/v1/" + api_path.lstrip("/")
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_obj = urlrequest.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(request_obj, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI request failed: {exc.code} {body[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc


def is_hindsight_enabled() -> bool:
    return env_flag("HINDSIGHT_ENABLED", default=True)


def get_hindsight_base_url() -> str:
    return str(env_config_value(
        "HINDSIGHT_BASE_URL",
        default="http://127.0.0.1:8888",
    )).strip().rstrip("/")


def get_hindsight_recall_max_tokens() -> int:
    return env_positive_int("HINDSIGHT_RECALL_MAX_TOKENS", 1200)


def get_hindsight_recall_limit() -> int:
    return env_positive_int("HINDSIGHT_RECALL_LIMIT", 5)


def get_hindsight_recall_budget() -> str:
    budget = str(env_config_value("HINDSIGHT_RECALL_BUDGET", default="low")).strip().lower()
    return budget if budget in {"low", "mid", "high"} else "low"


def get_hindsight_timeout_seconds() -> int:
    return env_positive_int("HINDSIGHT_TIMEOUT_SECONDS", 5)


def get_hindsight_recall_timeout_seconds() -> int:
    configured = env_config_value(
        "HINDSIGHT_RECALL_TIMEOUT_SECONDS",
        "HINDSIGHT_TIMEOUT_SECONDS",
        default="2",
    )
    try:
        return max(1, int(str(configured).strip()))
    except Exception:
        return 2


def post_hindsight_json(api_path: str, payload: dict, timeout: int = 5) -> dict:
    base_url = get_hindsight_base_url()
    if not base_url:
        raise RuntimeError("HINDSIGHT_BASE_URL is empty")

    headers = {"Content-Type": "application/json"}
    api_key = str(env_config_value("HINDSIGHT_API_KEY", default="") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request_obj = urlrequest.Request(
        base_url + "/" + api_path.lstrip("/"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlrequest.urlopen(request_obj, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Hindsight request failed: {exc.code} {body[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Hindsight request failed: {exc.reason}") from exc


def format_hindsight_recall_results(results, limit: int | None = None) -> str:
    if not isinstance(results, list):
        return ""
    max_results = limit if limit is not None else get_hindsight_recall_limit()
    try:
        max_results = max(1, int(max_results))
    except Exception:
        max_results = get_hindsight_recall_limit()
    lines = []
    for result in results:
        if not isinstance(result, dict):
            continue
        text = str(result.get("text") or "").strip()
        if not text:
            continue
        occurred_at = str(
            result.get("occurred_start")
            or result.get("timestamp")
            or ""
        ).strip()
        prefix = f"[{occurred_at}] " if occurred_at else ""
        lines.append(f"- {prefix}{text}")
        if len(lines) >= max_results:
            break
    return "\n".join(lines)


def recall_hindsight_memory(session_hash: str, query: str) -> dict:
    if not is_hindsight_enabled():
        return {"status": "disabled", "context": ""}
    session_hash = sanitize_session_hash(session_hash)
    query = str(query or "").strip()
    if not session_hash or not query:
        return {"status": "skipped", "context": ""}

    response = post_hindsight_json(
        f"/v1/default/banks/{quote(session_hash, safe='')}/memories/recall",
        {
            "query": query,
            "max_tokens": get_hindsight_recall_max_tokens(),
            "budget": get_hindsight_recall_budget(),
            "query_timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        },
        timeout=get_hindsight_recall_timeout_seconds(),
    )
    context = format_hindsight_recall_results(
        response.get("results"),
        limit=get_hindsight_recall_limit(),
    )
    return {
        "status": "recalled" if context else "empty",
        "context": context,
    }


def safe_recall_hindsight_memory(session_hash: str, query: str) -> dict:
    try:
        return recall_hindsight_memory(session_hash, query)
    except Exception as exc:
        return {
            "status": "unavailable",
            "context": "",
            "error": str(exc),
        }


def get_realtime_model() -> str:
    return str(env_config_value("OPENAI_REALTIME_MODEL", "REALTIME_MODEL", default=REALTIME_MODEL)).strip()


def get_web_search_model() -> str:
    return str(env_config_value("OPENAI_WEB_SEARCH_MODEL", "WEB_SEARCH_MODEL", default=WEB_SEARCH_MODEL)).strip()


def get_mood_assessment_model() -> str:
    return str(env_config_value(
        "OPENAI_MOOD_ASSESSMENT_MODEL",
        "MOOD_ASSESSMENT_MODEL",
        default=MOOD_ASSESSMENT_MODEL,
    )).strip()


def get_medical_qa_vector_store_id() -> str:
    return str(env_config_value(
        "OPENAI_MED_QA_VECTOR_STORE_ID",
        "MEDICAL_QA_VECTOR_STORE_ID",
        default="",
    ) or "").strip()


def get_traditional_converter():
    global traditional_converter, traditional_converter_checked
    global traditional_converter_config, traditional_converter_error

    if traditional_converter_checked:
        return traditional_converter
    traditional_converter_checked = True

    if OpenCC is None:
        traditional_converter_error = "opencc package is not installed"
        return None

    errors = []
    for config in ("s2twp", "s2twp.json", "s2tw", "s2tw.json", "s2t", "s2t.json"):
        try:
            traditional_converter = OpenCC(config)
            traditional_converter_config = config
            traditional_converter_error = ""
            return traditional_converter
        except Exception as exc:
            errors.append(f"{config}: {exc}")

    traditional_converter_error = "; ".join(errors) or "failed to initialize opencc"
    return None


def convert_to_taiwan_traditional(text: str) -> Tuple[str, bool, str, str]:
    raw_text = str(text or "")
    converter = get_traditional_converter()
    if converter is None:
        return raw_text, False, traditional_converter_config, traditional_converter_error
    try:
        return converter.convert(raw_text), True, traditional_converter_config, ""
    except Exception as exc:
        return raw_text, False, traditional_converter_config, str(exc)


def is_medical_qa_enabled() -> bool:
    return bool(get_medical_qa_vector_store_id())


def build_medical_qa_tool() -> dict:
    return build_medical_qa_tool_spec()


def build_mood_assessment_tool() -> dict:
    return build_mood_assessment_tool_spec()


def build_search_web_tool(medical_qa_enabled: bool = True) -> dict:
    return build_search_web_tool_spec(medical_qa_enabled)


def build_realtime_tools() -> list[dict]:
    medical_qa_enabled = is_medical_qa_enabled()
    tools = [build_mood_assessment_tool()]
    if medical_qa_enabled:
        tools.append(build_medical_qa_tool())
    tools.append(build_search_web_tool(medical_qa_enabled))
    return tools


def build_realtime_client_session_config() -> dict:
    return {
        "type": "realtime",
        "model": get_realtime_model(),
        "audio": {
            "input": {
                "noise_reduction": {"type": "far_field"},
                "transcription": {"model": REALTIME_TRANSCRIPTION_MODEL, "language": "zh"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.22,
                    "prefix_padding_ms": 400,
                    "silence_duration_ms": 1500,
                    "create_response": False,
                    "interrupt_response": True,
                },
            },
            "output": {"voice": REALTIME_DEFAULT_VOICE},
        },
        "instructions": build_default_assistant_instructions(),
        "output_modalities": ["audio"],
        "tools": build_realtime_tools(),
        "tool_choice": "auto",
    }


def extract_client_secret_value(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    direct = payload.get("value")
    if direct:
        return str(direct)
    client_secret = payload.get("client_secret")
    if isinstance(client_secret, dict):
        return str(client_secret.get("value") or "")
    if isinstance(client_secret, str):
        return client_secret
    return ""


def normalize_realtime_messages(raw_messages) -> List[Dict[str, str]]:
    normalized = []
    if not isinstance(raw_messages, list):
        return normalized
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = "assistant" if str(item.get("role") or "").strip().lower() == "assistant" else "user"
        text = str(item.get("text") or item.get("content") or "").strip()
        if text:
            normalized.append({"role": role, "text": text})
    return normalized


def build_web_search_prompt(query: str, recent_messages: List[Dict[str, str]]) -> str:
    return build_web_search_context_prompt(query, recent_messages)


def build_mood_assessment_prompt(
    query: str,
    recent_messages: List[Dict[str, str]],
    mood_aspect_state: dict | None = None,
) -> str:
    return build_mood_assessment_context_prompt(query, recent_messages, mood_aspect_state)


def parse_openai_output_text(payload: dict) -> str:
    direct = payload.get("output_text")
    if direct:
        return str(direct)
    text_parts = []
    for output_item in payload.get("output") or []:
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content") or []:
            if isinstance(content_item, dict) and content_item.get("text"):
                text_parts.append(str(content_item.get("text")))
    return "\n".join(text_parts).strip()


def parse_json_object_text(text: str) -> dict:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(raw[start:end + 1])
    return parsed if isinstance(parsed, dict) else {}


def bounded_string_list(value, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        text = str(item or "").strip()
        if text:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def normalize_mood_assessment_plan(
    query: str,
    plan: dict,
    previous_aspect_state: dict | None = None,
) -> dict:
    plan = plan if isinstance(plan, dict) else {}
    previous_state = normalize_mood_aspect_state(previous_aspect_state)
    previous_covered = previous_state["covered_aspects"]
    support_stage = str(plan.get("support_stage") or "comforting").strip()
    if support_stage not in {"exploration", "comforting", "action", "crisis_check"}:
        support_stage = "comforting"

    strategy = str(plan.get("strategy") or "validate").strip()
    if strategy not in {
        "ask_open_question",
        "reflect_feeling",
        "validate",
        "gentle_suggestion",
        "provide_information",
        "crisis_redirect",
    }:
        strategy = "validate"

    risk_level = str(plan.get("risk_level") or "none").strip()
    if risk_level not in {"none", "low", "moderate", "high", "imminent"}:
        risk_level = "none"

    response_guidance = str(plan.get("response_guidance") or "").strip()
    if not response_guidance:
        response_guidance = "先接住使用者感受，不診斷或評分；必要時問一個低負擔、貼近情境的追問。"

    suggested_follow_up = str(plan.get("suggested_follow_up") or "").strip()
    if not suggested_follow_up and support_stage in {"exploration", "comforting"}:
        suggested_follow_up = "你願意跟我說說，最近最困擾的是哪一部分嗎？"

    current_aspects = normalize_mood_aspect_keys(plan.get("current_aspects") or [])
    covered = []
    seen_aspects = set()
    for aspect_key in [*previous_covered, *current_aspects]:
        if aspect_key and aspect_key not in seen_aspects:
            covered.append(aspect_key)
            seen_aspects.add(aspect_key)
    remaining = normalize_mood_aspect_state({"covered_aspects": covered})["remaining_aspects"]
    next_focus_aspect = canonical_mood_aspect_key(plan.get("next_focus_aspect"))
    if next_focus_aspect not in remaining:
        next_focus_aspect = ""

    return {
        "status": str(plan.get("status") or "ok").strip() or "ok",
        "query": str(query or "").strip(),
        "support_stage": support_stage,
        "strategy": strategy,
        "risk_level": risk_level,
        "current_aspects": current_aspects,
        "current_aspect_labels": mood_aspect_labels(current_aspects),
        "next_focus_aspect": next_focus_aspect,
        "next_focus_aspect_label": mood_aspect_labels([next_focus_aspect])[0] if next_focus_aspect else "",
        "mood_aspect_state": {
            "covered_aspects": covered,
            "remaining_aspects": remaining,
        },
        "observed_signals": bounded_string_list(plan.get("observed_signals")),
        "user_need": str(plan.get("user_need") or "").strip(),
        "response_guidance": response_guidance,
        "suggested_follow_up": suggested_follow_up,
        "avoid": bounded_string_list(plan.get("avoid")),
    }


def assess_mood_support(
    query: str,
    recent_messages: List[Dict[str, str]],
    mood_aspect_state: dict | None = None,
) -> dict:
    normalized_aspect_state = normalize_mood_aspect_state(mood_aspect_state)
    prompt = build_mood_assessment_prompt(query, recent_messages, normalized_aspect_state)
    model = get_mood_assessment_model()
    print(
        "[MOOD-ASSESSMENT-PROMPT] "
        f"model={model}\n"
        f"{prompt}\n"
        "[/MOOD-ASSESSMENT-PROMPT]",
        flush=True,
    )
    payload = {
        "model": model,
        "max_output_tokens": 500,
        "input": prompt,
    }
    response = post_openai_json("/responses", payload, timeout=30)
    output_text = parse_openai_output_text(response)
    parse_error = ""
    try:
        parsed = parse_json_object_text(output_text)
    except Exception as exc:
        parsed = {}
        parse_error = str(exc)
    return {
        "prompt": prompt,
        "output_text": output_text,
        "parse_error": parse_error,
        "plan": normalize_mood_assessment_plan(query, parsed, normalized_aspect_state),
    }


def search_realtime_web_context(query: str, recent_messages: List[Dict[str, str]]) -> dict:
    prompt = build_web_search_prompt(query, recent_messages)
    payload = {
        "model": get_web_search_model(),
        "max_output_tokens": 350,
        "tool_choice": "required",
        "tools": [{"type": "web_search"}],
        "input": prompt,
    }
    response = post_openai_json("/responses", payload, timeout=35)
    return {"prompt": prompt, "output_text": parse_openai_output_text(response)}


def search_medical_qa(query: str) -> list[dict]:
    vector_store_id = get_medical_qa_vector_store_id()
    if not vector_store_id:
        return []
    payload = {"query": query.strip(), "max_num_results": 3}
    response = post_openai_json(f"/vector_stores/{quote(vector_store_id, safe='')}/search", payload, timeout=25)
    matches = []
    for item in response.get("data") or []:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or ""
        if not text and isinstance(item.get("content"), str):
            text = item.get("content")
        if not text and isinstance(item.get("content"), list):
            parts = []
            for content_item in item.get("content"):
                if isinstance(content_item, dict) and content_item.get("text"):
                    parts.append(str(content_item.get("text")))
            text = "\n".join(parts)
        text = str(text or "").strip()
        if text:
            matches.append({"rank": len(matches) + 1, "text": text})
        if len(matches) >= 3:
            break
    return matches


def load_openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip()


def initialize_openai_client() -> None:
    global openai_client, openai_client_error

    if openai_client is not None:
        return

    api_key = load_openai_api_key()
    if not api_key:
        openai_client_error = openai_key_missing_message()
        return

    try:
        openai_client = OpenAI(api_key=api_key)
        openai_client_error = None
    except Exception as exc:
        openai_client = None
        openai_client_error = str(exc)


def get_first_value_from_form(payload: dict, *field_names: str) -> str:
    """Return the first non-empty answer from Apps Script payload."""
    fields = payload.get("fields") or {}
    for field_name in field_names:
        value = fields.get(field_name)
        if isinstance(value, list):
            value = value[0] if value else ""
        value = str(value or "").strip()
        if value:
            return value
    return ""


def normalize_google_form_date(raw_value: str) -> str:
    """Convert Apps Script timestamp/date text to YYYY-MM-DD when possible."""
    raw_value = str(raw_value or "").strip()
    if not raw_value:
        return datetime.now().strftime("%Y-%m-%d")

    # Common values from Apps Script: 2026-05-02T07:30:00.000Z or 2026-05-02 15:30:00
    normalized = raw_value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).strftime("%Y-%m-%d")
    except Exception:
        pass

    # If it is already a date-like string, keep the first part.
    if len(raw_value) >= 10 and raw_value[4:5] in {"-", "/"}:
        return raw_value[:10].replace("/", "-")

    return datetime.now().strftime("%Y-%m-%d")


def generate_random_form_hash() -> str:
    """Generate a random 24-character id for a Google Form response folder."""
    return uuid4().hex[:24]


def create_form_identity(date_prefix: str) -> Tuple[str, str]:
    """Build the legacy YYYYMMDD_<hash> identifier without creating a folder."""
    form_hash = generate_random_form_hash()
    return f"{date_prefix}_{form_hash}", form_hash


def score_phq8_answer(answer: str):
    answer = str(answer or "").strip()
    score_map = {
        "完全沒有": 0,
        "好幾天": 1,
        "超過一半以上的天數": 2,
        "幾乎每天": 3,
    }
    return score_map.get(answer)


def compute_phq8_score(fields: dict) -> dict:
    item_scores = {}
    total_score = 0
    answered_count = 0

    for item_no in range(1, 9):
        matched_key = ""
        matched_answer = ""
        for key, value in fields.items():
            if str(key).strip().startswith(f"{item_no}."):
                matched_key = str(key)
                if isinstance(value, list):
                    matched_answer = str(value[0] if value else "").strip()
                else:
                    matched_answer = str(value or "").strip()
                break

        score = score_phq8_answer(matched_answer)
        item_scores[str(item_no)] = {
            "question": matched_key,
            "answer": matched_answer,
            "score": score,
        }
        if score is not None:
            total_score += score
            answered_count += 1

    return {
        "scale": "PHQ-8",
        "total_score": total_score,
        "answered_count": answered_count,
        "max_score": 24,
        "item_scores": item_scores,
    }


@app.get("/")
def realtime_index():
    return render_template("realtime.html")


@app.get("/realtime")
def realtime_page():
    return render_template("realtime.html")


@app.get("/realtime-analytics")
def realtime_analytics_index():
    return render_template("realtime_analytics.html")


@app.get("/api/realtime-analytics")
def realtime_analytics_api():
    try:
        initialize_database(DATABASE_PATH)
        return jsonify(build_realtime_analytics_snapshot(DATABASE_PATH))
    except Exception as e:
        print(f"[ERROR] /api/realtime-analytics failed: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/api/realtime-users")
def list_realtime_users():
    try:
        users = list_google_form_response_summaries(DATABASE_PATH)

        return jsonify({
            "status": "ok",
            "count": len(users),
            "users": users,
        })

    except Exception as e:
        print(f"[ERROR] /api/realtime-users failed: {e}", flush=True)
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "users": [],
        }), 500



@app.post("/api/realtime-client-secret")
def create_realtime_client_secret():
    try:
        session_config = build_realtime_client_session_config()
        registered_tools = [tool.get("name") or tool.get("type") for tool in session_config.get("tools", []) if isinstance(tool, dict)]
        secret_payload = post_openai_json("/realtime/client_secrets", {"session": session_config}, timeout=30)
        secret_value = extract_client_secret_value(secret_payload)
        if not secret_value:
            raise RuntimeError("OpenAI client secret response did not include a secret value")
        return jsonify({
            "status": "ok",
            "client_secret": secret_payload,
            "value": secret_value,
            "realtime_model": session_config.get("model"),
            "registered_tools": registered_tools,
            "medical_qa_enabled": is_medical_qa_enabled(),
            "web_search_enabled": True,
        })
    except Exception as e:
        print(f"[ERROR] /api/realtime-client-secret failed: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/realtime-response-instructions")
def realtime_response_instructions():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        kind = payload.get("kind") or "default"
        user_transcript = payload.get("user_transcript") or ""
        tool_output = payload.get("tool_output") or ""
        mood_aspect_state = normalize_mood_aspect_state(payload.get("mood_aspect_state") or {})
        account_id = sanitize_session_hash(payload.get("account_id") or payload.get("session_hash"))
        recall_memory = payload.get("recall_memory", True)
        if isinstance(recall_memory, str):
            recall_memory = recall_memory.strip().lower() not in {"0", "false", "no", "off"}

        memory_context = str(payload.get("memory_context") or "").strip()
        memory_status = str(payload.get("memory_status") or "").strip()
        memory_error = ""
        if recall_memory:
            memory_result = safe_recall_hindsight_memory(account_id, user_transcript)
            memory_context = memory_result.get("context") or ""
            memory_status = memory_result.get("status") or ""
            memory_error = memory_result.get("error") or ""
            if memory_error:
                print(
                    f"[HINDSIGHT] Recall unavailable | account={account_id} | error={memory_error}",
                    flush=True,
                )
        elif not memory_status:
            memory_status = "reused" if memory_context else "skipped"

        instructions = build_response_instructions(kind, user_transcript, mood_aspect_state)
        instructions = append_long_term_memory_instructions(instructions, memory_context)
        instructions = append_tool_output_to_instructions(instructions, tool_output)
        print(
            "[REALTIME-PROMPT] "
            f"account={account_id or '-'} | kind={kind} | "
            f"memory_status={memory_status or '-'}\n"
            f"{instructions}\n"
            "[/REALTIME-PROMPT]",
            flush=True,
        )
        return jsonify({
            "status": "ok",
            "kind": kind,
            "instructions": instructions,
            "memory_context": memory_context,
            "memory_status": memory_status,
            "memory_error": memory_error,
            "mood_aspect_state": mood_aspect_state,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/realtime-memory-recall")
def realtime_memory_recall():
    started_at = time.monotonic()
    try:
        payload = request.get_json(force=True, silent=True) or {}
        account_id = sanitize_session_hash(payload.get("account_id") or payload.get("session_hash"))
        user_transcript = str(payload.get("user_transcript") or "").strip()
        memory_result = safe_recall_hindsight_memory(account_id, user_transcript)
        duration_ms = round((time.monotonic() - started_at) * 1000)
        print(
            f"[HINDSIGHT] Recall completed | account={account_id} "
            f"| status={memory_result.get('status')} | duration_ms={duration_ms}",
            flush=True,
        )
        return jsonify({
            "status": "ok",
            "memory_context": memory_result.get("context") or "",
            "memory_status": memory_result.get("status") or "",
            "memory_error": memory_result.get("error") or "",
            "duration_ms": duration_ms,
        })
    except Exception as e:
        duration_ms = round((time.monotonic() - started_at) * 1000)
        print(
            f"[HINDSIGHT] Recall failed | duration_ms={duration_ms} | error={e}",
            flush=True,
        )
        return jsonify({
            "status": "ok",
            "memory_context": "",
            "memory_status": "error",
            "memory_error": str(e),
            "duration_ms": duration_ms,
        })


@app.post("/api/convert-traditional")
def convert_traditional():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        converted_text, converter_available, converter_config, converter_error = convert_to_taiwan_traditional(str(payload.get("text") or ""))
        return jsonify({
            "status": "ok",
            "text": converted_text,
            "converter_available": converter_available,
            "converter_config": converter_config,
            "converter_error": converter_error,
        })
    except Exception as e:
        print(f"[ERROR] /api/convert-traditional failed: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/api/realtime-medical-qa")
def realtime_medical_qa():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        query = str(payload.get("query") or "").strip()
        if not query:
            return jsonify({"status": "empty_query", "matches": []})
        if not get_medical_qa_vector_store_id():
            return jsonify({"status": "medical_qa_unavailable", "matches": []})
        matches = search_medical_qa(query)
        return jsonify({"status": "ok" if matches else "no_matches", "query": query, "matches": matches})
    except Exception as e:
        print(f"[ERROR] /api/realtime-medical-qa failed: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"status": "medical_qa_failed", "message": str(e), "matches": []}), 500


@app.post("/api/realtime-mood-assessment")
def realtime_mood_assessment():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        query = str(payload.get("query") or "").strip()
        recent_messages = normalize_realtime_messages(payload.get("recent_messages") or [])
        mood_aspect_state = normalize_mood_aspect_state(payload.get("mood_aspect_state") or {})
        if not query:
            return jsonify({
                "status": "empty_query",
                "query": "",
                "plan": {},
                "mood_aspect_state": mood_aspect_state,
            })

        result = assess_mood_support(query, recent_messages, mood_aspect_state)
        plan = result.get("plan") or {}
        status = "mood_assessment_parse_failed" if result.get("parse_error") else "ok"
        return jsonify({
            "status": status,
            "query": query,
            "model": get_mood_assessment_model(),
            "prompt": result.get("prompt", ""),
            "parse_error": result.get("parse_error", ""),
            "plan": plan,
            "mood_aspect_state": plan.get("mood_aspect_state", mood_aspect_state),
            "support_stage": plan.get("support_stage", ""),
            "strategy": plan.get("strategy", ""),
            "risk_level": plan.get("risk_level", ""),
            "current_aspects": plan.get("current_aspects", []),
            "current_aspect_labels": plan.get("current_aspect_labels", []),
            "next_focus_aspect": plan.get("next_focus_aspect", ""),
            "next_focus_aspect_label": plan.get("next_focus_aspect_label", ""),
            "observed_signals": plan.get("observed_signals", []),
            "user_need": plan.get("user_need", ""),
            "response_guidance": plan.get("response_guidance", ""),
            "suggested_follow_up": plan.get("suggested_follow_up", ""),
            "avoid": plan.get("avoid", []),
        })
    except Exception as e:
        print(f"[ERROR] /api/realtime-mood-assessment failed: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"status": "mood_assessment_failed", "message": str(e), "plan": {}}), 500


@app.post("/api/realtime-web-search")
def realtime_web_search():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        query = str(payload.get("query") or "").strip()
        recent_messages = normalize_realtime_messages(payload.get("recent_messages") or [])
        if not query:
            return jsonify({"status": "empty_query", "web_context": ""})
        search_result = search_realtime_web_context(query, recent_messages)
        web_context = search_result.get("output_text", "")
        return jsonify({
            "status": "ok" if web_context else "no_context",
            "query": query,
            "prompt": search_result.get("prompt", ""),
            "web_context": web_context,
        })
    except Exception as e:
        print(f"[ERROR] /api/realtime-web-search failed: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"status": "search_failed", "message": str(e), "web_context": ""}), 500


@app.post("/google_form")
def receive_google_form():
    try:
        payload = request.get_json(force=True, silent=True) or {}


        fields = payload.get("fields") or {}
        if not isinstance(fields, dict):
            fields = {}

        name = get_first_value_from_form(payload, "姓名", "name")
        age = get_first_value_from_form(payload, "年齡", "age")
        respondent_email = str(payload.get("respondent_email") or "").strip().lower()
        if not respondent_email:
            return jsonify({
                "status": "error",
                "message": "respondent_email is required; enable email collection on the Google Form.",
            }), 400
        submitted_at = str(payload.get("submitted_at") or datetime.now().isoformat(timespec="seconds"))
        form_date = get_first_value_from_form(payload, "日期", "date") or submitted_at
        date_text = normalize_google_form_date(form_date)
        date_prefix = date_text.replace("-", "")

        form_dir_name, form_hash = create_form_identity(date_prefix)
        account_id = google_form_account_key(respondent_email, form_hash)

        phq8_result = compute_phq8_score(fields)

        record = {
            "status": "ok",
            "source": "google_form",
            "form_title": payload.get("form_title") or "PHQ-8情緒量表",
            "form_id": payload.get("form_id") or "",
            "response_id": payload.get("response_id") or "",
            "respondent_email": respondent_email,
            "account_id": account_id,
            "account_hash": account_id,
            "submitted_at": submitted_at,
            "received_at": datetime.now().isoformat(timespec="seconds"),
            "name": name,
            "age": age,
            "date_prefix": date_prefix,
            "form_hash": form_hash,
            "form_dir_name": form_dir_name,
            "fields": fields,
            "phq8": phq8_result,
            "remote_addr": request.remote_addr,
            "user_agent": request.headers.get("User-Agent"),
        }

        insert_google_form_response(DATABASE_PATH, record)

        print(
            f"[GOOGLE FORM] Done | account={account_id} | dir={form_dir_name} | name={name or 'unknown'} | score={phq8_result.get('total_score')}",
            flush=True,
        )

        return jsonify({
            "status": "ok",
            "message": "google form response saved",
            "account_id": account_id,
            "account_hash": account_id,
            "session_hash": account_id,
            "form_hash": form_hash,
            "form_dir_name": form_dir_name,
            "saved_file": "",
            "saved_to": "sqlite",
            "phq8": phq8_result,
        })

    except Exception as e:
        print(f"[ERROR] /api/google-form failed: {e}", flush=True)
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


def sanitize_session_hash(value: str) -> str:
    return "".join(ch for ch in str(value or "").strip() if ch.isalnum() or ch in {"_", "-"})


def parse_session_datetime(value) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value) / 1000).astimezone()
        except Exception:
            return None

    raw_value = str(value).strip()
    if not raw_value:
        return None
    normalized_value = raw_value.replace("Z", "+00:00")
    fractional_match = re.fullmatch(
        r"(.*[T ]\d{2}:\d{2}:\d{2}\.)(\d+)([+-]\d{2}:\d{2})?",
        normalized_value,
    )
    if fractional_match:
        fractional_seconds = (fractional_match.group(2) + "000000")[:6]
        normalized_value = (
            fractional_match.group(1)
            + fractional_seconds
            + (fractional_match.group(3) or "")
        )
    try:
        parsed = datetime.fromisoformat(normalized_value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.astimezone()
    return parsed.astimezone()


def get_session_started_at(metadata: dict, fallback: datetime) -> datetime:
    if isinstance(metadata, dict):
        started_at = parse_session_datetime(metadata.get("started_at_iso"))
        if started_at is None:
            started_at = parse_session_datetime(metadata.get("started_at_ms"))
        if started_at is not None:
            return started_at
    return fallback.astimezone() if fallback.tzinfo else fallback.astimezone()


def format_transcript_timestamp(timestamp_ms) -> str:
    try:
        return datetime.fromtimestamp(float(timestamp_ms) / 1000).astimezone().isoformat(
            timespec="milliseconds"
        )
    except Exception:
        return ""


def build_hindsight_transcript(transcript, transcript_text: str = "") -> str:
    lines = []
    events = transcript.get("events") if isinstance(transcript, dict) else None
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            speaker = str(event.get("speaker") or "").strip().lower()
            if speaker not in {"user", "assistant"}:
                continue
            text = str(event.get("text") or "").strip()
            if not text:
                continue
            timestamp = format_transcript_timestamp(event.get("timestamp_ms"))
            label = "USER" if speaker == "user" else "ASSISTANT"
            timestamp_prefix = f"[{timestamp}] " if timestamp else ""
            lines.append(f"{timestamp_prefix}{label}: {text}")
    if lines:
        return "\n".join(lines)

    plain_text = ""
    if isinstance(transcript, dict):
        plain_text = str(transcript.get("plain_text") or "").strip()
    return plain_text or str(transcript_text or "").strip()


def retain_realtime_session_memory(
    session_hash: str,
    run_id: str,
    started_at: str,
    transcript,
    transcript_text: str = "",
) -> dict:
    if not is_hindsight_enabled():
        return {"status": "disabled"}

    session_hash = sanitize_session_hash(session_hash)
    content = build_hindsight_transcript(transcript, transcript_text)
    if not session_hash or not run_id or not content:
        return {"status": "skipped"}

    response = post_hindsight_json(
        f"/v1/default/banks/{quote(session_hash, safe='')}/memories",
        {
            "items": [
                {
                    "content": content,
                    "context": HINDSIGHT_RETAIN_CONTEXT,
                    "timestamp": started_at,
                    "document_id": f"realtime-session:{run_id}",
                    "metadata": {
                        "source": "web_realtime",
                        "account_id": session_hash,
                        "session_hash": session_hash,
                        "run_id": run_id,
                    },
                }
            ],
            "async": True,
        },
        timeout=get_hindsight_timeout_seconds(),
    )
    return {
        "status": "queued",
        "operation_id": str(response.get("operation_id") or "").strip() or None,
        "queued_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
    }


def get_session_identity(metadata: dict) -> str:
    if not isinstance(metadata, dict):
        return ""
    started_at_iso = str(metadata.get("started_at_iso") or "").strip()
    if started_at_iso:
        return f"iso:{started_at_iso}"
    started_at_ms = str(metadata.get("started_at_ms") or "").strip()
    return f"ms:{started_at_ms}" if started_at_ms else ""


def format_session_run_id(started_at: datetime) -> str:
    local_started_at = started_at.astimezone() if started_at.tzinfo else started_at.astimezone()
    return local_started_at.strftime("%Y%m%d_%H%M%S_") + f"{local_started_at.microsecond // 1000:03d}"


def load_session_metadata(session_dir: Path) -> dict:
    metadata = read_json_file(artifact_read_path(session_dir, "metadata.json"), default={})
    return metadata if isinstance(metadata, dict) else {}


def resolve_session_run_directory(
    upload_root: Path,
    session_hash: str,
    metadata: dict,
    fallback_started_at: datetime,
) -> Tuple[Path, str]:
    hash_dir = ensure_dir(upload_root / session_hash)
    base_run_id = format_session_run_id(get_session_started_at(metadata, fallback_started_at))
    incoming_identity = get_session_identity(metadata)
    candidate = hash_dir / base_run_id
    if not candidate.exists():
        return candidate, base_run_id

    if incoming_identity and get_session_identity(load_session_metadata(candidate)) == incoming_identity:
        return candidate, base_run_id

    suffix = 1
    while True:
        run_id = f"{base_run_id}_{suffix:02d}"
        candidate = hash_dir / run_id
        if not candidate.exists():
            return candidate, run_id
        if incoming_identity and get_session_identity(load_session_metadata(candidate)) == incoming_identity:
            return candidate, run_id
        suffix += 1


def field_name_for_saved_file(filename: str) -> str:
    return {
        "archive_manifest.json": "archive_manifest_file",
        "metadata.json": "metadata_file",
        "transcript.json": "transcript_file",
        TRANSLATED_TRANSCRIPT_FILENAME: "depression_transcript_file",
        PREPROCESSING_FILENAME: "depression_preprocessing_file",
        "user_audio.wav": "user_audio_file",
        "assistant_audio.wav": "assistant_audio_file",
        "video_frames.zip": "video_frames_file",
        "depression_result.json": "depression_result_file",
        "depression_error.json": "depression_error_file",
        "depression_aspect_retrieval.jsonl": "depression_aspect_retrieval_file",
        "user_speech_intervals.csv": "participant_transcript_file",
        "participant_utterances.jsonl": "participant_utterances_file",
    }.get(filename, "saved_file")


def build_saved_file_records(session_dir: Path) -> Tuple[list[str], list[str], list[dict]]:
    files = list(iter_artifact_files(session_dir))
    saved_files = [str(path) for path in files]
    saved_file_names = [path.relative_to(session_dir).as_posix() for path in files]
    saved_file_paths = [
        artifact_record(session_dir, path, field_name_for_saved_file(path.name))
        for path in files
    ]
    return saved_files, saved_file_names, saved_file_paths


def build_archive_manifest(
    session_hash: str,
    run_id: str,
    session_dir: Path,
    metadata: dict,
    transcript: dict | None,
) -> dict:
    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    artifacts = []
    for path in iter_artifact_files(session_dir):
        relative_path = path.relative_to(session_dir).as_posix()
        if path.name == "archive_manifest.json":
            continue
        artifacts.append(
            {
                "field_name": field_name_for_saved_file(path.name),
                "filename": path.name,
                "relative_path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    ground_truth = phq8_ground_truth(metadata)
    result_path = artifact_read_path(session_dir, "depression_result.json")
    error_path = artifact_read_path(session_dir, "depression_error.json")
    user_audio_path = artifact_read_path(session_dir, "user_audio.wav")
    video_frames_path = artifact_read_path(session_dir, "video_frames.zip")
    return {
        "schema_version": "companion-thesis-realtime-v1",
        "account_id": session_hash,
        "session_hash": session_hash,
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "prediction_status": (
            "ok"
            if result_path.is_file()
            else "error"
            if error_path.is_file()
            else "pending"
        ),
        "thesis_contract": {
            "text": {
                "source": "transcript.events",
                "participant_role": "user",
                "event_count": len(transcript.get("events") or [])
                if isinstance(transcript, dict)
                else 0,
                "participant_utterances_file": artifact_relative_path("participant_utterances.jsonl").as_posix(),
                "participant_interval_file": artifact_relative_path("user_speech_intervals.csv").as_posix(),
            },
            "audio": {
                "filename": "user_audio.wav",
                "relative_path": artifact_relative_path("user_audio.wav").as_posix(),
                "present": user_audio_path.is_file(),
                "format": "wav_pcm_s16le_mono",
                "sample_rate": metadata.get("user_audio_sample_rate"),
                "participant_only_at_feature_extraction": True,
                "missing_policy": "zero_feature_vector",
            },
            "video": {
                "filename": "video_frames.zip",
                "relative_path": artifact_relative_path("video_frames.zip").as_posix(),
                "present": video_frames_path.is_file(),
                "format": "zip_of_timestamp_ordered_jpeg_frames",
                "sampling_interval_ms": metadata.get(
                    "video_frame_sampling_interval_ms"
                ),
                "frame_count": metadata.get("video_frame_count"),
                "missing_policy": "zero_feature_vector",
            },
            "ground_truth": ground_truth,
        },
        "artifacts": artifacts,
    }


def index_realtime_session_run(
    database_path: Path,
    upload_root: Path,
    session_hash: str,
    run_id: str,
    session_dir: Path,
    legacy_record: Optional[dict] = None,
) -> None:
    existing_record = get_realtime_session_run(database_path, session_hash, run_id)
    metadata = load_session_metadata(session_dir)
    metadata, metadata_changed = attach_latest_google_form_response_to_metadata(
        database_path,
        session_hash=session_hash,
        metadata=metadata,
    )
    if metadata_changed:
        write_json_file(
            ensure_artifact_parent(artifact_write_path(session_dir, "metadata.json")),
            metadata,
        )
    transcript = read_json_file(artifact_read_path(session_dir, "transcript.json"), default=None)
    if not isinstance(transcript, dict):
        transcript = None
    transcript_text = (
        str(transcript.get("plain_text") or "").strip()
        if isinstance(transcript, dict)
        else None
    )
    depression_result_path = artifact_read_path(session_dir, "depression_result.json")
    depression_result = (
        read_json_file(depression_result_path, default=None)
        if depression_result_path.is_file()
        else None
    )
    valid_depression_result = (
        isinstance(depression_result, dict)
        and depression_result.get("status") == "ok"
        and isinstance(depression_result.get("aspects"), list)
    )
    if valid_depression_result:
        depression_result = attach_ground_truth(depression_result, metadata)
        write_json_file(depression_result_path, depression_result)
    else:
        depression_result = None
    depression_error_path = artifact_read_path(session_dir, "depression_error.json")
    depression_error = (
        read_json_file(depression_error_path, default=None)
        if depression_error_path.is_file()
        else None
    )
    if not isinstance(depression_error, dict):
        depression_error = None
    fallback_started_at = datetime.fromtimestamp(session_dir.stat().st_mtime).astimezone()
    started_at = str(metadata.get("started_at_iso") or "").strip()
    if not started_at:
        started_at = (
            str((legacy_record or {}).get("created_at") or "").strip()
            or get_session_started_at(metadata, fallback_started_at).isoformat(timespec="milliseconds")
        )
    ended_at = str(metadata.get("ended_at_iso") or "").strip() or None
    uploaded_at = (
        str((legacy_record or {}).get("updated_at") or "").strip()
        or str((existing_record or {}).get("uploaded_at") or "").strip()
        or datetime.fromtimestamp(session_dir.stat().st_mtime).astimezone().isoformat(timespec="milliseconds")
    )
    archive_manifest_path = ensure_artifact_parent(
        artifact_write_path(session_dir, "archive_manifest.json")
    )
    archive_manifest = build_archive_manifest(
        session_hash,
        run_id,
        session_dir,
        metadata,
        transcript,
    )
    write_json_file(archive_manifest_path, archive_manifest)
    _, saved_file_names, saved_file_paths = build_saved_file_records(session_dir)
    session_dir_name = session_dir.relative_to(upload_root).as_posix()
    ground_truth = phq8_ground_truth(metadata)

    upsert_realtime_session_run(
        database_path,
        {
            "session_hash": session_hash,
            "run_id": run_id,
            "session_dir_name": session_dir_name,
            "started_at": started_at,
            "ended_at": ended_at,
            "uploaded_at": uploaded_at,
            "remote_addr": (
                (legacy_record or {}).get("remote_addr")
                or (existing_record or {}).get("remote_addr")
            ),
            "user_agent": (
                (legacy_record or {}).get("user_agent")
                or (existing_record or {}).get("user_agent")
            ),
            "metadata": metadata or None,
            "transcript": transcript,
            "transcript_text": transcript_text,
            "saved_file_names": saved_file_names,
            "saved_file_paths": saved_file_paths,
            "archive_manifest": archive_manifest,
            "ground_truth_total_score": ground_truth.get("total_score"),
            "ground_truth_binary": ground_truth.get("binary_depression"),
        },
    )
    if depression_result is not None:
        update_realtime_session_run_depression(
            database_path,
            session_hash,
            run_id,
            {
                "status": depression_result.get("status") or "ok",
                "total_score": depression_result.get("total_score"),
                "binary": depression_result.get("binary_depression"),
                "result": depression_result,
                "completed_at": depression_result.get("completed_at"),
            },
        )
    elif depression_error is not None:
        update_realtime_session_run_depression(
            database_path,
            session_hash,
            run_id,
            {
                "status": depression_error.get("status") or "error",
                "result": depression_error,
                "error": depression_error.get("error"),
                "completed_at": depression_error.get("completed_at"),
            },
        )


def migrate_legacy_realtime_sessions(database_path: Path, upload_root: Path) -> None:
    if not upload_root.is_dir():
        return

    legacy_pattern = re.compile(r"^\d{8}_(.+)$")
    for legacy_dir in sorted(path for path in upload_root.iterdir() if path.is_dir()):
        match = legacy_pattern.fullmatch(legacy_dir.name)
        if not match:
            continue
        metadata = load_session_metadata(legacy_dir)
        session_hash = sanitize_session_hash(metadata.get("session_hash") or match.group(1))
        if not session_hash:
            continue

        legacy_record = get_realtime_session(database_path, session_hash)
        if legacy_record and legacy_record.get("session_dir_name") != legacy_dir.name:
            legacy_record = None
        fallback_started_at = datetime.fromtimestamp(legacy_dir.stat().st_mtime).astimezone()
        target_dir, run_id = resolve_session_run_directory(
            upload_root,
            session_hash,
            metadata,
            fallback_started_at,
        )
        ensure_dir(target_dir.parent)
        if target_dir.exists():
            _, run_id = resolve_session_run_directory(
                upload_root,
                session_hash,
                {},
                fallback_started_at,
            )
            target_dir = upload_root / session_hash / run_id
        shutil.move(str(legacy_dir), str(target_dir))
        index_realtime_session_run(
            database_path,
            upload_root,
            session_hash,
            run_id,
            target_dir,
            legacy_record=legacy_record,
        )
        print(
            f"[MIGRATION] Realtime session moved | from={legacy_dir.name} "
            f"| to={session_hash}/{run_id}",
            flush=True,
        )

    for hash_dir in sorted(path for path in upload_root.iterdir() if path.is_dir()):
        if legacy_pattern.fullmatch(hash_dir.name):
            continue
        session_hash = sanitize_session_hash(hash_dir.name)
        if not session_hash:
            continue
        run_dirs = sorted(path for path in hash_dir.iterdir() if path.is_dir())
        legacy_record = None
        if len(run_dirs) == 1:
            candidate_legacy_record = get_realtime_session(database_path, session_hash)
            legacy_dir_name = str((candidate_legacy_record or {}).get("session_dir_name") or "")
            if re.fullmatch(rf"\d{{8}}_{re.escape(session_hash)}", legacy_dir_name):
                legacy_record = candidate_legacy_record
        for run_dir in run_dirs:
            index_realtime_session_run(
                database_path,
                upload_root,
                session_hash,
                run_dir.name,
                run_dir,
                legacy_record=legacy_record,
            )


@app.post("/api/realtime-session")
def upload_realtime_session():
    try:
        account_id = (
            sanitize_session_hash(request.form.get("account_id") or request.form.get("session_hash"))
            or uuid4().hex[:24]
        )
        upload_received_datetime = datetime.now().astimezone()
        upload_received_at = upload_received_datetime.isoformat(timespec="milliseconds")

        metadata = {}
        transcript = None
        transcript_text = None
        cached_uploads = {}
        metadata_file = request.files.get("metadata_file")
        if metadata_file and metadata_file.filename:
            metadata_bytes = metadata_file.read()
            cached_uploads["metadata_file"] = metadata_bytes
            try:
                metadata = json.loads(metadata_bytes.decode("utf-8-sig", errors="replace"))
            except Exception:
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
        metadata, metadata_changed = attach_latest_google_form_response_to_metadata(
            DATABASE_PATH,
            session_hash=account_id,
            metadata=metadata,
        )
        if metadata_changed:
            cached_uploads["metadata_file"] = json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")

        session_dir, run_id = resolve_session_run_directory(
            UPLOAD_ROOT,
            account_id,
            metadata,
            upload_received_datetime,
        )
        session_dir = ensure_dir(session_dir)
        session_dir_name = session_dir.relative_to(UPLOAD_ROOT).as_posix()
        saved_files = []
        saved_file_names = []
        saved_file_paths = []
        canonical_upload_names = {
            "metadata_file": "metadata.json",
            "transcript_file": "transcript.json",
        }
        for field_name, filename in canonical_upload_names.items():
            file = request.files.get(field_name)
            if file and file.filename:
                target = ensure_artifact_parent(artifact_write_path(session_dir, filename))
                raw_bytes = cached_uploads.get(field_name)
                if raw_bytes is None:
                    raw_bytes = file.read()
                target.write_bytes(raw_bytes)
                raw_text = raw_bytes.decode("utf-8-sig", errors="replace")
                if field_name == "transcript_file":
                    try:
                        transcript = json.loads(raw_text)
                    except Exception:
                        transcript = None
                    if not isinstance(transcript, dict):
                        transcript = None
                    if isinstance(transcript, dict):
                        transcript_text = str(transcript.get("plain_text") or "").strip()
                saved_files.append(str(target))
                saved_file_names.append(target.relative_to(session_dir).as_posix())
                saved_file_paths.append(artifact_record(session_dir, target, field_name))

        canonical_binary_names = {
            "user_audio_file": "user_audio.wav",
            "assistant_audio_file": "assistant_audio.wav",
            "video_frames_file": "video_frames.zip",
        }
        for field_name, filename in canonical_binary_names.items():
            file = request.files.get(field_name)
            if file and file.filename:
                target = ensure_artifact_parent(artifact_write_path(session_dir, filename))
                file.save(target)
                saved_files.append(str(target))
                saved_file_names.append(target.relative_to(session_dir).as_posix())
                saved_file_paths.append(artifact_record(session_dir, target, field_name))

        archive_manifest = build_archive_manifest(
            account_id,
            run_id,
            session_dir,
            metadata,
            transcript,
        )
        archive_manifest_path = ensure_artifact_parent(
            artifact_write_path(session_dir, "archive_manifest.json")
        )
        write_json_file(archive_manifest_path, archive_manifest)
        saved_files.append(str(archive_manifest_path))
        saved_file_names.append(archive_manifest_path.relative_to(session_dir).as_posix())
        saved_file_paths.append(
            artifact_record(session_dir, archive_manifest_path, "archive_manifest_file")
        )

        started_at = (
            str(metadata.get("started_at_iso") or "").strip()
            or get_session_started_at(metadata, upload_received_datetime).isoformat(timespec="milliseconds")
        )
        ground_truth = archive_manifest["thesis_contract"]["ground_truth"]
        upsert_realtime_session_run(
            DATABASE_PATH,
            {
                "session_hash": account_id,
                "account_id": account_id,
                "run_id": run_id,
                "session_dir_name": session_dir_name,
                "started_at": started_at,
                "ended_at": str(metadata.get("ended_at_iso") or "").strip() or None,
                "uploaded_at": upload_received_at,
                "remote_addr": request.remote_addr,
                "user_agent": request.headers.get("User-Agent"),
                "metadata": metadata or None,
                "transcript": transcript,
                "transcript_text": transcript_text,
                "saved_file_names": saved_file_names,
                "saved_file_paths": saved_file_paths,
                "archive_manifest": archive_manifest,
                "ground_truth_total_score": ground_truth.get("total_score"),
                "ground_truth_binary": ground_truth.get("binary_depression"),
            },
        )

        try:
            memory_update = retain_realtime_session_memory(
                account_id,
                run_id,
                started_at,
                transcript,
                transcript_text,
            )
        except Exception as exc:
            memory_update = {
                "status": "unavailable",
                "message": str(exc),
            }
            print(
                f"[HINDSIGHT] Retain failed | account={account_id} "
                f"| run={run_id} | error={exc}",
                flush=True,
            )
        try:
            update_realtime_session_run_memory(
                DATABASE_PATH,
                account_id,
                run_id,
                memory_update,
            )
        except Exception as exc:
            print(
                f"[HINDSIGHT] Failed to save Retain status | account={account_id} "
                f"| run={run_id} | error={exc}",
                flush=True,
            )

        try:
            depression_update = queue_realtime_depression_detection(
                DATABASE_PATH,
                account_id,
                run_id,
                session_dir,
            )
        except Exception as exc:
            depression_update = {
                "status": "error",
                "message": str(exc),
            }
            try:
                update_realtime_session_run_depression(
                    DATABASE_PATH,
                    account_id,
                    run_id,
                    depression_update,
                )
            except Exception as update_exc:
                print(
                    f"[DEPRESSION] Failed to save queue error | account={account_id} "
                    f"| run={run_id} | error={update_exc}",
                    flush=True,
                )
            print(
                f"[DEPRESSION] Queue failed | account={account_id} "
                f"| run={run_id} | error={exc}",
                flush=True,
            )

        print(
            f"[UPLOAD] Done   | account={account_id} | run={run_id} "
            f"| files_saved={len(saved_files)} "
            f"| memory={memory_update.get('status')} "
            f"| depression={depression_update.get('status')}",
            flush=True,
        )

        return jsonify(
            {
                "status": "ok",
                "message": "upload saved",
                "account_id": account_id,
                "session_hash": account_id,
                "session_id": account_id,
                "run_id": run_id,
                "session_dir_name": session_dir_name,
                "saved_count": len(saved_files),
                "saved_files": saved_files,
                "saved_file_paths": saved_file_paths,
                "saved_to": "filesystem+sqlite",
                "memory_update": memory_update,
                "depression_detection": depression_update,
            }
        )

    except Exception as e:
        print(f"[ERROR] /api/realtime-session failed: {e}", flush=True)
        traceback.print_exc()

        return jsonify(
            {
                "status": "error",
                "message": str(e),
            }
        ), 500


@app.get("/api/realtime-session/<session_hash>/<run_id>/depression")
def realtime_session_depression(session_hash: str, run_id: str):
    try:
        account_id = sanitize_session_hash(session_hash)
        run_id = str(run_id or "").strip()
        if not account_id or not run_id:
            return jsonify({"status": "error", "message": "missing account_id or run_id"}), 400

        record = get_realtime_session_run(DATABASE_PATH, account_id, run_id)
        if not record:
            return jsonify({"status": "error", "message": "session run not found"}), 404
        job = get_depression_job(DATABASE_PATH, account_id, run_id)

        return jsonify(
            {
                "status": "ok",
                "account_id": record.get("account_id") or account_id,
                "session_hash": account_id,
                "run_id": run_id,
                "depression_status": record.get("depression_status") or "",
                "depression_total_score": record.get("depression_total_score"),
                "depression_binary": record.get("depression_binary"),
                "depression_error": record.get("depression_error"),
                "depression_completed_at": record.get("depression_completed_at"),
                "depression_result": record.get("depression_result"),
                "ground_truth_total_score": record.get(
                    "ground_truth_total_score"
                ),
                "ground_truth_binary": record.get("ground_truth_binary"),
                "archive_manifest": record.get("archive_manifest"),
                "depression_job": (
                    {
                        "id": job.get("id"),
                        "status": job.get("status"),
                        "attempts": job.get("attempts"),
                        "max_attempts": job.get("max_attempts"),
                        "queued_at": job.get("queued_at"),
                        "claimed_at": job.get("claimed_at"),
                        "heartbeat_at": job.get("heartbeat_at"),
                        "worker_id": job.get("worker_id"),
                        "last_error": job.get("last_error"),
                    }
                    if job
                    else None
                ),
            }
        )
    except Exception as e:
        print(f"[ERROR] /api/realtime-session/<session>/<run>/depression failed: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/health")
def healthcheck():
    return jsonify({
        "status": "ok",
        "openai_client_ready": openai_client is not None,
        "openai_client_error": openai_client_error,
        "realtime_model": get_realtime_model(),
        "mood_assessment_model": get_mood_assessment_model(),
        "web_search_model": get_web_search_model(),
        "medical_qa_enabled": is_medical_qa_enabled(),
        "hindsight_enabled": is_hindsight_enabled(),
        "hindsight_base_url": get_hindsight_base_url(),
        "database_path": str(DATABASE_PATH),
        "depression_queue": depression_queue_counts(DATABASE_PATH),
        "depression_workers": list_depression_workers(DATABASE_PATH),
        "traditional_converter_available": get_traditional_converter() is not None,
        "traditional_converter_error": traditional_converter_error,
    })


if __name__ == "__main__":
    initialize_database(DATABASE_PATH)
    migrate_legacy_realtime_sessions(DATABASE_PATH, UPLOAD_ROOT)
    initialize_openai_client()
    flask_debug = env_flag("FLASK_DEBUG", default=False)
    app.run(
        host="0.0.0.0",
        port=9050,
        debug=flask_debug,
        use_reloader=flask_debug
        and env_flag("FLASK_USE_RELOADER", default=True),
    )
else:
    initialize_database(DATABASE_PATH)
    migrate_legacy_realtime_sessions(DATABASE_PATH, UPLOAD_ROOT)
    initialize_openai_client()
