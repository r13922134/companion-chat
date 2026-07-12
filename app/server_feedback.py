from pathlib import Path
from uuid import uuid4
from datetime import datetime
import io
import json
import os
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
        count_feedback_records,
        ensure_realtime_session,
        google_form_account_key,
        initialize_database,
        insert_google_form_response,
        list_google_form_response_summaries,
        resolve_database_path,
        upsert_feedback_record,
        upsert_feedback_records,
        upsert_realtime_session,
    )
    from .prompts import (
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
        count_feedback_records,
        ensure_realtime_session,
        google_form_account_key,
        initialize_database,
        insert_google_form_response,
        list_google_form_response_summaries,
        resolve_database_path,
        upsert_feedback_record,
        upsert_feedback_records,
        upsert_realtime_session,
    )
    from prompts import (
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


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

UPLOAD_ROOT = (PROJECT_ROOT / "uploads" / "feedback").resolve()
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


def attach_previous_user_utterances(feedback_records: list[dict], transcript_payload: dict) -> None:
    events = transcript_payload.get("events") if isinstance(transcript_payload, dict) else []
    if not isinstance(events, list):
        return

    latest_user_utterance = ""
    questions_by_item_id = {}
    questions_by_response_text = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        speaker = str(event.get("speaker") or event.get("role") or "").strip().lower()
        text = str(event.get("text") or event.get("transcript") or "").strip()
        if speaker == "user":
            if text:
                latest_user_utterance = text
            continue
        if speaker != "assistant" or not latest_user_utterance:
            continue

        item_id = str(event.get("item_id") or "").strip()
        if item_id:
            questions_by_item_id[item_id] = latest_user_utterance
        if text:
            questions_by_response_text[text] = latest_user_utterance

    for record in feedback_records:
        if not isinstance(record, dict):
            continue
        existing_question = str(record.get("user_utterance") or "").strip()
        if existing_question:
            record["user_utterance"] = existing_question
            continue

        item_id = str(record.get("item_id") or "").strip()
        response_text = str(record.get("response_text") or "").strip()
        question = questions_by_item_id.get(item_id) if item_id else ""
        if not question and response_text:
            question = questions_by_response_text.get(response_text, "")
        if question:
            record["user_utterance"] = question


def build_feedback_qa_document(qa_id: str, user_utterance: str, feedback_text: str) -> str:
    return (
        f"### QA {qa_id.strip()}\n"
        f"問題：{user_utterance.strip()}\n"
        f"答覆：{feedback_text.strip()}\n"
    )


def upload_feedback_qa_pair(record: dict) -> dict:
    feedback_text = str(record.get("feedback_text") or "").strip()
    if not feedback_text:
        return {"status": "skipped", "reason": "empty_feedback"}

    user_utterance = str(record.get("user_utterance") or "").strip()
    if not user_utterance:
        return {"status": "skipped", "reason": "missing_user_utterance"}

    vector_store_id = get_medical_qa_vector_store_id()
    if not vector_store_id:
        return {"status": "skipped", "reason": "vector_store_not_configured"}

    initialize_openai_client()
    if openai_client is None:
        raise RuntimeError(openai_client_error or openai_key_missing_message())

    session_hash = str(record.get("session_hash") or "session").strip()
    item_id = str(record.get("item_id") or uuid4().hex[:12]).strip()
    qa_id = f"FEEDBACK_{session_hash}_{item_id}"
    filename = secure_filename(f"feedback_qa_{session_hash}_{item_id}.txt")
    if not filename:
        filename = f"feedback_qa_{uuid4().hex}.txt"

    content = build_feedback_qa_document(qa_id, user_utterance, feedback_text)
    file_buffer = io.BytesIO(content.encode("utf-8"))
    file_buffer.name = filename
    vector_file = openai_client.vector_stores.files.upload_and_poll(
        vector_store_id=vector_store_id,
        file=file_buffer,
        attributes={
            "source": "web_realtime_feedback",
            "session_hash": session_hash[:512],
            "item_id": item_id[:512],
        },
    )
    upload_status = str(getattr(vector_file, "status", "") or "").strip()
    if upload_status and upload_status != "completed":
        last_error = getattr(vector_file, "last_error", None)
        raise RuntimeError(f"Vector store file upload ended with status {upload_status}: {last_error}")

    return {
        "status": "uploaded",
        "vector_store_id": vector_store_id,
        "file_id": str(getattr(vector_file, "id", "") or ""),
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
    }


def upload_feedback_qa_records(feedback_records: list[dict]) -> list[dict]:
    results = []
    for record in feedback_records:
        if not isinstance(record, dict):
            continue
        try:
            result = upload_feedback_qa_pair(record)
        except Exception as exc:
            result = {"status": "error", "message": str(exc)}
            print(
                "[ERROR] Feedback vector store upload failed "
                f"| session={record.get('session_hash') or '-'} "
                f"| item={record.get('item_id') or '-'} | error={exc}",
                flush=True,
            )
        record["vector_store_update"] = result
        results.append(result)
    return results


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
    return render_template("feedback_realtime.html")


@app.get("/realtime")
def realtime_page():
    return render_template("feedback_realtime.html")


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
        instructions = build_response_instructions(kind, user_transcript, mood_aspect_state)
        instructions = append_tool_output_to_instructions(instructions, tool_output)
        print(
            "[REALTIME-PROMPT] "
            f"kind={kind}\n"
            f"{instructions}\n"
            "[/REALTIME-PROMPT]",
            flush=True,
        )
        return jsonify({
            "status": "ok",
            "kind": kind,
            "instructions": instructions,
            "mood_aspect_state": mood_aspect_state,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


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


@app.post("/api/realtime-feedback")
def save_realtime_feedback():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        session_hash = sanitize_session_hash(payload.get("account_id") or payload.get("session_hash")) or uuid4().hex[:24]
        item_id = str(payload.get("item_id") or "").strip()
        response_text = str(payload.get("response_text") or "").strip()
        feedback_text = str(payload.get("feedback_text") or "").strip()
        user_utterance = str(payload.get("user_utterance") or "").strip()

        if not feedback_text:
            return jsonify({
                "status": "error",
                "message": "feedback_text is required",
            }), 400

        date_prefix = datetime.now().strftime("%Y%m%d")
        session_dir_name = f"{date_prefix}_{session_hash}"
        saved_at = datetime.now().isoformat(timespec="seconds")
        ensure_realtime_session(
            DATABASE_PATH,
            session_hash,
            session_dir_name,
            saved_at,
            request.remote_addr,
            request.headers.get("User-Agent"),
        )

        record = {
            "session_hash": session_hash,
            "account_id": session_hash,
            "item_id": item_id or None,
            "response_text": response_text,
            "feedback_text": feedback_text,
            "user_utterance": user_utterance,
            "updated_at": saved_at,
            "remote_addr": request.remote_addr,
            "user_agent": request.headers.get("User-Agent"),
            "source": "web_realtime_feedback",
        }
        upsert_feedback_record(DATABASE_PATH, record)
        vector_store_update = upload_feedback_qa_records([record])[0]
        feedback_count = count_feedback_records(DATABASE_PATH, session_hash)

        print(f"[FEEDBACK] Saved | session={session_hash} | item={item_id or '-'}", flush=True)

        return jsonify({
            "status": "ok",
            "message": "feedback saved",
            "account_id": session_hash,
            "session_hash": session_hash,
            "session_id": session_hash,
            "session_dir_name": session_dir_name,
            "saved_file": "",
            "saved_to": "sqlite",
            "saved_at": saved_at,
            "feedback_count": feedback_count,
            "vector_store_update": vector_store_update,
        })

    except Exception as e:
        print(f"[ERROR] /api/realtime-feedback failed: {e}", flush=True)
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@app.post("/api/realtime-session")
def upload_realtime_session():
    try:
        session_hash = (
            sanitize_session_hash(request.form.get("account_id") or request.form.get("session_hash"))
            or uuid4().hex[:24]
        )
        date_prefix = datetime.now().strftime("%Y%m%d")
        session_dir_name = f"{date_prefix}_{session_hash}"
        session_dir = ensure_dir(UPLOAD_ROOT / session_dir_name)
        upload_received_at = datetime.now().isoformat(timespec="seconds")

        metadata = {}
        transcript_payload = {}
        feedback_payload = {}
        feedback_file_path = None
        feedback_records = []
        saved_files = []
        saved_file_names = []
        saved_file_paths = []
        for field_name in ["metadata_file", "transcript_file", "feedback_file"]:
            file = request.files.get(field_name)
            if file and file.filename:
                filename = secure_filename(file.filename) or f"{field_name}.bin"
                target = session_dir / filename
                raw_bytes = file.read()
                target.write_bytes(raw_bytes)
                raw_text = raw_bytes.decode("utf-8-sig", errors="replace")
                if field_name == "metadata_file":
                    try:
                        metadata = json.loads(raw_text)
                    except Exception:
                        metadata = {}
                if field_name == "transcript_file":
                    try:
                        transcript_payload = json.loads(raw_text)
                    except Exception:
                        transcript_payload = {}
                if field_name == "feedback_file":
                    feedback_file_path = target
                    try:
                        feedback_payload = json.loads(raw_text)
                    except Exception:
                        feedback_payload = {}
                    incoming_records = (
                        feedback_payload.get("feedback_records")
                        if isinstance(feedback_payload, dict)
                        else []
                    )
                    if isinstance(incoming_records, list):
                        for record in incoming_records:
                            if isinstance(record, dict):
                                record["session_hash"] = session_hash
                                record["account_id"] = session_hash
                                record.setdefault("remote_addr", request.remote_addr)
                                record.setdefault("user_agent", request.headers.get("User-Agent"))
                                record.setdefault("updated_at", upload_received_at)
                                record.setdefault("created_at", record.get("updated_at") or upload_received_at)
                                record.setdefault("source", "web_realtime_feedback")
                                feedback_records.append(record)
                saved_files.append(str(target))
                saved_file_names.append(filename)
                saved_file_paths.append({
                    "field_name": field_name,
                    "filename": filename,
                    "path": str(target),
                })

        for field_name in ["user_audio_file", "assistant_audio_file", "video_frames_file"]:
            file = request.files.get(field_name)
            if file and file.filename:
                filename = secure_filename(file.filename) or f"{field_name}.bin"
                target = session_dir / filename
                file.save(target)
                saved_files.append(str(target))
                saved_file_names.append(filename)
                saved_file_paths.append({
                    "field_name": field_name,
                    "filename": filename,
                    "path": str(target),
                })

        attach_previous_user_utterances(feedback_records, transcript_payload)
        vector_store_updates = upload_feedback_qa_records(feedback_records)
        vector_store_uploaded_count = sum(
            1 for result in vector_store_updates if result.get("status") == "uploaded"
        )
        vector_store_error_count = sum(
            1 for result in vector_store_updates if result.get("status") == "error"
        )
        if feedback_file_path is not None:
            if not isinstance(feedback_payload, dict):
                feedback_payload = {}
            feedback_payload["feedback_records"] = feedback_records
            feedback_payload["vector_store_updates"] = vector_store_updates
            feedback_payload["vector_store_uploaded_count"] = vector_store_uploaded_count
            feedback_payload["vector_store_error_count"] = vector_store_error_count
            write_json_file(feedback_file_path, feedback_payload)

        upsert_realtime_session(
            DATABASE_PATH,
            {
                "session_hash": session_hash,
                "account_id": session_hash,
                "session_dir_name": session_dir_name,
                "created_at": metadata.get("started_at_iso") if isinstance(metadata, dict) else upload_received_at,
                "updated_at": upload_received_at,
                "remote_addr": request.remote_addr,
                "user_agent": request.headers.get("User-Agent"),
                "saved_file_names": saved_file_names,
                "saved_file_paths": saved_file_paths,
            },
        )
        upsert_feedback_records(DATABASE_PATH, feedback_records)
        feedback_count = count_feedback_records(DATABASE_PATH, session_hash)

        print(f"[UPLOAD] Done   | session={session_hash} | files_saved={len(saved_files)}", flush=True)

        return jsonify(
            {
                "status": "ok",
                "message": "upload saved",
                "account_id": session_hash,
                "session_hash": session_hash,
                "session_id": session_hash,
                "session_dir_name": session_dir_name,
                "saved_count": len(saved_files),
                "saved_files": saved_files,
                "saved_file_paths": saved_file_paths,
                "feedback_count": feedback_count,
                "vector_store_updates": vector_store_updates,
                "vector_store_uploaded_count": vector_store_uploaded_count,
                "vector_store_error_count": vector_store_error_count,
                "saved_to": "filesystem+sqlite",
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
        "database_path": str(DATABASE_PATH),
        "traditional_converter_available": get_traditional_converter() is not None,
        "traditional_converter_error": traditional_converter_error,
    })


if __name__ == "__main__":
    initialize_database(DATABASE_PATH)
    initialize_openai_client()
    app.run(host="0.0.0.0", port=9051, debug=True)
else:
    initialize_database(DATABASE_PATH)
    initialize_openai_client()
