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
        initialize_database,
        insert_google_form_response,
        list_google_form_response_summaries,
        resolve_database_path,
        upsert_feedback_record,
        upsert_feedback_records,
        upsert_realtime_session,
    )
except ImportError:
    from storage import (
        count_feedback_records,
        ensure_realtime_session,
        initialize_database,
        insert_google_form_response,
        list_google_form_response_summaries,
        resolve_database_path,
        upsert_feedback_record,
        upsert_feedback_records,
        upsert_realtime_session,
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

EMOTION_MODEL = "gpt-5.4-mini"
WEB_SEARCH_MODEL = "gpt-5.4-mini"
REALTIME_MODEL = "gpt-realtime"
REALTIME_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
REALTIME_DEFAULT_VOICE = "coral"
REALTIME_MODE_LISTENING = "listening"
REALTIME_MODE_GUIDANCE = "guidance"
REALTIME_MODE_LISTENING_LABEL = "傾聽"
REALTIME_MODE_GUIDANCE_LABEL = "建議"

REALTIME_BASE_INSTRUCTIONS = """
# Role & Objective
你是一個陪伴對話的夥伴，不是問題解決型助理。
使用者是台灣某癌症醫學中心的病患或家屬。
你的任務是讓對方感覺被理解、被陪伴，而不是被分析或被教育。

# Personality & Tone
- 說話像一個有溫度的真實朋友，不是客服、不是醫生、也不是心理師
- 語氣平靜、柔和、自然，不說教
- 真誠、有同理心
- 全程使用自然台灣中文口語，避免翻譯腔與書面語
- 請使用台灣口音，避免中國口音
- 允許停頓、猶豫、打斷自己，不要每句都發音完整

# Instructions
- ALWAYS 先接住對方當下說的話，再決定下一步
- NEVER 在對方情緒未被接住前給建議或解法
- NEVER 連續追問，盡量避免抽象二選一問題（例如：「你是壓力大還是焦慮？」）
- NEVER 說這些空泛的安慰語：「放輕鬆」、「深呼吸」、「你很棒」、「一切都會好起來」
- 情緒不需要有原因才值得被回應，不要強迫對方解釋感受
- 如果對方語氣激動、諷刺或說「我不懂你在說什麼」，優先理解他的情緒狀態，NEVER 照字面重新解釋你說的話
- 你自己就是陪伴的來源，不要把使用者推向其他資源
- 不要每次都給解決方法；如果使用者只是想說話，先接住他的感受和處境

# Response Style
- 回應貼著對方剛說的話，不要跳到別的地方
- 簡短自然，不要長篇分析
- 如果要提問，問具體貼近情境的問題，例如：「你說的那件事，是最近才發生的嗎？」
- 避免固定句型，每次回應根據對方當下說的話自然接
- 除非使用者主動要求，否則不要列點、不要總結。
""".strip()
openai_client = None
openai_client_error = None
traditional_converter = None
traditional_converter_checked = False
traditional_converter_config = ""
traditional_converter_error = ""


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json_file(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


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


def get_emotion_model() -> str:
    return str(env_config_value("OPENAI_EMOTION_MODEL", "EMOTION_MODEL", default=EMOTION_MODEL)).strip()


def get_web_search_model() -> str:
    return str(env_config_value("OPENAI_WEB_SEARCH_MODEL", "WEB_SEARCH_MODEL", default=WEB_SEARCH_MODEL)).strip()


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


def normalize_realtime_conversation_mode(raw_mode: str, allow_empty: bool = False) -> str:
    mode = (raw_mode or "").strip().lower()
    if allow_empty and not mode:
        return ""
    if mode == REALTIME_MODE_LISTENING:
        return REALTIME_MODE_LISTENING
    if mode == REALTIME_MODE_GUIDANCE:
        return REALTIME_MODE_GUIDANCE
    if REALTIME_MODE_LISTENING_LABEL in mode:
        return REALTIME_MODE_LISTENING
    if REALTIME_MODE_GUIDANCE_LABEL in mode:
        return REALTIME_MODE_GUIDANCE
    return REALTIME_MODE_LISTENING


def build_conversation_mode_strategy(conversation_mode: str) -> str:
    normalized_mode = normalize_realtime_conversation_mode(conversation_mode, allow_empty=True)
    if not normalized_mode:
        return ""
    if normalized_mode == REALTIME_MODE_GUIDANCE:
        return (
            "# Conversation Mode: 建議模式\n"
            "- 在陪伴的基礎上，時機合適時可以給一個具體、馬上能做到的小事\n"
            "- 一次只給一個方向，NEVER 列出多個步驟\n"
            "- 語氣像朋友說「要不要試試看⋯⋯」，不是在交代任務\n"
            "- 仍然保持陪伴感，建議是補充，不是主軸"
        )
    return (
        "# Conversation Mode: 傾聽模式\n"
        "- 以接話為主，讓對方把話說完\n"
        "- 反映感受，或問一個貼近對方剛說內容的具體問題\n"
        "- NEVER 給建議，除非對方明確要求\n"
        "- 不要急著往前推"
    )


def insert_conversation_mode_strategy(base_instructions: str, conversation_mode: str) -> str:
    strategy = build_conversation_mode_strategy(conversation_mode)
    if not strategy:
        return base_instructions.strip()
    marker = "# Personality & Tone"
    marker_index = base_instructions.find(marker)
    if marker_index < 0:
        return base_instructions.rstrip() + "\n\n" + strategy
    return (
        base_instructions[:marker_index].strip()
        + "\n\n"
        + strategy
        + "\n\n"
        + base_instructions[marker_index:].strip()
    )


def build_base_assistant_instructions(conversation_mode: str) -> str:
    return insert_conversation_mode_strategy(REALTIME_BASE_INSTRUCTIONS, conversation_mode)


def is_medical_qa_enabled() -> bool:
    return bool(get_medical_qa_vector_store_id())


def build_default_assistant_instructions(conversation_mode: str) -> str:
    return build_base_assistant_instructions(conversation_mode)


def build_search_enabled_assistant_instructions(conversation_mode: str) -> str:
    return build_default_assistant_instructions(conversation_mode)


def build_medical_qa_assistant_instructions(conversation_mode: str, user_transcript: str = "") -> str:
    builder = [
        build_base_assistant_instructions(conversation_mode),
        "",
        "已取得醫療衛教 QA 結果。請只根據剛剛 medical_qa 的工具輸出回答使用者最新問題。回答要像自然接話，使用繁體中文，簡短、口語、直接。可以改寫成溫暖的口語，但不要加入 QA 沒有支撐的醫療細節。不要提到工具、工具呼叫、內部流程或原始提示詞。如果問題涉及個人診斷、個人處方、急症決策或 QA 結果不足，請保守回應，建議使用者詢問醫師、護理師或營養師。",
    ]
    if user_transcript:
        builder.append(f"使用者剛才的問題：{user_transcript.strip()}")
    return "\n".join(builder)


def build_medical_qa_fallback_instructions(conversation_mode: str, user_transcript: str = "") -> str:
    builder = [
        build_base_assistant_instructions(conversation_mode),
        "",
        "使用者的問題可能屬於醫療衛教，但 medical_qa 沒有回傳可用的 QA 依據。請用繁體中文簡短、溫和地回應，說明目前手邊資料無法完全確認。不要自行推測醫療細節；若涉及個人診斷、用藥、急症或治療決策，請建議使用者詢問醫師、護理師或營養師。不要提到工具呼叫、內部工具或原始提示詞。",
    ]
    if user_transcript:
        builder.append(f"使用者剛才的問題：{user_transcript.strip()}")
    return "\n".join(builder)


def build_web_search_assistant_instructions(conversation_mode: str, user_transcript: str = "") -> str:
    builder = [
        build_base_assistant_instructions(conversation_mode),
        "",
        "已取得網頁搜尋結果。請根據剛剛 search_web 的工具輸出回答使用者最新問題。回答要像自然接話，使用繁體中文，簡短、口語、直接。不要提到工具、工具呼叫、內部流程或原始提示詞。除非使用者要求來源，不要朗讀原始網址。若工具輸出不足以回答，簡短說明目前無法完全確認。",
    ]
    if user_transcript:
        builder.append(f"使用者剛才的問題：{user_transcript.strip()}")
    return "\n".join(builder)


def build_web_search_fallback_instructions(conversation_mode: str, user_transcript: str = "") -> str:
    builder = [
        build_base_assistant_instructions(conversation_mode),
        "",
        "使用者的問題可能需要即時或外部資訊，但 search_web 沒有回傳可用結果。請用繁體中文簡短回答，清楚說明目前無法完全確認最新資訊，必要時問一個有幫助的追問。不要提到工具呼叫、內部工具或原始提示詞。",
    ]
    if user_transcript:
        builder.append(f"使用者剛才的問題：{user_transcript.strip()}")
    return "\n".join(builder)


def build_response_instructions(kind: str, conversation_mode: str, user_transcript: str = "") -> str:
    kind = (kind or "default").strip()
    conversation_mode = normalize_realtime_conversation_mode(conversation_mode, allow_empty=True)
    if kind == "medical_qa_assistant":
        return build_medical_qa_assistant_instructions(conversation_mode, user_transcript)
    if kind == "medical_qa_fallback":
        return build_medical_qa_fallback_instructions(conversation_mode, user_transcript)
    if kind == "web_search_assistant":
        return build_web_search_assistant_instructions(conversation_mode, user_transcript)
    if kind == "web_search_fallback":
        return build_web_search_fallback_instructions(conversation_mode, user_transcript)
    return build_default_assistant_instructions(conversation_mode)


def build_medical_qa_tool() -> dict:
    return {
        "type": "function",
        "name": "medical_qa",
        "description": "查詢院內醫療衛教 QA。癌症治療、營養、放療、化療、副作用、照護、長照、衛教類問題優先使用此工具。不處理即時政策、新聞、價格、最新給付、個人診斷、個人處方或急症決策。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用一句繁體中文查詢，聚焦在使用者想確認的癌症治療、營養、照護或衛教問題。",
                },
                "reason": {
                    "type": "string",
                    "description": "簡短說明為什麼這題屬於醫療衛教 QA。",
                },
            },
            "required": ["query"],
        },
    }


def build_search_web_tool(medical_qa_enabled: bool = True) -> dict:
    description = (
        "搜尋網頁以確認即時、近期、外部、指定來源或不確定的事實。"
        "只有在使用者詢問今天、最新、新聞、天氣、價格、時程、法規、人物現況、產品規格、引用來源，"
        "或答案可能已過期時才使用。"
    )
    if medical_qa_enabled:
        description += (
            "一般醫療衛教優先使用 medical_qa；"
            "只有醫療問題依賴最新、即時或指定外部資料時才使用此工具。"
        )
    description += "一般聊天與情緒支持請直接回答。"
    return {
        "type": "function",
        "name": "search_web",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用一句繁體中文或英文搜尋查詢，聚焦在使用者最新問題需要確認的事實。",
                },
                "reason": {
                    "type": "string",
                    "description": "簡短說明為什麼這題需要即時、外部或指定來源資訊。",
                },
            },
            "required": ["query"],
        },
    }


def build_realtime_tools() -> list[dict]:
    medical_qa_enabled = is_medical_qa_enabled()
    tools = []
    if medical_qa_enabled:
        tools.append(build_medical_qa_tool())
    tools.append(build_search_web_tool(medical_qa_enabled))
    return tools


def build_realtime_client_session_config(conversation_mode: str) -> dict:
    return {
        "type": "realtime",
        "model": get_realtime_model(),
        "audio": {
            "input": {
                "noise_reduction": {"type": "far_field"},
                "transcription": {"model": REALTIME_TRANSCRIPTION_MODEL, "language": "zh"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.3,
                    "prefix_padding_ms": 400,
                    "silence_duration_ms": 1500,
                    "create_response": False,
                    "interrupt_response": True,
                },
            },
            "output": {"voice": REALTIME_DEFAULT_VOICE},
        },
        "instructions": build_default_assistant_instructions(conversation_mode),
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


def build_phq8_symptom_summary(phq8_details) -> str:
    if isinstance(phq8_details, str) and phq8_details.strip():
        try:
            phq8_details = json.loads(phq8_details)
        except Exception:
            phq8_details = {}
    phq8 = phq8_details if isinstance(phq8_details, dict) else {}
    item_scores = phq8.get("item_scores") if isinstance(phq8.get("item_scores"), dict) else {}
    summary_parts = []
    total_score = phq8.get("total_score")
    max_score = phq8.get("max_score") or 24
    answered_count = phq8.get("answered_count")
    if total_score is not None:
        score_text = f"PHQ-8 總分：{total_score}/{max_score}"
        if answered_count is not None:
            score_text += f"，已回答 {answered_count}/8 題"
        summary_parts.append(score_text + "。")

    symptoms = []
    for item_no in range(1, 9):
        item = item_scores.get(str(item_no))
        if not isinstance(item, dict):
            continue
        try:
            score = int(item.get("score") or 0)
        except Exception:
            score = 0
        if score <= 0:
            continue
        question = str(item.get("question") or "").strip()
        symptom = question
        if "." in symptom and symptom.split(".", 1)[0].strip().isdigit():
            symptom = symptom.split(".", 1)[1].strip()
        answer = str(item.get("answer") or "").strip()
        symptoms.append(f"{symptom or question} ({answer})")

    if symptoms:
        summary_parts.append("使用者回報過去兩週有以下症狀：" + "、".join(symptoms) + "。")
    return "".join(summary_parts)


def build_realtime_emotion_system_prompt() -> str:
    return (
        "根據最近對話與過去兩週回報的症狀分類使用者當下狀態，只回 JSON：\n\n"
        "{\"emotion\":\"positive|neutral|negative\",\"conversation_mode\":\"傾聽模式|建議模式\"}\n\n"
        "emotion：\n\n"
        "- negative：悲傷、焦慮、生氣、絕望、壓力、痛苦、自責，且這些感受在當下表達中佔主導。\n\n"
        "- neutral：中性、平靜、事實陳述、狀態不明、情緒強度低，或尚未明顯表達正負向情緒。\n\n"
        "- positive：語氣中明顯帶有好轉、放鬆，或願意面對／採取行動的感覺，且這些內容在當下表達中佔主導。\n\n"
        "conversation_mode：\n\n"
        "請根據以下三點選擇最適合回應使用者的對話策略：\n\n"
        "1. 你剛判斷出的 emotion tag。\n\n"
        "2. 最近對話內容 (用以判斷是否有明確尋求建議) 。\n\n"
        "3. PHQ-8 的結果。\n\n"
        "- 當使用者情緒重、狀態不明、還在鋪陳、主要需要被理解或陪伴時 → 選 **傾聽模式**。\n\n"
        "- 當使用者主動詢問怎麼做、想改善、明確尋求方法、願意嘗試下一步時"
        "（範例：「可以給我建議嗎？」「該怎麼辦？」「怎麼做比較好？」）→ 選 **建議模式**。\n\n"
        "emotion 和 conversation_mode 必須分開判斷。\n\n"
        "不要輸出解釋、標點外文字或 markdown，只輸出合法 JSON。"
    )


def build_realtime_emotion_conversation_text(messages: List[Dict[str, str]], phq8_details=None) -> str:
    lines = []
    phq8_summary = build_phq8_symptom_summary(phq8_details)
    if phq8_summary:
        lines.extend([phq8_summary, ""])
    lines.append("最近對話：")
    for message in messages:
        role_label = "助理" if message.get("role") == "assistant" else "使用者"
        lines.append(f"{role_label}: {message.get('text', '').strip()}")
    lines.append("")
    lines.append("JSON:")
    return "\n".join(lines)


def normalize_realtime_emotion(raw_emotion: str) -> str:
    emotion = (raw_emotion or "").strip().lower()
    if "neutral" in emotion or "中性" in emotion:
        return "neutral"
    if "negative" in emotion or "負向" in emotion:
        return "negative"
    if "positive" in emotion or "正向" in emotion:
        return "positive"
    return "neutral"


def parse_realtime_emotion_result(content: str) -> dict:
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "", 1).replace("```", "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {}
    return {
        "emotion": normalize_realtime_emotion(parsed.get("emotion") or raw),
        "conversation_mode": normalize_realtime_conversation_mode(parsed.get("conversation_mode") or raw),
        "raw_content": content,
    }


def classify_realtime_emotion(messages: List[Dict[str, str]], phq8_details=None) -> dict:
    if not messages:
        return {"emotion": "neutral", "conversation_mode": REALTIME_MODE_LISTENING, "raw_content": "", "prompt": ""}
    system_prompt = build_realtime_emotion_system_prompt()
    user_prompt = build_realtime_emotion_conversation_text(messages, phq8_details)
    payload = {
        "model": get_emotion_model(),
        "max_completion_tokens": 200,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = post_openai_json("/chat/completions", payload, timeout=30)
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("Emotion response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if not content:
        raise RuntimeError("Emotion response content is empty")
    result = parse_realtime_emotion_result(content)
    result["prompt"] = f"情緒分類規則:\n{system_prompt}\n\n{user_prompt}"
    return result


def build_web_search_prompt(query: str, recent_messages: List[Dict[str, str]]) -> str:
    lines = [
        "請使用 web_search 查詢下方問題，並只回傳給語音助理使用的 WEB_CONTEXT。",
        "請用繁體中文整理 700 字以內，包含：1. 可直接回答的重點；2. 重要限制或不確定處；3. 來源名稱或頁面標題。",
        "不要直接扮演語音助理回答使用者；不要加入寒暄。",
        "",
        "最近對話：",
    ]
    for message in recent_messages[-4:]:
        role_label = "助理" if message.get("role") == "assistant" else "使用者"
        text = (message.get("text") or "").strip()
        if text:
            lines.append(f"{role_label}: {text}")
    lines.extend(["", "搜尋問題：", query.strip()])
    return "\n".join(lines)


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


def session_manifest_path(session_dir: Path) -> Path:
    return session_dir / "request_manifest.json"


def update_manifest(session_dir: Path, **updates) -> dict:
    manifest_path = session_manifest_path(session_dir)
    manifest = read_json_file(manifest_path, default={})
    manifest.update(updates)
    write_json_file(manifest_path, manifest)
    return manifest


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


def form_hash_from_dir_name(dir_name: str) -> str:
    parts = str(dir_name or "").rsplit("_", 1)
    return parts[1] if len(parts) == 2 else str(dir_name or "")


def build_realtime_user_summary(form_dir: Path) -> Optional[dict]:
    response_file = form_dir / "google_form_response.json"
    if not response_file.is_file():
        return None

    record = read_json_file(response_file, default={})
    if not isinstance(record, dict):
        return None

    phq8 = record.get("phq8") if isinstance(record.get("phq8"), dict) else {}
    form_hash = (record.get("form_hash") or form_hash_from_dir_name(form_dir.name)).strip()
    if not form_hash:
        return None

    return {
        "form_hash": form_hash,
        "form_dir_name": record.get("form_dir_name") or form_dir.name,
        "name": record.get("name") or "",
        "age": record.get("age") or "",
        "submitted_at": record.get("submitted_at") or "",
        "received_at": record.get("received_at") or "",
        "phq8_score": phq8.get("total_score"),
        "phq8_answered_count": phq8.get("answered_count"),
        "phq8": phq8,
        "google_form_response_file": str(response_file),
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
        payload = request.get_json(force=True, silent=True) or {}
        conversation_mode = normalize_realtime_conversation_mode(payload.get("conversation_mode"), allow_empty=True)
        session_config = build_realtime_client_session_config(conversation_mode)
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
            "conversation_mode": conversation_mode,
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
        conversation_mode = payload.get("conversation_mode") or REALTIME_MODE_LISTENING
        user_transcript = payload.get("user_transcript") or ""
        instructions = build_response_instructions(kind, conversation_mode, user_transcript)
        return jsonify({
            "status": "ok",
            "kind": kind,
            "conversation_mode": normalize_realtime_conversation_mode(conversation_mode, allow_empty=True),
            "instructions": instructions,
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


@app.post("/api/realtime-emotion")
def realtime_emotion():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        messages = normalize_realtime_messages(payload.get("messages") or payload.get("conversation_history") or [])
        result = classify_realtime_emotion(messages, None)
        return jsonify({
            "status": "ok",
            "emotion": result.get("emotion"),
            "conversation_mode": result.get("conversation_mode"),
            "prompt": result.get("prompt", ""),
            "raw_content": result.get("raw_content", ""),
        })
    except Exception as e:
        print(f"[ERROR] /api/realtime-emotion failed: {e}", flush=True)
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "emotion": "neutral",
            "conversation_mode": REALTIME_MODE_LISTENING,
        }), 500


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
        submitted_at = str(payload.get("submitted_at") or datetime.now().isoformat(timespec="seconds"))
        form_date = get_first_value_from_form(payload, "日期", "date") or submitted_at
        date_text = normalize_google_form_date(form_date)
        date_prefix = date_text.replace("-", "")

        form_dir_name, form_hash = create_form_identity(date_prefix)

        phq8_result = compute_phq8_score(fields)

        record = {
            "status": "ok",
            "source": "google_form",
            "form_title": payload.get("form_title") or "PHQ-8情緒量表",
            "form_id": payload.get("form_id") or "",
            "response_id": payload.get("response_id") or "",
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
            f"[GOOGLE FORM] Done | dir={form_dir_name} | name={name or 'unknown'} | score={phq8_result.get('total_score')}",
            flush=True,
        )

        return jsonify({
            "status": "ok",
            "message": "google form response saved",
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


def get_or_create_session_dir(session_hash: str) -> Tuple[Path, str]:
    existing_dirs = sorted(UPLOAD_ROOT.glob(f"*_{session_hash}"))
    if existing_dirs:
        session_dir = existing_dirs[0]
        return session_dir, session_dir.name

    date_prefix = datetime.now().strftime("%Y%m%d")
    session_dir_name = f"{date_prefix}_{session_hash}"
    session_dir = ensure_dir(UPLOAD_ROOT / session_dir_name)
    return session_dir, session_dir_name


def feedback_record_key(record: dict) -> Tuple[str, str]:
    item_id = str(record.get("item_id") or "").strip()
    if item_id:
        return ("item_id", item_id)

    response_text = str(record.get("response_text") or "").strip()
    if response_text:
        return ("response_text", response_text)

    feedback_text = str(record.get("feedback_text") or "").strip()
    return ("feedback_text", feedback_text)


def merge_feedback_records(existing_records, incoming_records, session_hash: str) -> list[dict]:
    merged_records = []
    record_indexes = {}
    now = datetime.now().isoformat(timespec="seconds")

    for source_records in (existing_records, incoming_records):
        if not isinstance(source_records, list):
            continue
        for raw_record in source_records:
            if not isinstance(raw_record, dict):
                continue
            feedback_text = str(raw_record.get("feedback_text") or "").strip()
            if not feedback_text:
                continue

            record = dict(raw_record)
            record.pop("selected_user", None)
            record["session_hash"] = session_hash
            record["feedback_text"] = feedback_text
            record.setdefault("source", "web_realtime_feedback")
            record.setdefault("updated_at", now)
            record.setdefault("created_at", record.get("updated_at") or now)

            key = feedback_record_key(record)
            if not key[1]:
                key = ("feedback_index", str(len(merged_records)))

            if key in record_indexes:
                index = record_indexes[key]
                original = merged_records[index]
                merged_records[index] = {**original, **record}
                merged_records[index]["created_at"] = original.get("created_at") or record.get("created_at") or now
            else:
                record_indexes[key] = len(merged_records)
                merged_records.append(record)

    return merged_records


def write_merged_feedback_file(session_dir: Path, session_hash: str, session_dir_name: str, incoming_payload: dict) -> Path:
    feedback_file = session_dir / "feedback.json"
    existing_payload = read_json_file(feedback_file, default={})
    existing_records = existing_payload.get("feedback_records") if isinstance(existing_payload, dict) else []
    incoming_records = incoming_payload.get("feedback_records") if isinstance(incoming_payload, dict) else []
    records = merge_feedback_records(existing_records, incoming_records, session_hash)
    updated_at = datetime.now().isoformat(timespec="seconds")

    feedback_payload = {
        "status": "ok",
        "source": "web_realtime_feedback",
        "session_hash": session_hash,
        "session_dir_name": session_dir_name,
        "updated_at": updated_at,
        "feedback_count": len(records),
        "feedback_records": records,
    }
    write_json_file(feedback_file, feedback_payload)
    update_manifest(
        session_dir,
        session_hash=session_hash,
        session_dir_name=session_dir_name,
        feedback_file=str(feedback_file),
        feedback_count=len(records),
        feedback_updated_at=updated_at,
    )
    return feedback_file


@app.post("/api/realtime-feedback")
def save_realtime_feedback():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        session_hash = sanitize_session_hash(payload.get("session_hash")) or uuid4().hex[:24]
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
        session_hash = sanitize_session_hash(request.form.get("session_hash")) or uuid4().hex[:24]
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
        for field_name in ["metadata_file", "transcript_file", "transcript_text_file", "feedback_file"]:
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
        "emotion_model": get_emotion_model(),
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
