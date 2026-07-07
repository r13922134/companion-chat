from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

try:
    from vendor.mllm_dr.prompts import (
        PHQ8_ASPECTS,
        PHQ8_CLINICAL_DESCRIPTIONS,
        build_dialogue,
    )
except ImportError:
    from mllm_dr.prompts import (
        PHQ8_ASPECTS,
        PHQ8_CLINICAL_DESCRIPTIONS,
        build_dialogue,
    )

try:
    from .prompts import (
        CLINICAL_ASSISTANT_SYSTEM_PROMPT,
        DEPRESSION_ADAPTIVE_REFLECT_HEURISTIC_SUFFIX,
        DEPRESSION_ASPECT_ADAPTIVE_REFLECT_QUERY_PROMPT,
        DEPRESSION_ASPECT_CLINICAL_GRAPH_QUERY_PROMPT,
        DEPRESSION_ASPECT_PROFILE_PROMPT,
        DEPRESSION_ASPECT_QUERY_PROMPT,
        DEPRESSION_ASPECT_QUERY_PROMPT_VERSION,
        DEPRESSION_ASPECT_QUERY_PROMPTS,
        DEPRESSION_CLINICAL_GRAPH_HEURISTIC_SUFFIX,
        append_depression_optional_profile_context,
        build_default_depression_evidence_query,
        build_depression_prioritized_evidence_query,
        build_depression_aspect_query_prompt,
        build_depression_profile_prompt,
    )
except ImportError:
    from prompts import (
        CLINICAL_ASSISTANT_SYSTEM_PROMPT,
        DEPRESSION_ADAPTIVE_REFLECT_HEURISTIC_SUFFIX,
        DEPRESSION_ASPECT_ADAPTIVE_REFLECT_QUERY_PROMPT,
        DEPRESSION_ASPECT_CLINICAL_GRAPH_QUERY_PROMPT,
        DEPRESSION_ASPECT_PROFILE_PROMPT,
        DEPRESSION_ASPECT_QUERY_PROMPT,
        DEPRESSION_ASPECT_QUERY_PROMPT_VERSION,
        DEPRESSION_ASPECT_QUERY_PROMPTS,
        DEPRESSION_CLINICAL_GRAPH_HEURISTIC_SUFFIX,
        append_depression_optional_profile_context,
        build_default_depression_evidence_query,
        build_depression_prioritized_evidence_query,
        build_depression_aspect_query_prompt,
        build_depression_profile_prompt,
    )

QUERY_PROMPT_VERSION = DEPRESSION_ASPECT_QUERY_PROMPT_VERSION
PROFILE_PROMPT = DEPRESSION_ASPECT_PROFILE_PROMPT
QUERY_PROMPT = DEPRESSION_ASPECT_QUERY_PROMPT
CLINICAL_GRAPH_QUERY_PROMPT = DEPRESSION_ASPECT_CLINICAL_GRAPH_QUERY_PROMPT
ADAPTIVE_REFLECT_QUERY_PROMPT = DEPRESSION_ASPECT_ADAPTIVE_REFLECT_QUERY_PROMPT
QUERY_PROMPTS = DEPRESSION_ASPECT_QUERY_PROMPTS


@dataclass(frozen=True)
class TranscriptUtterance:
    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class RetrievedUtterance:
    utterance: TranscriptUtterance
    score: float
    raw_score: float | None = None
    evidence_rerank_bonus: float = 0.0
    aspect_keyword_hits: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateFilterResult:
    utterances: list[TranscriptUtterance]
    summary: dict[str, Any]


@dataclass(frozen=True)
class AspectRetrievalConfig:
    min_utterances: int = 5
    max_utterances: int = 20
    max_dialogue_tokens: int = 1536
    min_score: float = -0.55
    relative_score_margin: float = 0.06
    candidate_filter: str = "participant_evidence"
    candidate_filter_min_words: int = 3
    min_candidate_utterances: int = 8
    query_prompt_version: str = QUERY_PROMPT_VERSION
    context_window: int = 0
    adaptive_reflect: bool = False
    evidence_rerank: str = "none"
    min_score_applies_to_min: bool = False
    global_fallback_mode: str = "clinical_v1"
    global_fallback_utterances: int = 0
    global_fallback_max_dialogue_tokens: int = 256
    max_profile_chars: int = 24000


@dataclass(frozen=True)
class AspectRetrievalOutput:
    records: list[dict[str, Any]]
    raw_utterance_count: int
    candidate_filter_summary: dict[str, Any]
    backend_name: str
    backend_model_name: str
    fallback_reason: str | None = None


class RetrievalBackend(Protocol):
    name: str
    model_name: str

    def generate_profile(self, dialogue: str) -> str:
        ...

    def generate_query(
        self,
        *,
        profile: str,
        aspect: str,
        aspect_description: str,
        query_prompt_version: str = QUERY_PROMPT_VERSION,
    ) -> str:
        ...

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...

    def token_count(self, text: str) -> int:
        ...


QUESTION_OPENERS = {
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "do",
    "does",
    "did",
    "are",
    "is",
    "was",
    "were",
    "can",
    "could",
    "would",
    "will",
    "have",
    "has",
    "had",
}
DIRECTIVE_OPENERS = {"tell", "describe", "explain", "list", "share", "give", "talk"}
SECOND_PERSON_WORDS = {"you", "your", "yours", "yourself"}
FIRST_PERSON_WORDS = {
    "i",
    "im",
    "i'm",
    "ive",
    "i've",
    "id",
    "i'd",
    "ill",
    "i'll",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "we're",
    "weve",
    "we've",
    "us",
    "our",
    "ours",
}
ANSWER_CUE_WORDS = {"yes", "yeah", "yep", "no", "nope", "not", "never", "sometimes", "often", "always"}
CLINICAL_EVIDENCE_RE = re.compile(
    r"\b(?:"
    r"angry|anxious|appetite|asleep|blame|concentrat|confus|"
    r"depress|difficult|down|eat|edgy|energy|failure|fatigue|"
    r"focus|guilt|guilty|hopeless|irritable|joy|letharg|lonely|"
    r"negative|pain|ptsd|regret|restless|sad|shy|sleep|slow|"
    r"stress|tired|wake|weight|worth"
    r")\b",
    re.IGNORECASE,
)
SESSION_SETUP_RE = re.compile(
    r"\b(?:"
    r"ask(?:ing)?\s+you|ask\s+if\s+you|some\s+questions?|"
    r"going\s+to\s+ask|interview|record(?:ing)?|avatar|ellie|"
    r"would\s+love\s+to\s+learn\s+about\s+you"
    r")\b",
    re.IGNORECASE,
)
STRICT_SETUP_FRAGMENT_RE = re.compile(
    r"\b(?:"
    r"audio\s+recognition|recognition\s+system|system\s+is\s+working|"
    r"press\s+(?:that\s+)?button|button\s+every\s+once|"
    r"(?:she|it)\s+is\s+frozen|pauses?\s+for\s+a\s+very\s+long\s+time|"
    r"do\s+i\s+need\s+to\s+see\s+something|virtual\s+human|"
    r"can\s+taking\s+one\s+now|go\s+ahead\s+and\s+press"
    r")\b",
    re.IGNORECASE,
)
POSITIVE_COUNTER_RE = re.compile(
    r"\b(?:"
    r"pretty\s+good|not\s+depressed|no\s+not\s+down|fun\s+happy|"
    r"look\s+forward|big\s+relief|good\s+night\s+sleep|"
    r"decent\s+amount\s+of\s+sleep|positive\s+outlook"
    r")\b",
    re.IGNORECASE,
)

ASPECT_KEYWORD_PATTERNS: dict[int, tuple[tuple[str, re.Pattern[str]], ...]] = {
    0: (("anhedonia", re.compile(r"\b(?:little\s+interest|no\s+interest|stopped|withdraw|lack\s+of\s+(?:interest|motivation)|not\s+enjoy|enjoy(?:ment)?|pleasure|motivat)\b", re.I)),),
    1: (("low_mood", re.compile(r"\b(?:sad|down|depress|hopeless|empty|cry|irritable|grumpy)\b", re.I)),),
    2: (("sleep", re.compile(r"\b(?:sleep|asleep|wake|waking|night|insomnia|rest|routine)\b", re.I)),),
    3: (("energy", re.compile(r"\b(?:tired|fatigue|energy|exhaust|letharg|sluggish|drained)\b", re.I)),),
    4: (("appetite", re.compile(r"\b(?:appetite|eat|eating|food|hunger|hungry|weight|overeating|meal)\b", re.I)),),
    5: (("self_blame", re.compile(r"\b(?:guilt|guilty|regret|failure|blame|worthless|not\s+being\s+(?:a\s+)?good|appreciation)\b", re.I)),),
    6: (("concentration", re.compile(r"\b(?:concentrat|focus|remember|confus|fog|distract|attention|disorganiz|fragmented)\b", re.I)),),
    7: (("psychomotor", re.compile(r"\b(?:slow|slowed|sluggish|restless|fidget|move|moving|movement|speech|speaking|pace|pacing|irritable|grumpy)\b", re.I)),),
}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name) or default).strip())
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name) or default).strip())
    except Exception:
        return default


def config_from_env() -> AspectRetrievalConfig:
    return AspectRetrievalConfig(
        min_utterances=env_int("DEPRESSION_ASPECT_RETRIEVAL_MIN_UTTERANCES", 5),
        max_utterances=env_int("DEPRESSION_ASPECT_RETRIEVAL_MAX_UTTERANCES", 20),
        max_dialogue_tokens=env_int("DEPRESSION_ASPECT_RETRIEVAL_MAX_DIALOGUE_TOKENS", 1536),
        min_score=env_float("DEPRESSION_ASPECT_RETRIEVAL_MIN_SCORE", -0.55),
        relative_score_margin=env_float("DEPRESSION_ASPECT_RETRIEVAL_RELATIVE_SCORE_MARGIN", 0.06),
        candidate_filter=str(os.environ.get("DEPRESSION_ASPECT_RETRIEVAL_CANDIDATE_FILTER") or "participant_evidence"),
        candidate_filter_min_words=env_int("DEPRESSION_ASPECT_RETRIEVAL_CANDIDATE_FILTER_MIN_WORDS", 3),
        min_candidate_utterances=env_int("DEPRESSION_ASPECT_RETRIEVAL_MIN_CANDIDATE_UTTERANCES", 8),
        query_prompt_version=str(os.environ.get("DEPRESSION_ASPECT_RETRIEVAL_QUERY_PROMPT_VERSION") or QUERY_PROMPT_VERSION),
        context_window=env_int("DEPRESSION_ASPECT_RETRIEVAL_CONTEXT_WINDOW", 0),
        adaptive_reflect=env_bool("DEPRESSION_ASPECT_RETRIEVAL_ADAPTIVE_REFLECT", False),
        evidence_rerank=str(os.environ.get("DEPRESSION_ASPECT_RETRIEVAL_EVIDENCE_RERANK") or "none"),
        min_score_applies_to_min=env_bool("DEPRESSION_ASPECT_RETRIEVAL_MIN_SCORE_APPLIES_TO_MIN", False),
        global_fallback_mode=str(os.environ.get("DEPRESSION_ASPECT_RETRIEVAL_GLOBAL_FALLBACK_MODE") or "clinical_v1"),
        global_fallback_utterances=env_int("DEPRESSION_ASPECT_RETRIEVAL_GLOBAL_FALLBACK_UTTERANCES", 0),
        global_fallback_max_dialogue_tokens=env_int("DEPRESSION_ASPECT_RETRIEVAL_GLOBAL_FALLBACK_MAX_DIALOGUE_TOKENS", 256),
        max_profile_chars=env_int("DEPRESSION_ASPECT_RETRIEVAL_MAX_PROFILE_CHARS", 24000),
    )


def _chat_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    except Exception:
        pass
    rendered = [f"{message['role'].title()}: {message['content']}" for message in messages]
    rendered.append("Assistant:")
    return "\n\n".join(rendered)


def _normalize_embeddings(tensor: Any) -> Any:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class LoadedLlmHiddenBackend:
    name = "local_hidden_l2"

    def __init__(
        self,
        *,
        tokenizer: Any,
        llm: Any,
        model_name: str,
        batch_size: int = 16,
        max_embedding_length: int = 256,
        max_profile_new_tokens: int = 192,
        max_query_new_tokens: int = 96,
    ) -> None:
        self.tokenizer = tokenizer
        self.llm = llm
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_embedding_length = int(max_embedding_length)
        self.max_profile_new_tokens = int(max_profile_new_tokens)
        self.max_query_new_tokens = int(max_query_new_tokens)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _base_model_context(self):
        disable_adapter = getattr(self.llm, "disable_adapter", None)
        if callable(disable_adapter):
            return disable_adapter()
        if hasattr(self.llm, "peft_config"):
            raise RuntimeError(
                "Shared retrieval requires PEFT disable_adapter() to preserve "
                "base-Qwen hidden-state retrieval."
            )
        return nullcontext()

    @property
    def device(self) -> Any:
        return next(self.llm.parameters()).device

    def _generate(self, user_content: str, max_new_tokens: int) -> str:
        import torch

        messages = [
            {"role": "system", "content": CLINICAL_ASSISTANT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        text = _chat_text(self.tokenizer, messages)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_len = int(inputs["input_ids"].shape[1])
        with self._base_model_context():
            with torch.inference_mode():
                output = self.llm.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        generated_ids = output[0, input_len:]
        return " ".join(
            self.tokenizer.decode(generated_ids, skip_special_tokens=True).split()
        )

    def generate_profile(self, dialogue: str) -> str:
        return self._generate(
            build_depression_profile_prompt(dialogue),
            self.max_profile_new_tokens,
        )

    def generate_query(
        self,
        *,
        profile: str,
        aspect: str,
        aspect_description: str,
        query_prompt_version: str = QUERY_PROMPT_VERSION,
    ) -> str:
        prompt = build_depression_aspect_query_prompt(
            profile=profile,
            aspect=aspect,
            aspect_description=aspect_description,
            query_prompt_version=query_prompt_version,
        )
        return _coerce_retrieval_query(
            self._generate(prompt, self.max_query_new_tokens),
            aspect=aspect,
            aspect_description=aspect_description,
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        rows: list[Any] = []
        with self._base_model_context():
            with torch.inference_mode():
                for start in range(0, len(texts), self.batch_size):
                    batch = texts[start : start + self.batch_size]
                    encoded = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_embedding_length,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(self.device) for key, value in encoded.items()}
                    outputs = self.llm(
                        **encoded,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden = outputs.hidden_states[-1].float()
                    mask = encoded["attention_mask"].to(hidden.device).float().unsqueeze(-1)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                    rows.append(_normalize_embeddings(pooled).cpu())
        return torch.cat(rows, dim=0).numpy().astype(np.float32, copy=False)

    def token_count(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
            return_tensors=None,
        )
        ids = encoded["input_ids"]
        return len(ids[0] if ids and isinstance(ids[0], list) else ids)


class LocalHiddenBackend:
    name = "local_hidden_l2"

    def __init__(
        self,
        model_name: str,
        *,
        device_map: str | None = None,
        batch_size: int = 16,
        max_embedding_length: int = 256,
        max_profile_new_tokens: int = 192,
        max_query_new_tokens: int = 64,
    ) -> None:
        import torch

        try:
            from mllm_dr.model.mllm_dr import _install_transformers_torch_checkpoint_shim

            _install_transformers_torch_checkpoint_shim()
        except Exception:
            pass
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_embedding_length = int(max_embedding_length)
        self.max_profile_new_tokens = int(max_profile_new_tokens)
        self.max_query_new_tokens = int(max_query_new_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16
        if device_map:
            load_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if not device_map:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        else:
            self.device = next(self.model.parameters()).device
        self.model.eval()

    def _generate(self, user_content: str, max_new_tokens: int) -> str:
        import torch

        messages = [
            {"role": "system", "content": CLINICAL_ASSISTANT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        text = _chat_text(self.tokenizer, messages)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_len = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output[0, input_len:]
        return " ".join(
            self.tokenizer.decode(generated_ids, skip_special_tokens=True).split()
        )

    def generate_profile(self, dialogue: str) -> str:
        return self._generate(
            build_depression_profile_prompt(dialogue),
            self.max_profile_new_tokens,
        )

    def generate_query(
        self,
        *,
        profile: str,
        aspect: str,
        aspect_description: str,
        query_prompt_version: str = QUERY_PROMPT_VERSION,
    ) -> str:
        prompt = build_depression_aspect_query_prompt(
            profile=profile,
            aspect=aspect,
            aspect_description=aspect_description,
            query_prompt_version=query_prompt_version,
        )
        return _coerce_retrieval_query(
            self._generate(prompt, self.max_query_new_tokens),
            aspect=aspect,
            aspect_description=aspect_description,
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        rows: list[Any] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_embedding_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = outputs.hidden_states[-1].float()
                mask = encoded["attention_mask"].to(hidden.device).float().unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                rows.append(_normalize_embeddings(pooled).cpu())
        return torch.cat(rows, dim=0).numpy().astype(np.float32, copy=False)

    def token_count(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
            return_tensors=None,
        )
        ids = encoded["input_ids"]
        return len(ids[0] if ids and isinstance(ids[0], list) else ids)


class LexicalBackend:
    name = "lexical_l2"
    model_name = "lexical"

    def __init__(self, dim: int = 256) -> None:
        self.dim = int(dim)

    def generate_profile(self, dialogue: str) -> str:
        words = " ".join(dialogue.split())
        return words[:512] if words else "No transcript content was available."

    def generate_query(
        self,
        *,
        profile: str,
        aspect: str,
        aspect_description: str,
        query_prompt_version: str = QUERY_PROMPT_VERSION,
    ) -> str:
        query = _default_evidence_query(
            aspect=aspect,
            aspect_description=aspect_description,
        )
        if query_prompt_version == "clinical_graph_v1":
            query = f"{query} {DEPRESSION_CLINICAL_GRAPH_HEURISTIC_SUFFIX}"
        elif query_prompt_version == "adaptive_reflect_v1":
            query = f"{query} {DEPRESSION_ADAPTIVE_REFLECT_HEURISTIC_SUFFIX}"
        return append_depression_optional_profile_context(query, profile)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9']+", text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8)
                vectors[row, int.from_bytes(digest.digest(), "little") % self.dim] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def token_count(self, text: str) -> int:
        return len(re.findall(r"[a-z0-9']+|[^\s]", text.lower()))


def _candidate_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _default_evidence_query(*, aspect: str, aspect_description: str) -> str:
    return build_default_depression_evidence_query(
        aspect=aspect,
        aspect_description=aspect_description,
    )


def _coerce_retrieval_query(
    generated_query: str,
    *,
    aspect: str,
    aspect_description: str,
) -> str:
    query = " ".join(str(generated_query or "").split()).strip(" \"'`")
    query = re.sub(
        r"^(?:evidence\s+retrieval\s+query|retrieval\s+query|query)\s*:\s*",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()
    fallback = _default_evidence_query(
        aspect=aspect,
        aspect_description=aspect_description,
    )
    if not query:
        return fallback

    words = _candidate_words(query)
    first_word = words[0] if words else ""
    question_like = (
        query.endswith("?")
        or first_word in QUESTION_OPENERS
        or first_word in DIRECTIVE_OPENERS
    )
    if question_like:
        return fallback

    if not re.match(r"^(?:retrieve|find|select|rank)\b", query, flags=re.IGNORECASE):
        return build_depression_prioritized_evidence_query(
            aspect=aspect,
            aspect_description=aspect_description,
            generated_query=query,
        )
    return query


def _has_any(words: list[str], vocabulary: set[str]) -> bool:
    return any(word in vocabulary for word in words)


def _has_first_person_evidence(words: list[str]) -> bool:
    return _has_any(words, FIRST_PERSON_WORDS)


def _has_second_person(words: list[str]) -> bool:
    return _has_any(words, SECOND_PERSON_WORDS)


def _has_clinical_evidence(text: str) -> bool:
    return CLINICAL_EVIDENCE_RE.search(text) is not None


def _clinical_evidence_count(text: str) -> int:
    return len(CLINICAL_EVIDENCE_RE.findall(text))


def _first_person_count(words: list[str]) -> int:
    return sum(1 for word in words if word in FIRST_PERSON_WORDS)


def aspect_keyword_hits(aspect_index: int, text: str) -> tuple[str, ...]:
    hits: list[str] = []
    for label, pattern in ASPECT_KEYWORD_PATTERNS.get(int(aspect_index), ()):
        if pattern.search(text):
            hits.append(label)
    return tuple(hits)


def _has_embedded_answer_evidence(text: str, words: list[str]) -> bool:
    if _has_first_person_evidence(words):
        return True
    if _has_clinical_evidence(text) and _has_any(words, ANSWER_CUE_WORDS):
        return True
    return bool(
        _has_clinical_evidence(text)
        and re.search(r"\b(?:it|that|this)\s+(?:is|was|has|gets|feels)\b", text, re.I)
    )


def _is_strict_setup_fragment(text: str, words: list[str]) -> bool:
    if STRICT_SETUP_FRAGMENT_RE.search(text):
        return True
    if not _has_embedded_answer_evidence(text, words):
        if len(words) <= 5 and any(
            token in words
            for token in ("button", "frozen", "avatar", "ellie", "virtual", "recording")
        ):
            return True
        if words[:6] in (
            ["do", "i", "need", "to", "see", "something"],
            ["can", "you", "tell", "me", "about"],
        ):
            return True
    return False


def _looks_like_interviewer_prompt(text: str, words: list[str]) -> bool:
    if not words:
        return False
    starts_question = words[0] in QUESTION_OPENERS
    ends_question = "?" in text
    starts_directive = words[0] in DIRECTIVE_OPENERS and (
        "me" in words[:4] or _has_second_person(words[:8])
    )
    if not (starts_question or ends_question or starts_directive):
        return False
    if not _has_second_person(words) and not starts_directive:
        return False
    return not _has_embedded_answer_evidence(text, words)


def _candidate_drop_reason(
    utterance: TranscriptUtterance,
    *,
    min_words: int,
    mode: str = "participant_evidence",
) -> str | None:
    strict = mode == "participant_evidence_v2"
    text = " ".join(utterance.text.split())
    words = _candidate_words(text)
    if not words:
        return "empty"
    if strict and _is_strict_setup_fragment(text, words):
        return "setup_fragment"
    if SESSION_SETUP_RE.search(text) and not _has_embedded_answer_evidence(text, words):
        return "interview_setup"
    if _looks_like_interviewer_prompt(text, words):
        return "interviewer_prompt"
    if strict and words[0] in QUESTION_OPENERS and not _has_embedded_answer_evidence(
        text, words
    ):
        return "interviewer_prompt"
    if strict and len(words) < max(5, min_words) and not _has_embedded_answer_evidence(
        text, words
    ):
        return "low_information"
    if len(words) < min_words and not (
        _has_first_person_evidence(words) or _has_clinical_evidence(text)
    ):
        return "low_information"
    return None


def filter_retrieval_candidates(
    utterances: list[TranscriptUtterance],
    *,
    mode: str = "participant_evidence",
    min_words: int = 3,
    min_candidate_utterances: int = 8,
) -> CandidateFilterResult:
    mode = (mode or "participant_evidence").strip().lower()
    min_words = max(1, int(min_words))
    min_candidate_utterances = max(1, int(min_candidate_utterances))
    if mode in {"none", "off", "disabled"}:
        return CandidateFilterResult(
            utterances=list(utterances),
            summary={
                "mode": "none",
                "raw_utterances": len(utterances),
                "candidate_utterances": len(utterances),
                "dropped_utterances": 0,
                "drop_reasons": {},
                "min_words": min_words,
                "min_candidate_utterances": min_candidate_utterances,
                "relaxed_low_information": False,
            },
        )
    if mode not in {"participant_evidence", "participant_evidence_v2"}:
        raise ValueError(
            "Unsupported candidate filter. Use participant_evidence, "
            "participant_evidence_v2, or none."
        )

    strict_kept: list[TranscriptUtterance] = []
    low_information: list[TranscriptUtterance] = []
    reasons: Counter[str] = Counter()

    for utterance in utterances:
        reason = _candidate_drop_reason(utterance, min_words=min_words, mode=mode)
        if reason is None:
            strict_kept.append(utterance)
        elif reason == "low_information":
            low_information.append(utterance)
        else:
            reasons[reason] += 1

    relaxed_low_information = False
    if len(strict_kept) >= min_candidate_utterances:
        candidates = strict_kept
        reasons["low_information"] += len(low_information)
    else:
        candidates = sorted(
            [*strict_kept, *low_information],
            key=lambda item: item.index,
        )
        relaxed_low_information = bool(low_information)

    if not candidates and utterances:
        candidates = list(utterances)
        relaxed_low_information = True
        reasons["fallback_all_filtered"] += len(utterances)

    return CandidateFilterResult(
        utterances=candidates,
        summary={
            "mode": mode,
            "raw_utterances": len(utterances),
            "candidate_utterances": len(candidates),
            "dropped_utterances": max(0, len(utterances) - len(candidates)),
            "drop_reasons": dict(sorted(reasons.items())),
            "min_words": min_words,
            "min_candidate_utterances": min_candidate_utterances,
            "relaxed_low_information": relaxed_low_information,
        },
    )


def select_utterances_by_budget(
    query_embedding: np.ndarray,
    utterance_embeddings: np.ndarray,
    utterances: list[TranscriptUtterance],
    utterance_token_counts: list[int],
    min_utterances: int,
    max_utterances: int,
    max_dialogue_tokens: int,
    min_score: float | None = None,
    relative_score_margin: float | None = None,
    score_adjustments: list[float] | None = None,
    min_score_applies_to_min: bool = False,
    aspect_keyword_hits_by_index: dict[int, tuple[str, ...]] | None = None,
) -> list[RetrievedUtterance]:
    if not utterances or utterance_embeddings.size == 0:
        return []
    if len(utterance_token_counts) != len(utterances):
        raise ValueError("utterance_token_counts length must match utterances")
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    embeddings = np.asarray(utterance_embeddings, dtype=np.float32)
    raw_scores = -np.linalg.norm(embeddings - query, axis=1)
    if score_adjustments is None:
        adjustments = np.zeros(len(utterances), dtype=np.float32)
    else:
        if len(score_adjustments) != len(utterances):
            raise ValueError("score_adjustments length must match utterances")
        adjustments = np.asarray(score_adjustments, dtype=np.float32)
    scores = raw_scores + adjustments
    ranked = sorted(
        range(len(utterances)),
        key=lambda idx: (-float(scores[idx]), utterances[idx].index),
    )
    min_utterances = max(1, int(min_utterances))
    max_utterances = max(min_utterances, int(max_utterances))
    max_dialogue_tokens = max(1, int(max_dialogue_tokens))
    best_score = float(scores[ranked[0]]) if ranked else float("-inf")
    min_score_value = None if min_score is None else float(min_score)
    relative_margin_value = (
        None
        if relative_score_margin is None
        else max(0.0, float(relative_score_margin))
    )

    selected_indices: list[int] = []
    selected_tokens = 0
    for idx in ranked:
        score = float(scores[idx])
        raw_score = float(raw_scores[idx])
        tokens = max(1, int(utterance_token_counts[idx]))
        if len(selected_indices) >= max_utterances:
            break
        threshold_active = min_score_applies_to_min or len(selected_indices) >= min_utterances
        if threshold_active:
            if min_score_value is not None and score < min_score_value:
                continue
            if (
                relative_margin_value is not None
                and best_score - score > relative_margin_value
            ):
                continue
        if (
            len(selected_indices) >= min_utterances
            and selected_tokens + tokens > max_dialogue_tokens
        ):
            continue
        selected_indices.append(idx)
        selected_tokens += tokens
        if (
            len(selected_indices) >= min_utterances
            and selected_tokens >= max_dialogue_tokens
        ):
            break
    return sorted(
        [
            RetrievedUtterance(
                utterance=utterances[idx],
                score=float(scores[idx]),
                raw_score=float(raw_score),
                evidence_rerank_bonus=float(adjustments[idx]),
                aspect_keyword_hits=(aspect_keyword_hits_by_index or {}).get(
                    utterances[idx].index,
                    (),
                ),
            )
            for idx, raw_score in (
                (selected_idx, raw_scores[selected_idx])
                for selected_idx in selected_indices
            )
        ],
        key=lambda item: item.utterance.index,
    )


def expand_retrieved_context(
    retrieved: list[RetrievedUtterance],
    *,
    raw_utterances: list[TranscriptUtterance],
    backend: RetrievalBackend,
    context_window: int = 0,
    max_dialogue_tokens: int = 1536,
) -> list[RetrievedUtterance]:
    context_window = max(0, int(context_window))
    if context_window <= 0 or not retrieved:
        return retrieved
    by_index = {utterance.index: utterance for utterance in raw_utterances}
    raw_order = [utterance.index for utterance in raw_utterances]
    raw_position = {index: pos for pos, index in enumerate(raw_order)}
    selected: dict[int, RetrievedUtterance] = {
        item.utterance.index: item for item in retrieved
    }
    selected_tokens = sum(
        backend.token_count(item.utterance.text) for item in selected.values()
    )
    max_dialogue_tokens = max(1, int(max_dialogue_tokens))
    for item in retrieved:
        pos = raw_position.get(item.utterance.index)
        if pos is None:
            continue
        for offset in range(-context_window, context_window + 1):
            if offset == 0:
                continue
            ctx_pos = pos + offset
            if ctx_pos < 0 or ctx_pos >= len(raw_order):
                continue
            ctx_index = raw_order[ctx_pos]
            if ctx_index in selected:
                continue
            utterance = by_index[ctx_index]
            tokens = max(1, backend.token_count(utterance.text))
            if selected_tokens + tokens > max_dialogue_tokens:
                continue
            selected[ctx_index] = RetrievedUtterance(
                utterance=utterance,
                score=item.score,
                raw_score=item.raw_score,
                evidence_rerank_bonus=item.evidence_rerank_bonus,
            )
            selected_tokens += tokens
    return sorted(selected.values(), key=lambda row: row.utterance.index)


def evidence_rerank_adjustments(
    utterances: list[TranscriptUtterance],
    *,
    aspect_index: int,
    mode: str,
) -> tuple[list[float], dict[int, dict[str, Any]]]:
    mode = (mode or "none").strip().lower()
    metadata: dict[int, dict[str, Any]] = {}
    if mode in {"none", "off", "disabled"}:
        return [0.0 for _ in utterances], metadata
    if mode != "clinical_v1":
        raise ValueError("Unsupported evidence rerank mode. Use none or clinical_v1.")

    adjustments: list[float] = []
    for utterance in utterances:
        text = " ".join(utterance.text.split())
        words = _candidate_words(text)
        hits = aspect_keyword_hits(aspect_index, text)
        clinical_count = _clinical_evidence_count(text)
        first_person_count = _first_person_count(words)
        setup_like = _is_strict_setup_fragment(text, words) or bool(
            SESSION_SETUP_RE.search(text)
        )
        positive_counter = bool(POSITIVE_COUNTER_RE.search(text))

        bonus = 0.0
        if first_person_count:
            bonus += 0.06
        if clinical_count:
            bonus += min(0.10, 0.04 + 0.02 * clinical_count)
        if hits:
            bonus += 0.10 + min(0.06, 0.02 * (len(hits) - 1))
        if positive_counter:
            bonus -= 0.04
        if setup_like:
            bonus -= 0.14

        adjustments.append(float(bonus))
        metadata[utterance.index] = {
            "aspect_keyword_hits": list(hits),
            "clinical_evidence_count": int(clinical_count),
            "first_person_count": int(first_person_count),
            "setup_like": bool(setup_like),
            "positive_counter": bool(positive_counter),
            "evidence_rerank_bonus": float(bonus),
        }
    return adjustments, metadata


def dialogue_from_retrieved(retrieved: list[RetrievedUtterance]) -> str:
    return build_dialogue(
        [
            (item.utterance.start, item.utterance.end, item.utterance.text)
            for item in retrieved
        ],
        utterance_indices=[item.utterance.index for item in retrieved],
    )


def select_global_fallback_utterances(
    utterances: list[TranscriptUtterance],
    *,
    selected_indices: set[int],
    backend: RetrievalBackend,
    max_utterances: int,
    max_dialogue_tokens: int,
    mode: str = "clinical_v1",
    aspect_index: int | None = None,
    selected_aspect_keyword_hits: set[str] | None = None,
) -> list[TranscriptUtterance]:
    mode = (mode or "clinical_v1").strip().lower()
    if mode not in {"clinical_v1", "clinical_v2"}:
        raise ValueError(
            "Unsupported global fallback mode. Use clinical_v1 or clinical_v2."
        )
    max_utterances = max(0, int(max_utterances))
    max_dialogue_tokens = max(1, int(max_dialogue_tokens))
    if max_utterances <= 0:
        return []
    candidates = [
        utterance
        for utterance in utterances
        if utterance.index not in selected_indices
        and _has_clinical_evidence(utterance.text)
    ]
    if mode == "clinical_v1":
        ranked = sorted(
            candidates,
            key=lambda utterance: (
                -len(set(_candidate_words(utterance.text)) & FIRST_PERSON_WORDS),
                -len(CLINICAL_EVIDENCE_RE.findall(utterance.text)),
                utterance.index,
            ),
        )
    else:
        selected_aspect_keyword_hits = selected_aspect_keyword_hits or set()

        def fallback_rank(utterance: TranscriptUtterance) -> tuple[float, int]:
            text = " ".join(utterance.text.split())
            words = _candidate_words(text)
            hits = (
                set(aspect_keyword_hits(int(aspect_index), text))
                if aspect_index is not None
                else set()
            )
            missing_hit = bool(hits and not hits <= selected_aspect_keyword_hits)
            setup_like = _is_strict_setup_fragment(text, words) or bool(
                SESSION_SETUP_RE.search(text)
            )
            score = 0.0
            score += 3.0 if missing_hit else 0.0
            score += min(2.0, float(_clinical_evidence_count(text)))
            score += min(1.0, float(_first_person_count(words)))
            score -= 2.0 if setup_like else 0.0
            score -= 0.5 if POSITIVE_COUNTER_RE.search(text) else 0.0
            return (-score, utterance.index)

        ranked = sorted(candidates, key=fallback_rank)
    selected: list[TranscriptUtterance] = []
    selected_tokens = 0
    for utterance in ranked:
        tokens = max(1, backend.token_count(utterance.text))
        if len(selected) >= max_utterances:
            break
        if selected and selected_tokens + tokens > max_dialogue_tokens:
            continue
        selected.append(utterance)
        selected_tokens += tokens
    return sorted(selected, key=lambda utterance: utterance.index)


def dialogue_with_global_fallback(
    retrieved: list[RetrievedUtterance],
    fallback: list[TranscriptUtterance],
) -> str:
    dialogue = dialogue_from_retrieved(retrieved)
    if not fallback:
        return dialogue
    fallback_dialogue = build_dialogue(
        [(row.start, row.end, row.text) for row in fallback],
        utterance_indices=[row.index for row in fallback],
    )
    if not dialogue:
        return f"Global clinical fallback:\n{fallback_dialogue}"
    return f"{dialogue}\n\nGlobal clinical fallback:\n{fallback_dialogue}"


def transcript_user_utterances(transcript: dict[str, Any]) -> list[TranscriptUtterance]:
    events = transcript.get("events") if isinstance(transcript, dict) else None
    if not isinstance(events, list):
        return []
    utterances: list[TranscriptUtterance] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if str(event.get("speaker") or "").strip().lower() != "user":
            continue
        text = " ".join(str(event.get("text") or "").split())
        if not text:
            continue
        index = len(utterances) + 1
        start = _event_seconds(event, "start_time", "start_seconds", "audio_start_seconds")
        end = _event_seconds(event, "end_time", "end_seconds", "audio_end_seconds")
        if end <= start:
            timestamp = _event_seconds(event, "timestamp_seconds")
            start = timestamp if timestamp > 0 else float(index - 1)
            end = start + 0.01
        utterances.append(TranscriptUtterance(index=index, start=start, end=end, text=text))
    return utterances


def _event_seconds(event: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = event.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return max(0.0, number)
    return 0.0


def build_aspect_retrieval_records(
    *,
    utterances: list[TranscriptUtterance],
    backend: RetrievalBackend,
    config: AspectRetrievalConfig | None = None,
    split: str = "realtime",
    participant_id: int = 0,
) -> AspectRetrievalOutput:
    config = config or config_from_env()
    filter_result = filter_retrieval_candidates(
        utterances,
        mode=config.candidate_filter,
        min_words=config.candidate_filter_min_words,
        min_candidate_utterances=config.min_candidate_utterances,
    )
    candidates = filter_result.utterances
    candidate_dialogue = build_dialogue(
        [(row.start, row.end, row.text) for row in candidates],
        utterance_indices=[row.index for row in candidates],
    )
    profile_source = (
        candidate_dialogue[: int(config.max_profile_chars)]
        if config.max_profile_chars
        else candidate_dialogue
    )
    profile = backend.generate_profile(profile_source)
    utterance_texts = [utterance.text for utterance in candidates]
    token_counts = [backend.token_count(text) for text in utterance_texts]
    token_counts_by_index = {
        utterance.index: token_count
        for utterance, token_count in zip(candidates, token_counts, strict=False)
    }
    utterance_embeddings = backend.embed_texts(utterance_texts)

    records: list[dict[str, Any]] = []
    for aspect_index, ((aspect_key, aspect), aspect_description) in enumerate(
        zip(PHQ8_ASPECTS, PHQ8_CLINICAL_DESCRIPTIONS, strict=True)
    ):
        query = backend.generate_query(
            profile=profile,
            aspect=aspect,
            aspect_description=aspect_description,
            query_prompt_version=config.query_prompt_version,
        )
        query_embedding = backend.embed_texts([query])[0]
        score_adjustments, rerank_metadata = evidence_rerank_adjustments(
            candidates,
            aspect_index=aspect_index,
            mode=config.evidence_rerank,
        )
        retrieved = select_utterances_by_budget(
            query_embedding=query_embedding,
            utterance_embeddings=utterance_embeddings,
            utterances=candidates,
            utterance_token_counts=token_counts,
            min_utterances=config.min_utterances,
            max_utterances=config.max_utterances,
            max_dialogue_tokens=config.max_dialogue_tokens,
            min_score=config.min_score,
            relative_score_margin=config.relative_score_margin,
            score_adjustments=score_adjustments,
            min_score_applies_to_min=config.min_score_applies_to_min,
            aspect_keyword_hits_by_index={
                index: tuple(metadata.get("aspect_keyword_hits", ()))
                for index, metadata in rerank_metadata.items()
            },
        )
        retrieved = expand_retrieved_context(
            retrieved,
            raw_utterances=utterances,
            backend=backend,
            context_window=config.context_window,
            max_dialogue_tokens=config.max_dialogue_tokens,
        )
        selected_token_total = sum(
            token_counts_by_index.get(
                item.utterance.index,
                backend.token_count(item.utterance.text),
            )
            for item in retrieved
        )
        selected_aspect_keyword_hits = {
            hit for item in retrieved for hit in item.aspect_keyword_hits
        }
        adaptive_reflect_triggered = bool(
            config.adaptive_reflect and not selected_aspect_keyword_hits
        )
        fallback_utterance_limit = int(config.global_fallback_utterances)
        if adaptive_reflect_triggered:
            fallback_utterance_limit = max(fallback_utterance_limit, config.min_utterances)
        fallback = select_global_fallback_utterances(
            candidates,
            selected_indices={item.utterance.index for item in retrieved},
            backend=backend,
            max_utterances=fallback_utterance_limit,
            max_dialogue_tokens=config.global_fallback_max_dialogue_tokens,
            mode=config.global_fallback_mode,
            aspect_index=aspect_index,
            selected_aspect_keyword_hits=selected_aspect_keyword_hits,
        )
        fallback_token_total = sum(backend.token_count(item.text) for item in fallback)
        records.append(
            {
                "split": split,
                "participant_id": int(participant_id),
                "aspect_index": int(aspect_index),
                "aspect_key": aspect_key,
                "aspect": aspect,
                "aspect_description": aspect_description,
                "profile": profile,
                "query": query,
                "utterance_indices": [item.utterance.index for item in retrieved],
                "scores": [item.score for item in retrieved],
                "utterances": [
                    {
                        "index": item.utterance.index,
                        "start": item.utterance.start,
                        "end": item.utterance.end,
                        "text": item.utterance.text,
                        "score": item.score,
                        "raw_score": item.raw_score,
                        "evidence_rerank_bonus": item.evidence_rerank_bonus,
                        "aspect_keyword_hits": list(item.aspect_keyword_hits),
                    }
                    for item in retrieved
                ],
                "global_fallback_utterance_indices": [item.index for item in fallback],
                "global_fallback_utterance_rows": [
                    {
                        "index": item.index,
                        "start": item.start,
                        "end": item.end,
                        "text": item.text,
                        "aspect_keyword_hits": list(aspect_keyword_hits(aspect_index, item.text)),
                        "clinical_evidence_count": _clinical_evidence_count(item.text),
                    }
                    for item in fallback
                ],
                "dialogue": dialogue_with_global_fallback(retrieved, fallback),
                "retrieval_backend": backend.name,
                "retrieval_model": backend.model_name,
                "query_prompt_version": config.query_prompt_version,
                "context_window": int(config.context_window),
                "adaptive_reflect": bool(config.adaptive_reflect),
                "adaptive_reflect_triggered": adaptive_reflect_triggered,
                "candidate_filter": filter_result.summary["mode"],
                "candidate_filter_summary": filter_result.summary,
                "evidence_rerank": config.evidence_rerank,
                "min_score_applies_to_min": bool(config.min_score_applies_to_min),
                "selected_aspect_keyword_hits": sorted(selected_aspect_keyword_hits),
                "raw_utterance_count": int(len(utterances)),
                "candidate_utterance_count": int(len(candidates)),
                "min_utterances": int(config.min_utterances),
                "max_utterances": int(config.max_utterances),
                "max_dialogue_tokens": int(config.max_dialogue_tokens),
                "min_score": float(config.min_score),
                "relative_score_margin": float(config.relative_score_margin),
                "selected_dialogue_tokens": int(selected_token_total),
                "global_fallback_mode": config.global_fallback_mode,
                "global_fallback_utterances": int(config.global_fallback_utterances),
                "global_fallback_max_dialogue_tokens": int(
                    config.global_fallback_max_dialogue_tokens
                ),
                "global_fallback_dialogue_tokens": int(fallback_token_total),
            }
        )
    return AspectRetrievalOutput(
        records=records,
        raw_utterance_count=len(utterances),
        candidate_filter_summary=filter_result.summary,
        backend_name=backend.name,
        backend_model_name=backend.model_name,
    )
