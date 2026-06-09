from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from mllm_dr.training.text import extract_rationale_text


RATIONALE_PREFIX_RE = re.compile(r"^\s*Evaluation\s+Reason\s*:\s*", re.IGNORECASE)
SCORE_LEAKAGE_RE = re.compile(
    r"\b(?:"
    r"score(?:d|s)?|label(?:ed|s)?|PHQ(?:-?8)?|private\s+target|"
    r"private\s+supervision|evaluation\s+result|overall\s+score|"
    r"not\s+at\s+all|several\s+days|more\s+than\s+half\s+the\s+days|"
    r"nearly\s+every\s+day"
    r")\b",
    re.IGNORECASE,
)
NO_EVIDENCE_RE = re.compile(
    r"\b(?:"
    r"insufficient\s+(?:clinical\s+|direct\s+|specific\s+)?evidence|"
    r"no\s+(?:clear\s+|direct\s+|specific\s+|significant\s+)?(?:evidence|"
    r"indication|reference|mention)|"
    r"does\s+not\s+(?:mention|show|describe|indicate)|"
    r"did\s+not\s+(?:mention|show|describe|indicate)|"
    r"not\s+(?:show|mention|describe|indicate)|"
    r"without\s+(?:indicating|describing|mentioning)"
    r")\b",
    re.IGNORECASE,
)
REPEATED_SPAN_RE = re.compile(r"(.{20,}?)\1", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
OBSERVABILITY_LIMITED_ASPECT_RE = re.compile(
    r"\b(?:appetite|sleep|concentrat|psychomotor|moving|speaking)\b",
    re.IGNORECASE,
)
MULTIMODAL_PLACEHOLDER_RE = re.compile(
    r"\b(?:"
    r"multimodal\s+feature\s+tokens?\s+(?:are|were)?\s*(?:empty\s+)?placeholders?|"
    r"feature\s+tokens?\s+(?:are|were)?\s*empty|"
    r"empty\s+placeholders?\s+(?:that\s+)?(?:provide|providing|containing)|"
    r"without\s+accompanying\s+multimodal\s+audio-?visual\s+data|"
    r"without\s+access\s+to\s+(?:the\s+)?(?:specific\s+)?multimodal|"
    r"no\s+observable\s+(?:audio\s+or\s+visual|visual\s+or\s+audio|audio-?visual)\s+cues|"
    r"do\s+not\s+(?:present|provide|contain)\s+any\s+observable\s+(?:audio\s+or\s+visual|audio-?visual)\s+cues"
    r")\b",
    re.IGNORECASE,
)


def normalize_rationale(text: str | None) -> str:
    rationale = " ".join(extract_rationale_text(text).split())
    rationale = RATIONALE_PREFIX_RE.sub("", rationale).strip()
    return f"Evaluation Reason: {rationale}"


def rationale_body(text: str | None) -> str:
    return RATIONALE_PREFIX_RE.sub("", normalize_rationale(text)).strip()


def _tokens(text: str | None) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "")}


def token_jaccard(a: str | None, b: str | None) -> float:
    left = _tokens(a)
    right = _tokens(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def quality_flags(
    text: str | None,
    *,
    label: int | float | None = None,
    aspect: str | None = None,
    min_words: int = 8,
    max_words: int = 110,
) -> dict[str, Any]:
    original = " ".join((text or "").split())
    normalized = normalize_rationale(text)
    body = rationale_body(normalized)
    words = TOKEN_RE.findall(body)
    word_count = len(words)
    label_value = None if label is None else int(float(label))
    no_evidence = bool(NO_EVIDENCE_RE.search(body))
    multimodal_placeholder = bool(MULTIMODAL_PLACEHOLDER_RE.search(body))
    score_leakage = bool(SCORE_LEAKAGE_RE.search(original) or SCORE_LEAKAGE_RE.search(body))
    repeated = bool(REPEATED_SPAN_RE.search(body))
    flags: dict[str, Any] = {
        "word_count": word_count,
        "empty": word_count == 0,
        "too_short": word_count < min_words,
        "too_long": word_count > max_words,
        "score_leakage": score_leakage,
        "repetition": repeated,
        "no_evidence": no_evidence,
        "multimodal_placeholder": multimodal_placeholder,
        "high_label_no_evidence": bool(
            label_value is not None and label_value >= 2 and no_evidence
        ),
        "high_label_multimodal_placeholder": bool(
            label_value is not None and label_value >= 2 and multimodal_placeholder
        ),
        "format_failure": not bool(
            re.match(r"^\s*Evaluation\s+reason\s*:", original, re.IGNORECASE)
        ),
        "observability_limited": bool(
            no_evidence
            and aspect
            and OBSERVABILITY_LIMITED_ASPECT_RE.search(aspect)
        ),
    }
    flags["hard_filter_pass"] = not any(
        bool(flags[name])
        for name in (
            "empty",
            "too_short",
            "too_long",
            "score_leakage",
            "repetition",
            "format_failure",
        )
    )
    return flags


def candidate_quality_score(
    candidate: dict[str, Any],
    *,
    min_words: int = 8,
    max_words: int = 110,
    no_evidence_penalty: float | None = None,
    label_consistency: bool = False,
) -> tuple[float, dict[str, Any]]:
    rationale = str(candidate.get("rationale") or candidate.get("generation") or "")
    flags = quality_flags(
        rationale,
        label=candidate.get("label"),
        aspect=candidate.get("aspect"),
        min_words=min_words,
        max_words=max_words,
    )
    score = 0.0
    if flags.get("hard_filter_pass"):
        score += 10.0
    for name in ("empty", "too_short", "too_long", "score_leakage", "repetition"):
        if flags.get(name):
            score -= 100.0
    if flags.get("format_failure"):
        score -= 25.0
    penalty = 30.0 if no_evidence_penalty is None else float(no_evidence_penalty)
    if flags.get("high_label_no_evidence"):
        score -= penalty
    if flags.get("high_label_multimodal_placeholder"):
        score -= 25.0
    elif flags.get("no_evidence"):
        score -= 3.0
    if flags.get("multimodal_placeholder"):
        score -= 4.0
    if label_consistency and (
        flags.get("high_label_no_evidence")
        or flags.get("high_label_multimodal_placeholder")
    ):
        score -= 20.0
    word_count = float(flags.get("word_count", 0))
    score -= abs(word_count - 45.0) / 100.0
    try:
        score -= float(candidate.get("candidate_index", 0)) / 1000.0
    except (TypeError, ValueError):
        pass
    return score, dict(flags)


def best_heuristic_candidate(
    candidates: Iterable[dict[str, Any]],
    *,
    min_words: int = 8,
    max_words: int = 110,
    no_evidence_penalty: float | None = None,
    label_consistency: bool = False,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = float("-inf")
    for candidate in candidates:
        score, flags = candidate_quality_score(
            candidate,
            min_words=min_words,
            max_words=max_words,
            no_evidence_penalty=no_evidence_penalty,
            label_consistency=label_consistency,
        )
        if score > best_score:
            best = {**candidate, "quality_flags": flags}
            best_score = score
    return best
