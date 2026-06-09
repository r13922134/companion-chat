from __future__ import annotations

import re
from typing import Any

import torch

from mllm_dr.prompts import (
    build_lq_messages,
    build_mllm_messages,
    merge_system_into_user_messages,
)


TRACT_SCORE_PREFIX = "So the overall score is"
TRACT_SCORE_RE = re.compile(
    r"(?:"
    r"so\s+the\s+overall\s+score\s*(?:is|=|:)|"
    r"overall\s+score\s*(?:is|=|:)|"
    r"evaluation\s*result\s*[:=]|"
    r"result\s*[:=]|"
    r"score\s*(?:is|=|:)"
    r")\s*[0-3](?!\d)(?!\s*[-\u2013]\s*[0-3])\s*[\.).,:;-]?",
    re.IGNORECASE,
)
LEADING_SCORE_WORD_RE = re.compile(
    r"^\s*[0-3]\s+(?:score|result)\b[\s.\u3002]*",
    re.IGNORECASE,
)
REPEATED_SINGLE_SCORE_RE = re.compile(
    r"^\s*([0-3])(?:\s+\1)+\s*[.\u3002]?\s*$",
    re.IGNORECASE,
)


def _manual_chat(messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    text = ""
    for message in messages:
        text += f"{message['role'].capitalize()}: {message['content']}\n"
    if add_generation_prompt:
        text += "Assistant: "
    return text


def render_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        kwargs = dict(chat_template_kwargs or {})
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
    return _manual_chat(messages, add_generation_prompt)


def _tokenize(tokenizer: Any, text: str, max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )


def _tokenize_no_trunc(tokenizer: Any, text: str) -> torch.Tensor:
    return tokenizer(text, truncation=False, padding=False, return_tensors="pt")[
        "input_ids"
    ].squeeze(0)


def _tokenize_fragment(tokenizer: Any, text: str) -> torch.Tensor:
    try:
        encoded = tokenizer(
            text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
            return_tensors="pt",
        )
    except TypeError:
        encoded = tokenizer(text, truncation=False, padding=False, return_tensors="pt")
    return encoded["input_ids"].squeeze(0)


def _find_subsequence(sequence: torch.Tensor, needle: torch.Tensor) -> int | None:
    if needle.numel() == 0 or needle.numel() > sequence.numel():
        return None
    needle_len = int(needle.numel())
    for start in range(0, int(sequence.numel()) - needle_len + 1):
        if torch.equal(sequence[start : start + needle_len], needle):
            return start
    return None


def _strip_rationale_prefix(rationale: str | None) -> str:
    reason = rationale or "The participant's responses support this score."
    for marker in ("Evaluation Reason:", "Evaluation reason:"):
        if reason.startswith(marker):
            reason = reason.split(":", 1)[1].strip()
    return reason


def extract_rationale_text(text: str | None) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "The participant's responses support this score."
    for marker in ("Evaluation reason:", "Evaluation Reason:"):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
            break
    score_match = TRACT_SCORE_RE.search(text)
    if score_match:
        text = text[: score_match.start()].strip()
    text = TRACT_SCORE_RE.sub("", text).strip()
    text = LEADING_SCORE_WORD_RE.sub("", text).strip()
    if REPEATED_SINGLE_SCORE_RE.match(text):
        text = ""
    return text or "The participant's responses support this score."


def _score_token_mask(
    tokenizer: Any,
    input_ids: torch.Tensor,
    response_ids: torch.Tensor,
    response_start: int,
    score_text: str | None,
) -> torch.Tensor:
    mask = torch.zeros_like(input_ids, dtype=torch.long)
    if score_text is None:
        return mask
    candidates = (
        (f"{TRACT_SCORE_PREFIX} {score_text}", f" {score_text}"),
        (f"Evaluation result: {score_text}", f" {score_text}"),
        (f"result: {score_text}", f" {score_text}"),
        (f": {score_text}", f" {score_text}"),
        (f"is {score_text}", f" {score_text}"),
        (f" {score_text}", f" {score_text}"),
        (str(score_text), str(score_text)),
    )
    for candidate, score_fragment in candidates:
        candidate_ids = _tokenize_fragment(tokenizer, candidate)
        start = _find_subsequence(response_ids, candidate_ids)
        if start is None:
            continue
        score_ids = _tokenize_fragment(tokenizer, score_fragment)
        score_start = _find_subsequence(candidate_ids, score_ids)
        if score_start is not None:
            start += score_start + int(score_ids.numel()) - 1
            candidate_ids = score_ids[-1:]
        global_start = response_start + start
        end = min(global_start + int(candidate_ids.numel()), int(input_ids.numel()))
        mask[global_start:end] = 1
        return mask
    return mask


def _apply_response_masks(
    tokenizer: Any,
    input_ids: torch.Tensor,
    response_start: int,
    response_ids: torch.Tensor,
    score_text: str | None,
    include_score_token_mask: bool,
) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    if include_score_token_mask:
        masks["score_token_mask"] = _score_token_mask(
            tokenizer=tokenizer,
            input_ids=input_ids,
            response_ids=response_ids,
            response_start=response_start,
            score_text=score_text,
        )
    return masks


def _truncate_prompt_from_left(
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor | None,
    max_length: int,
) -> torch.Tensor:
    response_len = 0 if response_ids is None else int(response_ids.shape[0])
    prompt_budget = max(1, max_length - response_len)
    if prompt_ids.shape[0] <= prompt_budget:
        return prompt_ids
    return prompt_ids[-prompt_budget:]


def build_training_encoding(
    tokenizer: Any,
    stage: str,
    dialogue: str,
    aspect: str,
    label: int,
    rationale: str | None,
    max_length: int,
    include_score_token_mask: bool = False,
    score_after_rationale: bool = False,
    score_only_response: bool = False,
    rationale_only_response: bool = False,
    merge_system_into_user: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    if stage == "stage1":
        messages = build_lq_messages(aspect=aspect, label=label)
        reason_text = None
    else:
        reason_text = _strip_rationale_prefix(rationale)
        messages = build_mllm_messages(
            dialogue=dialogue,
            aspect=aspect,
            label=label,
            rationale=reason_text,
            score_after_rationale=score_after_rationale,
            score_only_response=score_only_response,
            rationale_only_response=rationale_only_response,
        )
    if merge_system_into_user:
        messages = merge_system_into_user_messages(messages)
    prompt_messages = messages[:-1]
    prompt_text = render_chat(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    full_text = render_chat(
        tokenizer,
        messages,
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    prompt_ids = _tokenize_no_trunc(tokenizer, prompt_text)
    full_ids = _tokenize_no_trunc(tokenizer, full_text)
    response_ids = full_ids[prompt_ids.shape[0] :]
    if response_ids.numel() == 0:
        # Conservative fallback for tokenizers whose chat templates do not
        # produce a byte-identical prompt prefix.
        assistant_text = messages[-1]["content"]
        response_ids = _tokenize_no_trunc(tokenizer, assistant_text)
    if response_ids.shape[0] >= max_length:
        response_ids = response_ids[: max_length - 1]
    prompt_ids = _truncate_prompt_from_left(prompt_ids, response_ids, max_length)
    response_start = int(prompt_ids.shape[0])
    input_ids = torch.cat([prompt_ids, response_ids], dim=0)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:response_start] = -100
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    out.update(
        _apply_response_masks(
            tokenizer=tokenizer,
            input_ids=input_ids,
            response_start=response_start,
            response_ids=response_ids,
            score_text=None if rationale_only_response else str(label),
            include_score_token_mask=include_score_token_mask,
        )
    )
    return out


def build_inference_encoding(
    tokenizer: Any,
    dialogue: str,
    aspect: str,
    max_length: int,
    score_after_rationale: bool = False,
    score_only_response: bool = False,
    rationale_only_response: bool = False,
    private_label: int | None = None,
    merge_system_into_user: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    messages = build_mllm_messages(
        dialogue=dialogue,
        aspect=aspect,
        score_after_rationale=score_after_rationale,
        score_only_response=score_only_response,
        rationale_only_response=rationale_only_response,
        private_label=private_label,
    )
    if merge_system_into_user:
        messages = merge_system_into_user_messages(messages)
    text = render_chat(
        tokenizer,
        messages,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    ids = _tokenize_no_trunc(tokenizer, text)
    if ids.shape[0] > max_length:
        ids = ids[-max_length:]
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
    }


def build_scoring_encoding(
    tokenizer: Any,
    dialogue: str,
    aspect: str,
    generation_text: str,
    max_length: int,
    include_score_token_mask: bool = False,
    score_after_rationale: bool = False,
    score_only_response: bool = False,
    score_placeholder: int = 0,
    merge_system_into_user: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    messages = build_mllm_messages(
        dialogue=dialogue,
        aspect=aspect,
        score_after_rationale=score_after_rationale,
        score_only_response=score_only_response,
    )
    if include_score_token_mask and score_after_rationale:
        rationale_text = extract_rationale_text(generation_text)
        assistant_text = (
            f"Evaluation reason: {rationale_text} "
            f"{TRACT_SCORE_PREFIX} {int(score_placeholder)}."
        )
    elif include_score_token_mask:
        assistant_text = f"Evaluation result: {int(score_placeholder)}."
    else:
        assistant_text = generation_text.strip()
    messages.append({"role": "assistant", "content": assistant_text})
    if merge_system_into_user:
        messages = merge_system_into_user_messages(messages)
    prompt_text = render_chat(
        tokenizer,
        messages[:-1],
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    full_text = render_chat(
        tokenizer,
        messages,
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    prompt_ids = _tokenize_no_trunc(tokenizer, prompt_text)
    full_ids = _tokenize_no_trunc(tokenizer, full_text)
    response_ids = full_ids[prompt_ids.shape[0] :]
    if response_ids.numel() == 0:
        response_ids = _tokenize_no_trunc(tokenizer, assistant_text)
    if response_ids.shape[0] >= max_length:
        response_ids = response_ids[: max_length - 1]
    prompt_ids = _truncate_prompt_from_left(prompt_ids, response_ids, max_length)
    response_start = int(prompt_ids.shape[0])
    input_ids = torch.cat([prompt_ids, response_ids], dim=0)
    attention_mask = torch.ones_like(input_ids)
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    out.update(
        _apply_response_masks(
            tokenizer=tokenizer,
            input_ids=input_ids,
            response_start=response_start,
            response_ids=response_ids,
            score_text=str(score_placeholder) if include_score_token_mask else None,
            include_score_token_mask=include_score_token_mask,
        )
    )
    return out


def pad_token_batch(
    encodings: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in encodings)
    batch = len(encodings)
    input_ids = torch.full((batch, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch, max_len, dtype=torch.long)
    labels = None
    if "labels" in encodings[0]:
        labels = torch.full((batch, max_len), -100, dtype=torch.long)
    extra_keys = [
        key
        for key in encodings[0]
        if key not in {"input_ids", "attention_mask", "labels"}
    ]
    extras = {
        key: torch.zeros(batch, max_len, dtype=encodings[0][key].dtype)
        for key in extra_keys
    }
    for i, item in enumerate(encodings):
        length = item["input_ids"].shape[0]
        input_ids[i, :length] = item["input_ids"]
        attention_mask[i, :length] = item["attention_mask"]
        if labels is not None:
            labels[i, :length] = item["labels"]
        for key in extra_keys:
            extras[key][i, :length] = item[key]
    out = {"input_ids": input_ids, "attention_mask": attention_mask}
    if labels is not None:
        out["labels"] = labels
    out.update(extras)
    return out
