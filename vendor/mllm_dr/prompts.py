from __future__ import annotations

PHQ8_ASPECTS: tuple[tuple[str, str], ...] = (
    ("PHQ_8NoInterest", "interest or pleasure loss"),
    ("PHQ_8Depressed", "low mood or depressed feelings"),
    ("PHQ_8Sleep", "sleep problems"),
    ("PHQ_8Tired", "tiredness or low energy"),
    ("PHQ_8Appetite", "appetite problems"),
    ("PHQ_8Failure", "feelings of failure or self-blame"),
    ("PHQ_8Concentrating", "difficulty concentrating"),
    ("PHQ_8Moving", "psychomotor changes"),
)


PHQ8_CLINICAL_DESCRIPTIONS: tuple[str, ...] = (
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself or that you are a failure",
    "Trouble concentrating on things such as reading or watching television",
    "Moving or speaking so slowly, or being so fidgety or restless",
)


SYSTEM_PROMPT = (
    "You are a psychiatrist who evaluates participants' mental health "
    "conditions through a series of interview questions. A score of 0 means "
    "\"not at all,\" 1 means \"several days,\" 2 means \"more than half the "
    "days,\" and 3 means \"nearly every day.\""
)


LQ_SYSTEM_PROMPT = (
    "You are a multimodal intelligent assistant capable of analyzing visual "
    "and acoustic features of the input and providing an evaluation result."
)


MLLM_USER_TEMPLATE = (
    "{dialogue}\n\n"
    "Please evaluate the participant's psychological state regarding {aspect} "
    "based on the dialogue content. The output format should be--Evaluation "
    "result: <score>, followed by Evaluation reason: with a brief and "
    "objective explanation. The score must be exactly one digit selected from "
    "0, 1, 2, or 3. Each output must strictly follow this format to avoid "
    "ambiguity. For example, Evaluation result: 2. Evaluation reason: The "
    "participant shows partial loss of interest but still engages in some "
    "activities.\n\n<AudioHere>\n<VideoHere>"
)


MLLM_TRACT_USER_TEMPLATE = (
    "{dialogue}\n\n"
    "Please evaluate the participant's psychological state regarding {aspect} "
    "based on the dialogue content. First provide Evaluation reason: with a "
    "brief and objective explanation, then end with exactly one final score "
    "phrase using one digit selected from 0, 1, 2, or 3: So the overall score "
    "is <score>. Do not write ranges, translations, extra scores, or text "
    "after the final score phrase. For example, Evaluation reason: The "
    "participant shows partial loss of interest but still engages in some "
    "activities. So the overall score is 2.\n\n<AudioHere>\n<VideoHere>"
)


MLLM_SCORE_ONLY_USER_TEMPLATE = (
    "{dialogue}\n\n"
    "Please evaluate the participant's psychological state regarding {aspect} "
    "based on the dialogue content and multimodal features. Output exactly one "
    "score phrase using one digit selected from 0, 1, 2, or 3: Evaluation "
    "result: <score>. Do not write a rationale, ranges, translations, extra "
    "scores, or text after the score.\n\n<AudioHere>\n<VideoHere>"
)


MLLM_RATIONALE_ONLY_USER_TEMPLATE = (
    "{dialogue}\n\n"
    "Write an evidence-only rationale for the participant's {aspect}. The "
    "rationale must be grounded in the dialogue and observable audio/visual "
    "evidence represented by the multimodal feature tokens. Do not output a "
    "score, label, PHQ category, frequency category, range, or final "
    "evaluation result. Output exactly one line in this format: Evaluation "
    "reason: <1-3 concise sentences>."
    "\n\n<AudioHere>\n<VideoHere>"
)


MLLM_PRIVATE_LABEL_RATIONALE_USER_TEMPLATE = (
    "{dialogue}\n\n"
    "You are given a private PHQ-8 supervision target for the participant's "
    "{aspect}. Private target: {private_label}. Use this target only to "
    "calibrate the severity implied by the dialogue and multimodal evidence.\n\n"
    "Write an evidence-only rationale grounded in the dialogue plus the "
    "observable audio and visual cues represented by the multimodal feature "
    "tokens. Do not reveal, quote, or refer to the private target. Do not "
    "mention self-rating, self-report, score, label, PHQ, numeric values, or "
    "frequency categories such as several days or nearly every day. Do not say "
    "that the participant has a score. Do not invent facts not supported by "
    "the dialogue or multimodal cues.\n\n"
    "Output exactly one line in this format: Evaluation reason: <1-3 concise "
    "sentences describing clinical evidence for the participant's {aspect}>."
    "\n\n<AudioHere>\n<VideoHere>"
)


LQ_USER_TEMPLATE = (
    "Based on the given visual and speech features, evaluate the participant's "
    "level of {aspect}. The output format should be--Evaluation result: "
    "<score>. The score must be exactly one digit selected from 0, 1, 2, or "
    "3. Ensure that only one result is produced. For example, Evaluation "
    "result: 2.\n\n<AudioHere>\n<VideoHere>"
)


def merge_system_into_user_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    system_parts = [
        message["content"].strip()
        for message in messages
        if message["role"].strip().lower() == "system" and message["content"].strip()
    ]
    if not system_parts:
        return messages
    merged: list[dict[str, str]] = []
    system_prefix = "System instruction:\n" + "\n\n".join(system_parts)
    injected = False
    for message in messages:
        role = message["role"].strip().lower()
        if role == "system":
            continue
        if role == "user" and not injected:
            merged.append(
                {
                    "role": message["role"],
                    "content": (
                        f"{system_prefix}\n\nUser task:\n"
                        f"{message['content'].strip()}"
                    ),
                }
            )
            injected = True
        else:
            merged.append(message)
    if not injected:
        merged.insert(0, {"role": "user", "content": system_prefix})
    return merged


def build_dialogue(
    transcript_rows: list[tuple[float, float, str]],
    utterance_indices: list[int] | None = None,
) -> str:
    parts: list[str] = []
    for index, (_, _, text) in enumerate(transcript_rows, start=1):
        text = " ".join(str(text).split())
        if text:
            utterance_index = (
                utterance_indices[index - 1]
                if utterance_indices is not None and index <= len(utterance_indices)
                else index
            )
            parts.append(f"Utterance {utterance_index}: {text}")
    return "\n".join(parts)


def build_lq_messages(aspect: str, label: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": LQ_SYSTEM_PROMPT},
        {"role": "user", "content": LQ_USER_TEMPLATE.format(aspect=aspect)},
        {"role": "assistant", "content": f"Evaluation result: {int(label)}."},
    ]


def build_mllm_messages(
    dialogue: str,
    aspect: str,
    label: int | None = None,
    rationale: str | None = None,
    score_after_rationale: bool = False,
    score_only_response: bool = False,
    rationale_only_response: bool = False,
    private_label: int | None = None,
) -> list[dict[str, str]]:
    if score_only_response:
        user_template = MLLM_SCORE_ONLY_USER_TEMPLATE
    elif rationale_only_response and private_label is not None:
        user_template = MLLM_PRIVATE_LABEL_RATIONALE_USER_TEMPLATE
    elif rationale_only_response:
        user_template = MLLM_RATIONALE_ONLY_USER_TEMPLATE
    elif score_after_rationale:
        user_template = MLLM_TRACT_USER_TEMPLATE
    else:
        user_template = MLLM_USER_TEMPLATE
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_template.format(
                dialogue=dialogue,
                aspect=aspect,
                private_label=private_label,
            ),
        },
    ]
    if label is not None:
        reason = rationale or "The participant's responses support this score."
        if reason.startswith("Evaluation Reason:"):
            reason = reason.split(":", 1)[1].strip()
        if score_only_response:
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Evaluation result: {int(label)}.",
                }
            )
            return messages
        if rationale_only_response:
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Evaluation reason: {reason}",
                }
            )
            return messages
        if score_after_rationale:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Evaluation reason: {reason} "
                        f"So the overall score is {int(label)}."
                    ),
                }
            )
            return messages
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f"Evaluation result: {int(label)}. "
                    f"Evaluation reason: {reason}"
                ),
            }
        )
    return messages
