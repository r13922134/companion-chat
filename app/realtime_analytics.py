from __future__ import annotations

from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from .storage import connect_database, google_form_account_key, load_json
except ImportError:
    from storage import connect_database, google_form_account_key, load_json


PHQ_MAX_SCORE = 24.0
PHQ_DEPRESSION_THRESHOLD = 10.0


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return int(round(number))


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    return None


def _iso_sort_key(value: Any) -> str:
    return str(value or "")


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return mean(clean)


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalize_phq_items(phq8: dict[str, Any]) -> list[dict[str, Any]]:
    item_scores = phq8.get("item_scores")
    raw_items = phq8.get("items")
    items_by_index: dict[int, dict[str, Any]] = {}
    if not isinstance(item_scores, dict) and isinstance(raw_items, list):
        for offset, raw_item in enumerate(raw_items, start=1):
            item = _safe_dict(raw_item)
            index = _int_or_none(item.get("index")) or offset
            items_by_index[index] = item
    items: list[dict[str, Any]] = []
    for index in range(1, 9):
        item = (
            item_scores.get(str(index))
            if isinstance(item_scores, dict)
            else items_by_index.get(index)
        )
        item = _safe_dict(item)
        score = _int_or_none(item.get("score"))
        items.append(
            {
                "index": index,
                "question": str(item.get("question") or f"PHQ-8 item {index}"),
                "answer": str(item.get("answer") or ""),
                "score": score,
            }
        )
    return items


def _normalize_phq8(phq8: Any) -> dict[str, Any]:
    phq8 = _safe_dict(phq8)
    total_score = _float_or_none(phq8.get("total_score"))
    items = _normalize_phq_items(phq8)
    if total_score is None and all(item["score"] is not None for item in items):
        total_score = float(sum(int(item["score"]) for item in items))
    return {
        "scale": str(phq8.get("scale") or "PHQ-8"),
        "total_score": total_score,
        "answered_count": _int_or_none(phq8.get("answered_count")),
        "max_score": _float_or_none(phq8.get("max_score")) or PHQ_MAX_SCORE,
        "binary_depression": None
        if total_score is None
        else total_score >= PHQ_DEPRESSION_THRESHOLD,
        "items": items,
    }


def _selected_user_from_metadata(metadata: Any) -> dict[str, Any]:
    metadata = _safe_dict(metadata)
    selected_user = dict(_safe_dict(metadata.get("selected_user")))
    selected_user.pop("respondent_email", None)
    selected_user.pop("email", None)
    account_id = str(
        selected_user.get("account_id")
        or selected_user.get("account_hash")
        or selected_user.get("session_hash")
        or selected_user.get("user_key")
        or metadata.get("account_id")
        or metadata.get("account_hash")
        or metadata.get("session_hash")
        or ""
    ).strip()
    if account_id:
        selected_user["account_id"] = account_id
        selected_user["account_hash"] = account_id
        selected_user["session_hash"] = account_id
        selected_user["user_key"] = account_id
    return selected_user


def _best_user_key(
    *,
    session_hash: str,
    selected_user: dict[str, Any],
    forms_by_hash: dict[str, dict[str, Any]],
) -> str:
    selected_form_hash = str(selected_user.get("latest_form_hash") or selected_user.get("form_hash") or "").strip()
    selected_form = forms_by_hash.get(selected_form_hash, {})
    candidates = [
        selected_user.get("account_id"),
        selected_user.get("account_hash"),
        selected_user.get("user_key"),
        session_hash,
        selected_user.get("session_hash"),
        selected_form.get("account_hash"),
        selected_user.get("form_hash"),
    ]
    for candidate in candidates:
        key = str(candidate or "").strip()
        if key:
            return key
    return "unknown"


def _aspect_rows(result: Any) -> list[dict[str, Any]]:
    result = _safe_dict(result)
    rows: list[dict[str, Any]] = []
    for index, aspect in enumerate(_safe_list(result.get("aspects")), start=1):
        aspect = _safe_dict(aspect)
        prediction = _float_or_none(aspect.get("prediction"))
        ground_truth = _float_or_none(aspect.get("ground_truth"))
        rows.append(
            {
                "index": int(_int_or_none(aspect.get("aspect_index")) or index - 1) + 1,
                "key": str(aspect.get("aspect_key") or ""),
                "label": str(
                    aspect.get("clinical_description")
                    or aspect.get("aspect")
                    or f"PHQ-8 item {index}"
                ),
                "prediction": prediction,
                "ground_truth": ground_truth,
                "abs_error": (
                    None
                    if prediction is None or ground_truth is None
                    else abs(prediction - ground_truth)
                ),
                "source": str(aspect.get("prediction_source") or ""),
            }
        )
    return rows


def _run_duration_seconds(started_at: str, ended_at: str | None) -> float | None:
    if not started_at or not ended_at:
        return None
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(ended_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (end - start).total_seconds())


def _load_forms(db_path: Path) -> dict[str, dict[str, Any]]:
    with connect_database(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                form_hash,
                form_dir_name,
                form_title,
                form_id,
                response_id,
                respondent_email,
                account_id,
                account_hash,
                name,
                age,
                submitted_at,
                received_at,
                date_prefix,
                fields_json,
                phq8_json
            FROM google_form_responses
            ORDER BY submitted_at DESC, received_at DESC, form_dir_name DESC
            """
        ).fetchall()

    forms: dict[str, dict[str, Any]] = {}
    for row in rows:
        form_hash = str(row["form_hash"] or "").strip()
        if not form_hash:
            continue
        respondent_email = str(row["respondent_email"] or "").strip().lower()
        account_id = str(row["account_id"] or "").strip() or google_form_account_key(
            respondent_email,
            str(row["account_hash"] or "").strip() or form_hash,
        )
        phq8 = _normalize_phq8(load_json(row["phq8_json"], {}))
        forms[form_hash] = {
            "user_key": account_id,
            "account_id": account_id,
            "account_hash": account_id,
            "session_hash": account_id,
            "form_hash": form_hash,
            "latest_form_hash": form_hash,
            "form_dir_name": row["form_dir_name"] or "",
            "form_title": row["form_title"] or "",
            "form_id": row["form_id"] or "",
            "response_id": row["response_id"] or "",
            "name": row["name"] or "",
            "age": row["age"] or "",
            "submitted_at": row["submitted_at"] or "",
            "received_at": row["received_at"] or "",
            "date_prefix": row["date_prefix"] or "",
            "fields": load_json(row["fields_json"], {}),
            "phq8": phq8,
        }
    return forms


def _load_runs(db_path: Path) -> list[dict[str, Any]]:
    with connect_database(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                session_hash,
                account_id,
                run_id,
                session_dir_name,
                started_at,
                ended_at,
                uploaded_at,
                metadata_json,
                transcript_text,
                depression_status,
                depression_total_score,
                depression_binary,
                depression_result_json,
                depression_error,
                depression_completed_at,
                ground_truth_total_score,
                ground_truth_binary
            FROM realtime_session_runs
            ORDER BY uploaded_at DESC, started_at DESC
            """
        ).fetchall()

    runs: list[dict[str, Any]] = []
    for row in rows:
        result = load_json(row["depression_result_json"], None)
        metadata = load_json(row["metadata_json"], {})
        transcript_text = str(row["transcript_text"] or "")
        prediction = _float_or_none(row["depression_total_score"])
        if prediction is None:
            prediction = _float_or_none(_safe_dict(result).get("total_score"))
        ground_truth = _float_or_none(row["ground_truth_total_score"])
        result_ground_truth = _safe_dict(_safe_dict(result).get("ground_truth"))
        if ground_truth is None:
            ground_truth = _float_or_none(result_ground_truth.get("total_score"))
        selected_user = _selected_user_from_metadata(metadata)
        snapshot_phq8 = _normalize_phq8(selected_user.get("phq8") or {})
        if ground_truth is None:
            ground_truth = _float_or_none(snapshot_phq8.get("total_score"))
        abs_error = (
            None
            if prediction is None or ground_truth is None
            else abs(prediction - ground_truth)
        )
        aspect_predictions = _fill_missing_aspect_ground_truth(
            _aspect_rows(result),
            snapshot_phq8,
        )
        runs.append(
            {
                "session_hash": row["session_hash"] or "",
                "account_id": row["account_id"] or row["session_hash"] or "",
                "run_id": row["run_id"] or "",
                "session_dir_name": row["session_dir_name"] or "",
                "started_at": row["started_at"] or "",
                "ended_at": row["ended_at"] or "",
                "uploaded_at": row["uploaded_at"] or "",
                "duration_seconds": _run_duration_seconds(
                    row["started_at"] or "",
                    row["ended_at"],
                ),
                "depression_status": row["depression_status"] or "",
                "prediction": prediction,
                "prediction_binary": _bool_or_none(row["depression_binary"]),
                "ground_truth": ground_truth,
                "ground_truth_binary": _bool_or_none(row["ground_truth_binary"]),
                "abs_error": abs_error,
                "error": row["depression_error"] or "",
                "completed_at": row["depression_completed_at"] or "",
                "transcript_character_count": len(transcript_text),
                "selected_user": selected_user,
                "aspect_predictions": aspect_predictions,
            }
        )
    return runs


def _fill_missing_aspect_ground_truth(
    aspect_predictions: list[dict[str, Any]],
    phq8: Any,
) -> list[dict[str, Any]]:
    normalized_phq8 = _normalize_phq8(phq8)
    item_scores: dict[int, float | None] = {}
    for item in _safe_list(normalized_phq8.get("items")):
        item_dict = _safe_dict(item)
        index = _int_or_none(item_dict.get("index"))
        if index is not None:
            item_scores[index] = _float_or_none(item_dict.get("score"))
    aspect_rows = []
    for aspect in _safe_list(aspect_predictions):
        aspect_copy = dict(_safe_dict(aspect))
        if _float_or_none(aspect_copy.get("ground_truth")) is not None:
            aspect_rows.append(aspect_copy)
            continue
        index = _int_or_none(aspect_copy.get("index"))
        truth = item_scores.get(index) if index is not None else None
        prediction = _float_or_none(aspect_copy.get("prediction"))
        if truth is not None:
            aspect_copy["ground_truth"] = truth
            aspect_copy["abs_error"] = (
                None if prediction is None else abs(prediction - truth)
            )
        aspect_rows.append(aspect_copy)
    return aspect_rows


def _build_users(
    *,
    forms_by_hash: dict[str, dict[str, Any]],
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    users: dict[str, dict[str, Any]] = {}
    latest_forms_by_account: dict[str, dict[str, Any]] = {}
    form_counts_by_account: dict[str, int] = {}

    for form in forms_by_hash.values():
        account_id = str(form.get("account_id") or form.get("account_hash") or form.get("form_hash") or "").strip()
        if not account_id:
            continue
        form_counts_by_account[account_id] = form_counts_by_account.get(account_id, 0) + 1
        current = latest_forms_by_account.get(account_id)
        if current is None or (
            _iso_sort_key(form.get("submitted_at")),
            _iso_sort_key(form.get("received_at")),
        ) > (
            _iso_sort_key(current.get("submitted_at")),
            _iso_sort_key(current.get("received_at")),
        ):
            latest_forms_by_account[account_id] = form

    for account_id, form in latest_forms_by_account.items():
        users[account_id] = {
            "user_key": account_id,
            "account_id": account_id,
            "account_hash": account_id,
            "session_hash": account_id,
            "form_hash": form.get("form_hash") or "",
            "latest_form_hash": form.get("form_hash") or "",
            "form_count": form_counts_by_account.get(account_id, 1),
            "name": form.get("name") or account_id[:8],
            "age": form.get("age") or "",
            "submitted_at": form.get("submitted_at") or "",
            "received_at": form.get("received_at") or "",
            "phq8": form.get("phq8") or _normalize_phq8({}),
            "runs": [],
        }

    for run in runs:
        selected_user = _safe_dict(run.get("selected_user"))
        user_key = _best_user_key(
            session_hash=str(run.get("account_id") or run.get("session_hash") or ""),
            selected_user=selected_user,
            forms_by_hash=forms_by_hash,
        )
        form = latest_forms_by_account.get(user_key)
        if form is None:
            selected_form_hash = str(
                selected_user.get("latest_form_hash") or selected_user.get("form_hash") or ""
            ).strip()
            selected_form = forms_by_hash.get(selected_form_hash, {})
            selected_account_id = str(selected_form.get("account_id") or selected_form.get("account_hash") or "").strip()
            if selected_account_id:
                user_key = selected_account_id
                form = latest_forms_by_account.get(user_key) or selected_form
            else:
                form = {}
        if user_key not in users:
            selected_phq = selected_user.get("phq8")
            users[user_key] = {
                "user_key": user_key,
                "account_id": str(selected_user.get("account_id") or selected_user.get("account_hash") or user_key),
                "account_hash": str(selected_user.get("account_id") or selected_user.get("account_hash") or user_key),
                "session_hash": str(selected_user.get("session_hash") or user_key),
                "form_hash": str(form.get("form_hash") or selected_user.get("form_hash") or user_key),
                "latest_form_hash": str(form.get("form_hash") or selected_user.get("latest_form_hash") or selected_user.get("form_hash") or ""),
                "form_count": form_counts_by_account.get(user_key, 0),
                "name": str(selected_user.get("name") or form.get("name") or user_key[:8]),
                "age": str(selected_user.get("age") or form.get("age") or ""),
                "submitted_at": str(
                    selected_user.get("submitted_at") or form.get("submitted_at") or ""
                ),
                "received_at": str(
                    selected_user.get("received_at") or form.get("received_at") or ""
                ),
                "phq8": _normalize_phq8(selected_phq or form.get("phq8") or {}),
                "runs": [],
            }
        user = users[user_key]
        if not user.get("name") and selected_user.get("name"):
            user["name"] = str(selected_user.get("name") or "")
        if not user.get("age") and selected_user.get("age"):
            user["age"] = str(selected_user.get("age") or "")
        if not user.get("phq8", {}).get("items"):
            user["phq8"] = _normalize_phq8(selected_user.get("phq8") or {})

        run_form_hash = str(
            selected_user.get("latest_form_hash") or selected_user.get("form_hash") or ""
        ).strip()
        run_form = forms_by_hash.get(run_form_hash, {})
        run_phq8 = _normalize_phq8(selected_user.get("phq8") or run_form.get("phq8") or {})
        if not run_form_hash:
            run_form_hash = str(run_form.get("form_hash") or "").strip()
        run_copy = {key: value for key, value in run.items() if key != "selected_user"}
        run_copy["user_key"] = user_key
        run_copy["account_id"] = user.get("account_id") or user.get("account_hash") or user_key
        run_copy["account_hash"] = user.get("account_hash") or user_key
        run_copy["form_hash"] = run_form_hash
        run_copy["gt_form_hash"] = run_form_hash
        run_copy["latest_form_hash"] = user.get("latest_form_hash") or user.get("form_hash") or ""
        run_copy["user_name"] = user.get("name") or user_key[:8]
        run_copy["age"] = str(selected_user.get("age") or run_form.get("age") or user.get("age") or "")
        run_copy["submitted_at"] = str(
            selected_user.get("submitted_at") or run_form.get("submitted_at") or ""
        )
        run_copy["received_at"] = str(
            selected_user.get("received_at") or run_form.get("received_at") or ""
        )
        run_copy["phq8"] = run_phq8
        user["runs"].append(run_copy)

    normalized_users: list[dict[str, Any]] = []
    for user in users.values():
        user_runs = sorted(
            user.get("runs") or [],
            key=lambda item: (_iso_sort_key(item.get("uploaded_at")), _iso_sort_key(item.get("started_at"))),
            reverse=True,
        )
        predictions = [_float_or_none(run.get("prediction")) for run in user_runs]
        abs_errors = [_float_or_none(run.get("abs_error")) for run in user_runs]
        completed_runs = [
            run for run in user_runs if str(run.get("depression_status") or "") == "ok"
        ]
        user["runs"] = user_runs
        user["latest_run"] = user_runs[0] if user_runs else None
        user["metrics"] = {
            "run_count": len(user_runs),
            "completed_prediction_count": len(completed_runs),
            "mean_prediction": _round(_mean(predictions)),
            "mean_abs_error": _round(_mean(abs_errors)),
            "latest_prediction": _round(
                _float_or_none(user_runs[0].get("prediction")) if user_runs else None
            ),
        }
        normalized_users.append(user)

    return sorted(
        normalized_users,
        key=lambda item: (
            _iso_sort_key(_safe_dict(item.get("latest_run")).get("uploaded_at")),
            _iso_sort_key(item.get("received_at")),
        ),
        reverse=True,
    )


def _summary(users: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [_float_or_none(run.get("prediction")) for run in runs]
    ground_truths = [_float_or_none(run.get("ground_truth")) for run in runs]
    abs_errors = [_float_or_none(run.get("abs_error")) for run in runs]
    phq_scores = [
        _float_or_none(_safe_dict(user.get("phq8")).get("total_score"))
        for user in users
    ]
    completed = [
        run for run in runs if str(run.get("depression_status") or "") == "ok"
    ]
    mismatches = 0
    comparable = 0
    for run in runs:
        pred = _float_or_none(run.get("prediction"))
        truth = _float_or_none(run.get("ground_truth"))
        if pred is None or truth is None:
            continue
        comparable += 1
        if (pred >= PHQ_DEPRESSION_THRESHOLD) != (truth >= PHQ_DEPRESSION_THRESHOLD):
            mismatches += 1
    return {
        "user_count": len(users),
        "run_count": len(runs),
        "completed_prediction_count": len(completed),
        "labeled_run_count": len([value for value in ground_truths if value is not None]),
        "mean_phq8_score": _round(_mean(phq_scores)),
        "mean_prediction_score": _round(_mean(predictions)),
        "mean_abs_error": _round(_mean(abs_errors)),
        "threshold_mismatch_count": mismatches,
        "threshold_comparable_count": comparable,
        "phq8_high_risk_user_count": len(
            [
                score
                for score in phq_scores
                if score is not None and score >= PHQ_DEPRESSION_THRESHOLD
            ]
        ),
        "predicted_high_risk_run_count": len(
            [
                score
                for score in predictions
                if score is not None and score >= PHQ_DEPRESSION_THRESHOLD
            ]
        ),
    }


def _aspect_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, Any]] = {}
    for run in runs:
        for aspect in _safe_list(run.get("aspect_predictions")):
            aspect = _safe_dict(aspect)
            index = _int_or_none(aspect.get("index"))
            if index is None:
                continue
            bucket = buckets.setdefault(
                index,
                {
                    "index": index,
                    "label": str(aspect.get("label") or f"PHQ-8 item {index}"),
                    "predictions": [],
                    "ground_truths": [],
                    "errors": [],
                },
            )
            bucket["predictions"].append(_float_or_none(aspect.get("prediction")))
            bucket["ground_truths"].append(_float_or_none(aspect.get("ground_truth")))
            bucket["errors"].append(_float_or_none(aspect.get("abs_error")))

    summary = []
    for index in sorted(buckets):
        bucket = buckets[index]
        summary.append(
            {
                "index": index,
                "label": bucket["label"],
                "mean_prediction": _round(_mean(bucket["predictions"])),
                "mean_ground_truth": _round(_mean(bucket["ground_truths"])),
                "mean_abs_error": _round(_mean(bucket["errors"])),
                "count": len([v for v in bucket["predictions"] if v is not None]),
            }
        )
    return summary


def build_realtime_analytics_snapshot(db_path: Path) -> dict[str, Any]:
    db_path = Path(db_path)
    forms_by_hash = _load_forms(db_path)
    runs = _load_runs(db_path)
    users = _build_users(forms_by_hash=forms_by_hash, runs=runs)

    flattened_runs: list[dict[str, Any]] = []
    for user in users:
        for run in _safe_list(user.get("runs")):
            run_copy = dict(run)
            run_copy["user_key"] = user.get("user_key")
            run_copy["user_name"] = user.get("name")
            flattened_runs.append(run_copy)
    flattened_runs.sort(
        key=lambda item: (_iso_sort_key(item.get("uploaded_at")), _iso_sort_key(item.get("started_at"))),
        reverse=True,
    )

    return {
        "status": "ok",
        "generated_at": _now_iso(),
        "source": {
            "database_path": str(db_path),
            "form_count": len(forms_by_hash),
            "realtime_run_count": len(flattened_runs),
        },
        "threshold": PHQ_DEPRESSION_THRESHOLD,
        "summary": _summary(users, flattened_runs),
        "aspect_summary": _aspect_summary(flattened_runs),
        "users": users,
        "runs": flattened_runs,
    }
