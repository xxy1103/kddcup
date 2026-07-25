from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from data_agent_baseline.asr_eval.core import SCHEMA_VERSION, _atomic_write_json, utc_now_iso

_OPENCC: Any | None = None
_ENGLISH_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float:
        return self.errors / self.reference_length if self.reference_length else 0.0


def _opencc_t2s(text: str) -> str:
    global _OPENCC
    if _OPENCC is None:
        try:
            from opencc import OpenCC
        except ImportError as exc:
            raise RuntimeError(
                "opencc-python-reimplemented is required for ASR normalization. "
                "Install the project dependencies first."
            ) from exc
        _OPENCC = OpenCC("t2s")
    return str(_OPENCC.convert(text))


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def normalize_chinese(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = _opencc_t2s(normalized).casefold()
    return "".join(
        char
        for char in normalized
        if _is_cjk(char) or ("a" <= char <= "z") or ("0" <= char <= "9")
    )


def normalize_english(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = _opencc_t2s(normalized).casefold()
    return _ENGLISH_TOKEN_PATTERN.findall(normalized)


def edit_counts(reference: Sequence[str] | str, hypothesis: Sequence[str] | str) -> EditCounts:
    ref = list(reference)
    hyp = list(hypothesis)
    rows = len(ref) + 1
    columns = len(hyp) + 1
    distance = [[0] * columns for _ in range(rows)]
    operation = [[""] * columns for _ in range(rows)]
    for row in range(1, rows):
        distance[row][0] = row
        operation[row][0] = "D"
    for column in range(1, columns):
        distance[0][column] = column
        operation[0][column] = "I"

    for row in range(1, rows):
        for column in range(1, columns):
            if ref[row - 1] == hyp[column - 1]:
                distance[row][column] = distance[row - 1][column - 1]
                operation[row][column] = "M"
                continue
            candidates = (
                (distance[row - 1][column - 1] + 1, "S"),
                (distance[row - 1][column] + 1, "D"),
                (distance[row][column - 1] + 1, "I"),
            )
            best_distance, best_operation = candidates[0]
            for candidate_distance, candidate_operation in candidates[1:]:
                if candidate_distance < best_distance:
                    best_distance, best_operation = candidate_distance, candidate_operation
            distance[row][column] = best_distance
            operation[row][column] = best_operation

    substitutions = deletions = insertions = 0
    row, column = len(ref), len(hyp)
    while row or column:
        current = operation[row][column]
        if current == "M":
            row -= 1
            column -= 1
        elif current == "S":
            substitutions += 1
            row -= 1
            column -= 1
        elif current == "D":
            deletions += 1
            row -= 1
        elif current == "I":
            insertions += 1
            column -= 1
        else:
            raise AssertionError(f"Missing edit operation at ({row}, {column}).")
    return EditCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_length=len(ref),
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ASR manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Incompatible ASR manifest: {path}")
    return payload


def _entries_by_task(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["task_id"]): item
        for item in manifest.get("samples") or []
        if isinstance(item, dict) and item.get("task_id")
    }


def _language_for_locale(locale: str) -> str:
    lowered = locale.lower()
    if lowered.startswith("zh"):
        return "zh"
    if lowered.startswith("en"):
        return "en"
    raise ValueError(f"Unsupported gold locale: {locale}")


def _language_matches(reference_language: str, detected_language: object) -> bool:
    detected = str(detected_language or "").strip().lower()
    return detected == reference_language or detected.startswith(reference_language + "-")


def score_asr_run(
    *,
    project_root: Path,
    gold_id: str,
    run_id: str,
    evaluation_root: Path | None = None,
    artifacts_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    eval_root = evaluation_root or (project_root / "evaluation" / "asr")
    output_root = artifacts_root or (project_root / "artifacts" / "asr")
    gold_manifest_path = eval_root / "gold" / gold_id / "manifest.json"
    run_dir = output_root / "runs" / run_id
    run_manifest_path = run_dir / "manifest.json"
    gold_manifest = _load_manifest(gold_manifest_path)
    run_manifest = _load_manifest(run_manifest_path)
    if gold_manifest.get("gold_id") != gold_id:
        raise ValueError("gold_id does not match its manifest.")
    if run_manifest.get("run_id") != run_id:
        raise ValueError("run_id does not match its manifest.")
    if not gold_manifest.get("completed"):
        raise ValueError("Gold manifest is incomplete; scoring partial Azure gold is not allowed.")

    gold_entries = _entries_by_task(gold_manifest)
    run_entries = _entries_by_task(run_manifest)
    task_ids = list(gold_manifest.get("requested_task_ids") or gold_entries)
    missing_gold = [
        task_id
        for task_id in task_ids
        if task_id not in gold_entries or gold_entries[task_id].get("status") != "ok"
    ]
    if missing_gold:
        raise ValueError(f"Gold is missing successful samples: {', '.join(missing_gold)}")

    per_sample: list[dict[str, Any]] = []
    for task_id in sorted(task_ids, key=_task_sort_key):
        gold = gold_entries[task_id]
        candidate = run_entries.get(task_id)
        if candidate is not None and candidate.get("audio_sha256") != gold.get("audio_sha256"):
            raise ValueError(
                f"Canonical audio hash mismatch for {task_id}; gold and candidate are not comparable."
            )
        language = _language_for_locale(str(gold.get("locale", "")))
        reference_text = str(gold.get("text", ""))
        candidate_ok = candidate is not None and candidate.get("status") == "ok"
        hypothesis_text = str(candidate.get("text", "")) if candidate_ok and candidate else ""
        if language == "zh":
            normalized_reference: str | list[str] = normalize_chinese(reference_text)
            normalized_hypothesis: str | list[str] = normalize_chinese(hypothesis_text)
            metric = "CER"
        else:
            normalized_reference = normalize_english(reference_text)
            normalized_hypothesis = normalize_english(hypothesis_text)
            metric = "WER"
        if len(normalized_reference) == 0:
            raise ValueError(f"Gold text becomes empty after normalization for {task_id}.")
        counts = edit_counts(normalized_reference, normalized_hypothesis)
        detected_language = candidate.get("detected_language") if candidate else None
        row = {
            "task_id": task_id,
            "locale": gold["locale"],
            "language": language,
            "metric": metric,
            "candidate_succeeded": candidate_ok,
            "substitutions": counts.substitutions,
            "deletions": counts.deletions,
            "insertions": counts.insertions,
            "reference_length": counts.reference_length,
            "errors": counts.errors,
            "error_rate": counts.error_rate,
            "detected_language": detected_language,
            "language_correct": _language_matches(language, detected_language) if candidate_ok else False,
            "elapsed_seconds": candidate.get("elapsed_seconds") if candidate else None,
            "real_time_factor": candidate.get("real_time_factor") if candidate else None,
            "failure_reason": None if candidate_ok else (
                candidate.get("error") if candidate else "candidate result missing"
            ),
            "reference_text": reference_text,
            "hypothesis_text": hypothesis_text,
            "normalized_reference": (
                normalized_reference
                if isinstance(normalized_reference, str)
                else " ".join(normalized_reference)
            ),
            "normalized_hypothesis": (
                normalized_hypothesis
                if isinstance(normalized_hypothesis, str)
                else " ".join(normalized_hypothesis)
            ),
        }
        per_sample.append(row)

    language_summaries: dict[str, dict[str, Any]] = {}
    for language, metric in (("zh", "CER"), ("en", "WER")):
        rows = [row for row in per_sample if row["language"] == language]
        if not rows:
            continue
        substitutions = sum(int(row["substitutions"]) for row in rows)
        deletions = sum(int(row["deletions"]) for row in rows)
        insertions = sum(int(row["insertions"]) for row in rows)
        reference_length = sum(int(row["reference_length"]) for row in rows)
        language_summaries[language] = {
            "metric": metric,
            "sample_count": len(rows),
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "reference_length": reference_length,
            "corpus_error_rate": (
                (substitutions + deletions + insertions) / reference_length
                if reference_length
                else 0.0
            ),
            "macro_error_rate": mean(float(row["error_rate"]) for row in rows),
        }

    successful = sum(1 for row in per_sample if row["candidate_succeeded"])
    language_correct = sum(1 for row in per_sample if row["language_correct"])
    elapsed_values = [
        float(row["elapsed_seconds"])
        for row in per_sample
        if row["elapsed_seconds"] is not None
    ]
    score_payload = {
        "schema_version": SCHEMA_VERSION,
        "score_type": "ASR relative to frozen Azure Fast Transcription output",
        "gold_id": gold_id,
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "normalization": {
            "unicode": "NFKC",
            "case": "casefold",
            "traditional_to_simplified": True,
            "chinese_units": "CJK characters plus ASCII letters and digits",
            "english_units": "ASCII alphanumeric words",
        },
        "overview": {
            "sample_count": len(per_sample),
            "successful_sample_count": successful,
            "success_rate": successful / len(per_sample) if per_sample else 0.0,
            "language_detection_correct_count": language_correct,
            "language_detection_accuracy": (
                language_correct / len(per_sample) if per_sample else 0.0
            ),
            "total_elapsed_seconds": sum(elapsed_values),
            "mean_elapsed_seconds": mean(elapsed_values) if elapsed_values else None,
        },
        "language_summaries": language_summaries,
        "samples": per_sample,
    }
    score_dir = run_dir / "scores" / gold_id
    score_path = score_dir / "score.json"
    _atomic_write_json(score_path, score_payload)
    _write_per_sample_csv(score_dir / "per_sample.csv", per_sample)
    (score_dir / "score_report.md").write_text(
        _render_score_report(score_payload),
        encoding="utf-8",
    )
    return score_path, score_payload


def _task_sort_key(task_id: str) -> tuple[int, int | str]:
    suffix = str(task_id).removeprefix("task_")
    return (0, int(suffix)) if suffix.isdigit() else (1, str(task_id))


def _write_per_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id",
        "locale",
        "language",
        "metric",
        "candidate_succeeded",
        "substitutions",
        "deletions",
        "insertions",
        "reference_length",
        "errors",
        "error_rate",
        "detected_language",
        "language_correct",
        "elapsed_seconds",
        "real_time_factor",
        "failure_reason",
        "normalized_reference",
        "normalized_hypothesis",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_score_report(score: dict[str, Any]) -> str:
    overview = score["overview"]
    lines = [
        f"# ASR Evaluation: {score['run_id']} vs {score['gold_id']}",
        "",
        (
            "> These metrics measure agreement with frozen Azure Fast Transcription output. "
            "They are not absolute transcription accuracy."
        ),
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Samples | {overview['sample_count']} |",
        (
            "| Successful candidates | "
            f"{overview['successful_sample_count']} "
            f"({overview['success_rate']:.2%}) |"
        ),
        (
            "| Language detection accuracy | "
            f"{overview['language_detection_correct_count']}/{overview['sample_count']} "
            f"({overview['language_detection_accuracy']:.2%}) |"
        ),
        f"| Total candidate time | {overview['total_elapsed_seconds']:.3f} s |",
        "",
        "## Error Rates",
        "",
        "| Language | Metric | Samples | Corpus | Macro | S | D | I | Reference units |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for language in ("zh", "en"):
        summary = score["language_summaries"].get(language)
        if not summary:
            continue
        lines.append(
            f"| {language} | {summary['metric']} | {summary['sample_count']} | "
            f"{summary['corpus_error_rate']:.4%} | {summary['macro_error_rate']:.4%} | "
            f"{summary['substitutions']} | {summary['deletions']} | "
            f"{summary['insertions']} | {summary['reference_length']} |"
        )
    lines.extend(
        [
            "",
            "## Per-sample Results",
            "",
            "| Task | Locale | Metric | Error rate | S | D | I | Success | Language |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in score["samples"]:
        lines.append(
            f"| {row['task_id']} | {row['locale']} | {row['metric']} | "
            f"{row['error_rate']:.4%} | {row['substitutions']} | {row['deletions']} | "
            f"{row['insertions']} | {'yes' if row['candidate_succeeded'] else 'no'} | "
            f"{row['detected_language'] or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)
