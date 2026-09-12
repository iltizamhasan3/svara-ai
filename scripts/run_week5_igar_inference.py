#!/usr/bin/env python3
"""Run reproducible external IGAR sentiment inference and error analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.model_bundle import ModelBundleError  # noqa: E402
from app.ai.preprocessing import (  # noqa: E402
    CANONICAL_LABELS,
    PreprocessingError,
    normalize_label,
)
from app.ai.sentiment_inference import (  # noqa: E402
    SentimentBatchInferencer,
    SentimentPrediction,
)


_CONTRASTIVE_MARKERS = (" tapi ", " namun ", " meskipun ", " walaupun ", " walau ")
_ERROR_CATEGORY_DESCRIPTIONS = {
    "mixed_signal": "Text contains a contrastive marker and merits mixed-sentiment review.",
    "short_text": "Text has three or fewer whitespace-delimited tokens.",
    "question_or_request": "Text contains a question marker and merits intent review.",
    "long_text": "Text has at least thirty whitespace-delimited tokens.",
    "other": "No deterministic triage pattern matched this error.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    """Read a non-empty CSV while preserving source row order."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV header is required")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path}: CSV must contain at least one data row")
    return rows


def _validate_columns(rows: Sequence[dict[str, object]], *, text_column: str, label_column: str) -> None:
    if not text_column.strip() or not label_column.strip():
        raise ValueError("text_column and label_column are required")
    for row_number, row in enumerate(rows, start=1):
        missing = [column for column in (text_column, label_column) if column not in row]
        if missing:
            raise ValueError(
                f"source row {row_number} is missing required column(s): {', '.join(missing)}"
            )


def error_category(text: str) -> str:
    """Assign a deterministic review bucket, not a causal model explanation."""

    normalized = f" {text.casefold()} "
    tokens = normalized.split()
    if any(marker in normalized for marker in _CONTRASTIVE_MARKERS):
        return "mixed_signal"
    if "?" in text:
        return "question_or_request"
    if len(tokens) <= 3:
        return "short_text"
    if len(tokens) >= 30:
        return "long_text"
    return "other"


def _prediction_row(
    prediction: SentimentPrediction,
    *,
    expected_label: str,
) -> dict[str, object]:
    correct = prediction.sentiment == expected_label
    return {
        "source_row_number": prediction.source_row_number,
        "text": prediction.text,
        "expected_label": expected_label,
        "predicted_label": prediction.sentiment,
        "correct": correct,
        "confidence": prediction.confidence,
        "positive_probability": prediction.probabilities.positive,
        "neutral_probability": prediction.probabilities.neutral,
        "negative_probability": prediction.probabilities.negative,
        "error_category": "" if correct else error_category(prediction.text),
    }


def _classification_metrics(
    expected_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> tuple[dict[str, float], dict[str, Any], list[list[int]]]:
    from sklearn.metrics import (  # noqa: PLC0415
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        expected_labels,
        predicted_labels,
        labels=list(CANONICAL_LABELS),
        average="macro",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(expected_labels, predicted_labels)),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(macro_f1),
    }
    report = classification_report(
        expected_labels,
        predicted_labels,
        labels=list(CANONICAL_LABELS),
        target_names=list(CANONICAL_LABELS),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        expected_labels,
        predicted_labels,
        labels=list(CANONICAL_LABELS),
    ).tolist()
    return metrics, report, matrix


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "source_row_number",
        "text",
        "expected_label",
        "predicted_label",
        "correct",
        "confidence",
        "positive_probability",
        "neutral_probability",
        "negative_probability",
        "error_category",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _artifact_reference(path: Path, *, output_dir: Path) -> str:
    """Use output-relative names so reports are reproducible anywhere."""

    return str(path.resolve().relative_to(output_dir.resolve()))


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    rows = read_csv_rows(input_path)
    _validate_columns(rows, text_column=args.text_column, label_column=args.label_column)

    inferencer = SentimentBatchInferencer.from_pretrained(
        str(args.model_dir.resolve()),
        batch_size=args.batch_size,
        max_length=args.max_length,
        device="cpu",
        local_files_only=True,
        torch_threads=args.torch_threads,
    )
    predictions, preparation_report = inferencer.predict_rows(
        rows,
        text_column=args.text_column,
    )
    if not predictions:
        raise ValueError("no non-empty text rows were available for evaluation")

    prediction_rows: list[dict[str, object]] = []
    expected_labels: list[str] = []
    predicted_labels: list[str] = []
    for prediction in predictions:
        source_row = rows[prediction.source_row_number - 1]
        try:
            expected_label = normalize_label(source_row[args.label_column])
        except PreprocessingError as exc:
            raise ValueError(
                f"source row {prediction.source_row_number} has an invalid evaluation label"
            ) from exc
        expected_labels.append(expected_label)
        predicted_labels.append(prediction.sentiment)
        prediction_rows.append(
            _prediction_row(prediction, expected_label=expected_label)
        )

    metrics, classification_report, confusion = _classification_metrics(
        expected_labels,
        predicted_labels,
    )
    errors = [row for row in prediction_rows if not row["correct"]]
    category_counts = Counter(str(row["error_category"]) for row in errors)
    confusion_pairs = Counter(
        f"{row['expected_label']}->{row['predicted_label']}" for row in errors
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "igar_predictions.csv"
    errors_path = output_dir / "igar_error_analysis.csv"
    metrics_path = output_dir / "igar_metrics.json"
    error_report_path = output_dir / "igar_error_analysis.json"
    _write_csv(predictions_path, prediction_rows)
    _write_csv(errors_path, errors)

    bundle = inferencer.loaded_model.bundle
    metrics_payload: dict[str, Any] = {
        "dataset": "IGAR",
        "input": {
            "path": _relative_path(input_path),
            "sha256": sha256_file(input_path),
            "input_rows": len(rows),
            "evaluated_rows": len(predictions),
            "skipped_missing_text": preparation_report.missing_text_rows,
            "duplicate_rows_retained": preparation_report.duplicate_rows,
        },
        "model": {
            "model_version": bundle.model_version,
            "name": bundle.model_name,
            "revision": bundle.model_revision,
            "preprocessing_version": bundle.preprocessing_version,
            "label_mapping": dict(bundle.label_to_id),
        },
        "metrics": metrics,
        "classification_report": classification_report,
        "confusion_matrix": {
            "labels": list(CANONICAL_LABELS),
            "values": confusion,
        },
        "external_evaluation": {
            "igar_only": True,
            "training_or_tuning": False,
            "caveat": "The tracked IGAR sample is a small external/domain-validation sample and is not representative of all SVARA feedback.",
        },
        "artifacts": {
            "predictions": _artifact_reference(predictions_path, output_dir=output_dir),
            "errors": _artifact_reference(errors_path, output_dir=output_dir),
            "metrics": _artifact_reference(metrics_path, output_dir=output_dir),
            "error_report": _artifact_reference(error_report_path, output_dir=output_dir),
        },
    }
    error_payload: dict[str, Any] = {
        "dataset": "IGAR",
        "evaluated_rows": len(predictions),
        "error_count": len(errors),
        "error_rate": round(len(errors) / len(predictions), 4),
        "category_counts": dict(sorted(category_counts.items())),
        "confusion_pairs": dict(sorted(confusion_pairs.items())),
        "category_definitions": _ERROR_CATEGORY_DESCRIPTIONS,
        "errors": errors,
        "interpretation": "Categories are deterministic review buckets for prioritization; they do not establish why the model made an error.",
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=float) + "\n", encoding="utf-8")
    error_report_path.write_text(json.dumps(error_payload, indent=2, default=float) + "\n", encoding="utf-8")
    return metrics_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/samples/igar/Rating_labeled_sample.csv",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "artifacts/week3/model-v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/week5/igar",
    )
    parser.add_argument("--text-column", default="content")
    parser.add_argument("--label-column", default="labelScoreBase")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser


if __name__ == "__main__":
    try:
        run(build_parser().parse_args())
    except (OSError, ValueError, ModelBundleError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
