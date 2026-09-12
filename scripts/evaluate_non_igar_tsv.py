#!/usr/bin/env python3
"""Evaluate a local sentiment bundle on a prepared, non-IGAR TSV holdout.

This command is intentionally separate from the IGAR evaluator. It fails
closed for IGAR-looking paths and accepts only a two-column ``text``/``label``
TSV whose labels normalize to the AI contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.model_bundle import ModelBundleError  # noqa: E402
from app.ai.preprocessing import (  # noqa: E402
    CANONICAL_LABELS,
    PreparedInferenceRow,
    normalize_label,
    prepare_labeled_rows,
)
from app.ai.sentiment_inference import SentimentBatchInferencer  # noqa: E402
from app.ai.sentiment_ensemble import blend_prediction, load_tfidf_model  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_not_igar_input_path(path: Path) -> None:
    resolved = path.resolve()
    if "igar" in {part.casefold() for part in resolved.parts} or "rating_labeled" in resolved.name.casefold():
        raise ValueError("IGAR is sealed external-test data and cannot be evaluated by this command")


def read_labeled_tsv(path: Path):
    assert_not_igar_input_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, values in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) != 2:
                raise ValueError(f"{path}: line {line_number} must have text and label fields")
            raw_rows.append({"text": values[0], "label": values[1]})
    rows, report = prepare_labeled_rows(raw_rows, text_column="text", label_column="label")
    if not rows:
        raise ValueError(f"{path}: no usable labeled rows")
    return rows, report


def load_decision_bias(path: Path | None) -> dict[str, float] | None:
    """Load a trusted calibration artifact without retuning on this input."""

    if path is None:
        return None
    path = path.resolve()
    assert_not_igar_input_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: decision-bias artifact is not valid JSON") from exc
    calibrated = artifact.get("calibrated") if isinstance(artifact, dict) else None
    raw_biases = calibrated.get("biases") if isinstance(calibrated, dict) else None
    if not isinstance(raw_biases, dict):
        raise ValueError(f"{path}: decision-bias artifact must contain calibrated.biases")
    expected = set(CANONICAL_LABELS)
    if set(raw_biases) != expected:
        raise ValueError(f"{path}: decision biases must cover exactly {sorted(expected)}")
    biases: dict[str, float] = {}
    for label in CANONICAL_LABELS:
        try:
            value = float(raw_biases[label])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: decision bias for {label} is not numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{path}: decision bias for {label} must be finite")
        biases[label] = value
    policy = artifact.get("evaluation_policy")
    if not isinstance(policy, dict) or policy.get("igar_read") is not False:
        raise ValueError(f"{path}: calibration artifact must prove IGAR was not read")
    return biases


def classification_metrics(expected: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    from sklearn.metrics import (  # noqa: PLC0415
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        expected,
        predicted,
        labels=list(CANONICAL_LABELS),
        average="macro",
        zero_division=0,
    )
    return {
        "metrics": {
            "accuracy": float(accuracy_score(expected, predicted)),
            "precision": float(precision),
            "recall": float(recall),
            "macro_f1": float(macro_f1),
        },
        "classification_report": classification_report(
            expected,
            predicted,
            labels=list(CANONICAL_LABELS),
            target_names=list(CANONICAL_LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            expected,
            predicted,
            labels=list(CANONICAL_LABELS),
        ).tolist(),
        "labels": list(CANONICAL_LABELS),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    rows, preparation_report = read_labeled_tsv(input_path)
    decision_bias = load_decision_bias(args.decision_bias_manifest)
    tfidf_model_path = getattr(args, "tfidf_model", None)
    tfidf_manifest_path = getattr(args, "tfidf_manifest", None)
    bert_weight = getattr(args, "bert_weight", 0.5)
    tfidf_model = None
    if tfidf_model_path is not None or tfidf_manifest_path is not None:
        if tfidf_model_path is None or tfidf_manifest_path is None:
            raise ValueError("--tfidf-model and --tfidf-manifest must be provided together")
        tfidf_model = load_tfidf_model(tfidf_model_path, tfidf_manifest_path)
    inferencer = SentimentBatchInferencer.from_pretrained(
        str(args.model_dir.resolve()),
        batch_size=args.batch_size,
        max_length=args.max_length,
        device="cpu",
        local_files_only=True,
        torch_threads=args.torch_threads,
        export_manifest_path=args.export_manifest,
        decision_bias=None if tfidf_model is not None else decision_bias,
    )
    predictions = inferencer.predict_prepared_rows(
        [PreparedInferenceRow(row.source_row_number, row.text) for row in rows]
    )
    if tfidf_model is not None:
        tfidf_probabilities = tfidf_model.predict_proba([row.text for row in rows])
        predictions = [
            blend_prediction(prediction, tfidf_values, bert_weight=bert_weight, decision_bias=decision_bias)
            for prediction, tfidf_values in zip(predictions, tfidf_probabilities, strict=True)
        ]
    expected = [normalize_label(row.label) for row in rows]
    predicted = [prediction.sentiment for prediction in predictions]
    bundle = inferencer.loaded_model.bundle
    payload: dict[str, Any] = {
        "dataset": args.dataset_name,
        "input": {
            "path": str(input_path.relative_to(ROOT)) if input_path.is_relative_to(ROOT) else str(input_path),
            "sha256": sha256_file(input_path),
            "evaluated_rows": len(predictions),
            "preparation": preparation_report.__dict__,
        },
        "model": {
            "path": str(bundle.path),
            "model_version": bundle.model_version,
            "model_name": bundle.model_name,
            "model_revision": bundle.model_revision,
            "preprocessing_version": bundle.preprocessing_version,
            "label_mapping": dict(bundle.label_to_id),
        },
        "inference": {
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "torch_threads": args.torch_threads,
            "local_files_only": True,
            "decision_bias_manifest": (
                str(args.decision_bias_manifest.resolve())
                if args.decision_bias_manifest is not None
                else None
            ),
            "decision_bias": decision_bias,
            "tfidf_model": str(tfidf_model_path.resolve()) if tfidf_model_path else None,
            "tfidf_manifest": str(tfidf_manifest_path.resolve()) if tfidf_manifest_path else None,
            "bert_weight": bert_weight if tfidf_model is not None else None,
        },
        **classification_metrics(expected, predicted),
        "evaluation_policy": {
            "igar_read": False,
            "igar_labels_used": False,
            "igar_metrics_used": False,
            "training_or_tuning": False,
            "decision_bias_applied": decision_bias is not None,
            "tfidf_used": tfidf_model is not None,
        },
    }
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="prepared non-IGAR text/label TSV")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/week3/model-v1")
    parser.add_argument("--export-manifest", type=Path, default=None)
    parser.add_argument(
        "--decision-bias-manifest",
        type=Path,
        default=None,
        help="calibration artifact produced on a separate non-IGAR validation set",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dataset-name", default="non-IGAR holdout")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--tfidf-model", type=Path, default=None)
    parser.add_argument("--tfidf-manifest", type=Path, default=None)
    parser.add_argument("--bert-weight", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    try:
        result = run(build_parser().parse_args())
        print(json.dumps(result["metrics"], sort_keys=True))
    except (OSError, ValueError, ModelBundleError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
