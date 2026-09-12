#!/usr/bin/env python3
"""Tune a constrained additive decision-bias calibration for a local model.

This is a development-set tuning artifact. It never reads or evaluates IGAR
data, and it does not train a model or evaluate a test split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.preprocessing import CANONICAL_LABELS, PreparedInferenceRow, normalize_label  # noqa: E402
from app.ai.sentiment_inference import SentimentBatchInferencer  # noqa: E402


def _evaluator_module():
    """Load the existing evaluator without making scripts a Python package."""

    path = Path(__file__).with_name("evaluate_non_igar_tsv.py")
    spec = importlib.util.spec_from_file_location("non_igar_tsv_evaluator", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"could not load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_biases(
    probabilities: Sequence[Mapping[str, float]],
    biases: Mapping[str, float],
) -> list[str]:
    """Apply additive logit biases to probability maps and return labels.

    Since log(probability) differs from the original logits by one shared
    normalization constant, argmax(log(probability) + bias) is equivalent to
    applying the bias to the model logits. Positive is constrained to zero by
    the search policy, but this function accepts and validates all classes.
    """

    labels = tuple(CANONICAL_LABELS)
    result: list[str] = []
    for row_number, probability_map in enumerate(probabilities, start=1):
        scores: dict[str, float] = {}
        for label in labels:
            try:
                probability = float(probability_map[label])
                bias = float(biases.get(label, 0.0))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"probability row {row_number} is malformed") from exc
            if not math.isfinite(probability) or probability <= 0:
                raise ValueError(f"probability row {row_number} has invalid {label} probability")
            if not math.isfinite(bias):
                raise ValueError(f"bias for {label} must be finite")
            scores[label] = math.log(probability) + bias
        # Canonical order makes ties reproducible and preserves positive as
        # the first choice when scores are exactly equal.
        result.append(max(labels, key=lambda label: scores[label]))
    return result


def _probability_mapping(probabilities: object) -> Mapping[str, float]:
    if isinstance(probabilities, Mapping):
        return probabilities
    dump = getattr(probabilities, "model_dump", None)
    if callable(dump):
        values = dump()
        if isinstance(values, Mapping):
            return values
    raise ValueError("inference probability map is malformed")


def _selection_metrics(expected: Sequence[str], predicted: Sequence[str]) -> dict[str, float]:
    """Return only the metrics needed while scanning the bias grid.

    The full evaluator builds a classification report for every class on every
    candidate.  That is useful for the selected candidate, but unnecessarily
    expensive inside a large grid search.  This compact implementation keeps
    the selection metric identical while deferring the full report to the end.
    """

    if len(expected) != len(predicted) or not expected:
        raise ValueError("expected and predicted labels must be non-empty and have equal length")
    label_to_index = {label: index for index, label in enumerate(CANONICAL_LABELS)}
    confusion = [[0 for _ in CANONICAL_LABELS] for _ in CANONICAL_LABELS]
    for actual, guess in zip(expected, predicted):
        try:
            confusion[label_to_index[actual]][label_to_index[guess]] += 1
        except KeyError as exc:
            raise ValueError("expected and predicted labels must use canonical labels") from exc
    true_positive = [confusion[index][index] for index in range(len(CANONICAL_LABELS))]
    actual_count = [sum(row) for row in confusion]
    predicted_count = [sum(confusion[row][column] for row in range(len(CANONICAL_LABELS))) for column in range(len(CANONICAL_LABELS))]
    f1_scores = []
    for index in range(len(CANONICAL_LABELS)):
        denominator = actual_count[index] + predicted_count[index]
        f1_scores.append(2 * true_positive[index] / denominator if denominator else 0.0)
    return {
        "accuracy": sum(true_positive) / len(expected),
        "macro_f1": sum(f1_scores) / len(f1_scores),
    }


def _candidate_grid(minimum: float, maximum: float, step: float) -> list[float]:
    if not all(math.isfinite(value) for value in (minimum, maximum, step)):
        raise ValueError("bias grid bounds and step must be finite")
    if step <= 0 or minimum > maximum:
        raise ValueError("bias grid requires min <= max and a positive step")
    count = math.floor((maximum - minimum) / step + 1e-10) + 1
    if count > 10001:
        raise ValueError("bias grid is too large; use a wider step")
    values = [minimum + index * step for index in range(count)]
    if not math.isclose(values[-1], maximum, rel_tol=0, abs_tol=1e-9):
        values.append(maximum)
    return values


def search_biases(
    probabilities: Sequence[Mapping[str, float]],
    expected: Sequence[str],
    *,
    bias_min: float = -1.0,
    bias_max: float = 1.0,
    bias_step: float = 0.1,
    guard_probabilities: Sequence[Mapping[str, float]] | None = None,
    guard_expected: Sequence[str] | None = None,
    min_guard_accuracy: float | None = None,
    min_guard_macro_f1: float | None = None,
    primary_accuracy_floor: float | None = None,
    primary_macro_f1_floor: float | None = None,
    selection_metric: str = "macro_f1",
) -> dict[str, Any]:
    """Select biases, optionally subject to non-IGAR guard constraints."""

    if len(probabilities) != len(expected) or not expected:
        raise ValueError("probabilities and expected labels must be non-empty and have equal length")
    if any(label not in CANONICAL_LABELS for label in expected):
        raise ValueError("expected labels must use canonical labels")
    guarded = any(value is not None for value in (
        guard_probabilities, guard_expected, min_guard_accuracy,
        min_guard_macro_f1, primary_accuracy_floor, primary_macro_f1_floor,
    ))
    if (guard_probabilities is None) != (guard_expected is None):
        raise ValueError("guard probabilities and expected labels must be supplied together")
    if guard_probabilities is not None and (len(guard_probabilities) != len(guard_expected) or not guard_expected):
        raise ValueError("guard probabilities and expected labels must be non-empty and have equal length")
    if guard_expected is not None and any(label not in CANONICAL_LABELS for label in guard_expected):
        raise ValueError("guard expected labels must use canonical labels")
    for name, value in (("minimum guard accuracy", min_guard_accuracy),
                        ("minimum guard macro-F1", min_guard_macro_f1),
                        ("primary accuracy floor", primary_accuracy_floor),
                        ("primary macro-F1 floor", primary_macro_f1_floor)):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be between 0 and 1")
    if (min_guard_accuracy is not None or min_guard_macro_f1 is not None) and guard_probabilities is None:
        raise ValueError("guard thresholds require --guard-input")
    if selection_metric not in {"macro_f1", "accuracy"}:
        raise ValueError("selection metric must be macro_f1 or accuracy")
    evaluator = _evaluator_module()
    grid = _candidate_grid(bias_min, bias_max, bias_step)
    best: tuple[float, ...] | None = None
    best_biases: dict[str, float] | None = None
    best_predicted: list[str] | None = None
    best_guard_predicted: list[str] | None = None
    for neutral in grid:
        for negative in grid:
            biases = {"positive": 0.0, "neutral": neutral, "negative": negative}
            predicted = apply_biases(probabilities, biases)
            score = _selection_metrics(expected, predicted)
            guard_score = None
            if guard_probabilities is not None:
                guard_predicted = apply_biases(guard_probabilities, biases)
                guard_score = _selection_metrics(guard_expected, guard_predicted)
            if primary_accuracy_floor is not None and score["accuracy"] < primary_accuracy_floor:
                continue
            if primary_macro_f1_floor is not None and score["macro_f1"] < primary_macro_f1_floor:
                continue
            if guard_score is not None:
                if (min_guard_accuracy is not None and guard_score["accuracy"] < min_guard_accuracy) or (
                    min_guard_macro_f1 is not None and guard_score["macro_f1"] < min_guard_macro_f1
                ):
                    continue
            if selection_metric == "accuracy":
                key = (score["accuracy"], score["macro_f1"], -abs(neutral) - abs(negative), -neutral, -negative)
            elif guarded:
                # Constrained selection prioritizes primary quality, then uses
                # the existing deterministic bias tie-breaks.
                key = (score["macro_f1"], score["accuracy"], -abs(neutral) - abs(negative), -neutral, -negative)
            else:
                # Preserve the original unconstrained ranking exactly.
                key = (score["macro_f1"], -abs(neutral) - abs(negative), score["accuracy"], -neutral, -negative)
            if best is None or key > best:
                best = key
                best_biases = biases
                best_predicted = predicted
                best_guard_predicted = guard_predicted if guard_probabilities is not None else None
    if best_biases is None or best_predicted is None:
        raise ValueError("no bias candidate satisfies the primary accuracy floor and guard thresholds")
    result: dict[str, Any] = {
        "biases": best_biases,
        "predicted": best_predicted,
        **evaluator.classification_metrics(expected, best_predicted),
    }
    if guard_probabilities is not None:
        assert guard_expected is not None and best_guard_predicted is not None
        result["guard_metrics"] = evaluator.classification_metrics(guard_expected, best_guard_predicted)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    evaluator = _evaluator_module()
    input_path = args.input.resolve()
    rows, preparation_report = evaluator.read_labeled_tsv(input_path)
    guard_input_arg = getattr(args, "guard_input", None)
    guard_input_path = guard_input_arg.resolve() if guard_input_arg is not None else None
    guard_rows = guard_preparation_report = None
    if guard_input_path is not None:
        guard_rows, guard_preparation_report = evaluator.read_labeled_tsv(guard_input_path)
    inferencer = SentimentBatchInferencer.from_pretrained(
        str(args.model_dir.resolve()), batch_size=args.batch_size, max_length=args.max_length,
        device="cpu", local_files_only=True, torch_threads=args.torch_threads,
        export_manifest_path=args.export_manifest,
    )
    predictions = inferencer.predict_prepared_rows(
        [PreparedInferenceRow(row.source_row_number, row.text) for row in rows]
    )
    expected = [normalize_label(row.label) for row in rows]
    probability_maps = [_probability_mapping(prediction.probabilities) for prediction in predictions]
    baseline_predicted = [prediction.sentiment for prediction in predictions]
    baseline_metrics = evaluator.classification_metrics(expected, baseline_predicted)
    guard_probability_maps = guard_expected = guard_baseline_metrics = None
    if guard_rows is not None:
        guard_predictions = inferencer.predict_prepared_rows(
            [PreparedInferenceRow(row.source_row_number, row.text) for row in guard_rows]
        )
        guard_expected = [normalize_label(row.label) for row in guard_rows]
        guard_probability_maps = [_probability_mapping(prediction.probabilities) for prediction in guard_predictions]
        guard_baseline_metrics = evaluator.classification_metrics(
            guard_expected, [prediction.sentiment for prediction in guard_predictions]
        )
    calibrated = search_biases(
        probability_maps, expected, bias_min=args.bias_min, bias_max=args.bias_max, bias_step=args.bias_step,
        guard_probabilities=guard_probability_maps, guard_expected=guard_expected,
        min_guard_accuracy=getattr(args, "min_guard_accuracy", None),
        min_guard_macro_f1=getattr(args, "min_guard_macro_f1", None),
        primary_accuracy_floor=getattr(args, "primary_accuracy_floor", None),
        primary_macro_f1_floor=getattr(args, "primary_macro_f1_floor", None),
        selection_metric=getattr(args, "selection_metric", "macro_f1"),
    )
    bundle = inferencer.loaded_model.bundle
    payload: dict[str, Any] = {
        "input": {
            "path": str(input_path.relative_to(ROOT)) if input_path.is_relative_to(ROOT) else str(input_path),
            "sha256": evaluator.sha256_file(input_path), "evaluated_rows": len(rows),
            "preparation": preparation_report.__dict__,
        },
        "model": {"path": str(bundle.path), "model_version": bundle.model_version,
                  "model_name": bundle.model_name, "model_revision": bundle.model_revision,
                  "preprocessing_version": bundle.preprocessing_version,
                  "label_mapping": dict(bundle.label_to_id)},
        "inference": {"batch_size": args.batch_size, "max_length": args.max_length,
                      "torch_threads": args.torch_threads, "local_files_only": True},
        "grid": {"bias_min": args.bias_min, "bias_max": args.bias_max, "bias_step": args.bias_step,
                 "positive_fixed": 0.0, "classes": list(CANONICAL_LABELS),
                 "primary_accuracy_floor": getattr(args, "primary_accuracy_floor", None),
                 "primary_macro_f1_floor": getattr(args, "primary_macro_f1_floor", None),
                 "min_guard_accuracy": getattr(args, "min_guard_accuracy", None),
                 "min_guard_macro_f1": getattr(args, "min_guard_macro_f1", None),
                 "selection_metric": getattr(args, "selection_metric", "macro_f1")},
        "baseline": baseline_metrics,
        "calibrated": {"biases": calibrated["biases"], **{
            key: value for key, value in calibrated.items()
            if key not in {"biases", "predicted", "guard_metrics"}
        }},
        "evaluation_policy": {"igar_read": False, "igar_labels_used": False,
                               "igar_metrics_used": False, "training_or_tuning": True},
    }
    if guard_rows is not None:
        selected_guard_metrics = calibrated.get("guard_metrics")
        assert selected_guard_metrics is not None
        payload["guard"] = {
            "input": {
                "path": str(guard_input_path.relative_to(ROOT)) if guard_input_path.is_relative_to(ROOT) else str(guard_input_path),
                "sha256": evaluator.sha256_file(guard_input_path), "evaluated_rows": len(guard_rows),
                "preparation": guard_preparation_report.__dict__,
            },
            "baseline": guard_baseline_metrics,
            "selected": selected_guard_metrics,
        }
    if args.output is not None:
        output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--guard-input", type=Path, default=None,
                        help="separate non-IGAR labeled TSV used only for candidate guards")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/week3/model-v1")
    parser.add_argument("--export-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--bias-min", type=float, default=-1.0)
    parser.add_argument("--bias-max", type=float, default=1.0)
    parser.add_argument("--bias-step", type=float, default=0.1)
    parser.add_argument("--min-guard-accuracy", type=float, default=None)
    parser.add_argument("--min-guard-macro-f1", type=float, default=None)
    parser.add_argument("--primary-accuracy-floor", "--min-primary-accuracy",
                        dest="primary_accuracy_floor", type=float, default=None)
    parser.add_argument("--primary-macro-f1-floor", type=float, default=None)
    parser.add_argument("--selection-metric", choices=("macro_f1", "accuracy"), default="macro_f1")
    return parser


if __name__ == "__main__":
    try:
        result = run(build_parser().parse_args())
        print(json.dumps({"baseline": result["baseline"]["metrics"], "calibrated": result["calibrated"]["metrics"]}, sort_keys=True))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
