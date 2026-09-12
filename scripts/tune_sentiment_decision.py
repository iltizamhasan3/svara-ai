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
) -> dict[str, Any]:
    """Select neutral/negative biases by macro-F1 with a deterministic tie-break."""

    if len(probabilities) != len(expected) or not expected:
        raise ValueError("probabilities and expected labels must be non-empty and have equal length")
    if any(label not in CANONICAL_LABELS for label in expected):
        raise ValueError("expected labels must use canonical labels")
    evaluator = _evaluator_module()
    grid = _candidate_grid(bias_min, bias_max, bias_step)
    best: tuple[float, float, float, float, float, dict[str, float], dict[str, Any]] | None = None
    for neutral in grid:
        for negative in grid:
            biases = {"positive": 0.0, "neutral": neutral, "negative": negative}
            predicted = apply_biases(probabilities, biases)
            metrics = evaluator.classification_metrics(expected, predicted)
            score = metrics["metrics"]
            # max macro-F1, then min L1 bias, then max accuracy, then stable
            # numeric ordering of the two tunable biases.
            key = (score["macro_f1"], -abs(neutral) - abs(negative), score["accuracy"], -neutral, -negative)
            if best is None or key > best[:5]:
                best = (*key, biases, {"predicted": predicted, **metrics})
    assert best is not None
    return {"biases": best[5], **best[6]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    evaluator = _evaluator_module()
    input_path = args.input.resolve()
    rows, preparation_report = evaluator.read_labeled_tsv(input_path)
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
    calibrated = search_biases(
        probability_maps, expected, bias_min=args.bias_min, bias_max=args.bias_max, bias_step=args.bias_step
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
                 "positive_fixed": 0.0, "classes": list(CANONICAL_LABELS)},
        "baseline": baseline_metrics,
        "calibrated": {"biases": calibrated["biases"], **{key: value for key, value in calibrated.items() if key != "biases" and key != "predicted"}},
        "evaluation_policy": {"igar_read": False, "igar_labels_used": False,
                               "igar_metrics_used": False, "training_or_tuning": True},
    }
    if args.output is not None:
        output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/week3/model-v1")
    parser.add_argument("--export-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--bias-min", type=float, default=-1.0)
    parser.add_argument("--bias-max", type=float, default=1.0)
    parser.add_argument("--bias-step", type=float, default=0.1)
    return parser


if __name__ == "__main__":
    try:
        result = run(build_parser().parse_args())
        print(json.dumps({"baseline": result["baseline"]["metrics"], "calibrated": result["calibrated"]["metrics"]}, sort_keys=True))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
