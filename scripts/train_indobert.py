#!/usr/bin/env python3
"""Train the Week 3 IndoBERT SmSA experiment.

Only the standard library and the repository's lightweight preprocessing
module are imported at module load time. Install the optional ``ai`` extra to
run the actual experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.preprocessing import (  # noqa: E402
    CANONICAL_LABELS,
    PREPROCESSING_VERSION,
    PreparedRow,
    normalize_text,
    prepare_labeled_rows,
)

MODEL_NAME = "indobenchmark/indobert-base-p1"
MODEL_REVISION = "c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2"
LABEL_TO_ID = {label: index for index, label in enumerate(CANONICAL_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> dict[str, str]:
    """Return checksums for every file below an artifact directory."""

    return {
        str(file.relative_to(path)): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def read_smsa(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, values in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) != 2:
                raise ValueError(f"{path}: line {line_number} must have two TSV fields")
            rows.append({"sentence": values[0], "label": values[1]})
    return rows


def assert_disjoint_splits(splits: Mapping[str, Iterable[PreparedRow]]) -> None:
    """Fail closed if normalized text occurs in two prepared splits."""

    seen: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            key = normalize_text(row.text).casefold()
            if not key:
                continue
            previous = seen.get(key)
            if previous is not None and previous != split:
                raise ValueError(f"normalized text overlaps {previous} and {split}: {row.text[:80]!r}")
            seen[key] = split


def test_evaluation_allowed(split_manifest: Mapping[str, Any], requested: bool) -> bool:
    """Require both the CLI opt-in and an explicit frozen-manifest condition."""

    if not requested:
        return False
    frozen = split_manifest.get("test_evaluation_frozen") is True
    evaluation = split_manifest.get("evaluation")
    if isinstance(evaluation, Mapping):
        frozen = frozen or evaluation.get("test_frozen") is True
    if not frozen:
        raise ValueError(
            "test evaluation requires --allow-test-evaluation and an explicit "
            "test_evaluation_frozen=true condition in the split manifest"
        )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/smsa")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "artifacts/week2/smsa_split_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/week3")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-train-rows", type=int, default=None, help="deterministic smoke-test cap")
    parser.add_argument("--allow-test-evaluation", action="store_true", default=False)
    return parser


def _resolve_data_path(data_dir: Path, manifest_path: str) -> Path:
    """Resolve a split file below the caller-selected SmSA data directory.

    Week 2 manifests store repository-relative paths. The training CLI treats
    ``--data-dir`` as the directory containing the three split files, so a
    relative manifest path contributes only its filename. Absolute manifest
    paths remain supported for one-off fixtures and external data layouts.
    """

    candidate = Path(manifest_path)
    if candidate.is_absolute():
        return candidate
    return data_dir / candidate.name


def load_splits(data_dir: Path, manifest_path: Path) -> tuple[dict[str, list[PreparedRow]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "SmSA":
        raise ValueError("split manifest must describe SmSA")
    if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("split manifest preprocessing version does not match the code")
    prepared: dict[str, list[PreparedRow]] = {}
    for split in SPLITS:
        details = manifest["splits"][split]
        path = _resolve_data_path(data_dir, details["file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = details.get("sha256")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
            raise ValueError(f"{path}: split manifest must include a 64-character SHA-256")
        if sha256_file(path).casefold() != expected.casefold():
            raise ValueError(f"{path}: SHA-256 does not match split manifest")
        rows, _ = prepare_labeled_rows(read_smsa(path), text_column="sentence", label_column="label")
        prepared[split] = rows
    protected = {
        normalize_text(row.text).casefold()
        for split in ("validation", "test")
        for row in prepared[split]
    }
    prepared["train"] = [row for row in prepared["train"] if normalize_text(row.text).casefold() not in protected]
    assert_disjoint_splits(prepared)
    return prepared, manifest


def _write_matrix(output_dir: Path, matrix: list[list[int]], *, split: str = "validation") -> dict[str, str]:
    suffix = "" if split == "validation" else f"_{split}"
    paths: dict[str, str] = {}
    json_path = output_dir / f"confusion_matrix{suffix}.json"
    json_path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    paths["json"] = str(json_path)
    csv_path = output_dir / f"confusion_matrix{suffix}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["label", *CANONICAL_LABELS])
        for label, values in zip(CANONICAL_LABELS, matrix):
            writer.writerow([label, *values])
    paths["csv"] = str(csv_path)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return paths
    image_path = output_dir / f"confusion_matrix{suffix}.png"
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(3), yticks=range(3), xticklabels=CANONICAL_LABELS, yticklabels=CANONICAL_LABELS,
             xlabel="Predicted", ylabel="Actual", title=f"SmSA {split} confusion matrix")
    for row_index, values in enumerate(matrix):
        for column_index, value in enumerate(values):
            axis.text(column_index, row_index, value, ha="center", va="center")
    figure.tight_layout(); figure.savefig(image_path, dpi=150); plt.close(figure)
    paths["png"] = str(image_path)
    return paths


def _prediction_artifacts(rows: list[PreparedRow], prediction_output: Any, np: Any, metrics_api: Mapping[str, Any]) -> dict[str, Any]:
    """Build explicit metrics/report/matrix data for one evaluated split."""

    raw_predictions = getattr(prediction_output, "predictions", prediction_output)
    if isinstance(raw_predictions, tuple):
        raw_predictions = raw_predictions[0]
    labels = np.array([LABEL_TO_ID[row.label] for row in rows])
    predicted = np.argmax(raw_predictions, axis=-1)
    accuracy_score = metrics_api["accuracy_score"]
    classification_report = metrics_api["classification_report"]
    confusion_matrix = metrics_api["confusion_matrix"]
    precision_recall_fscore_support = metrics_api["precision_recall_fscore_support"]
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=list(range(3)), average="macro", zero_division=0
    )
    return {
        "metrics": {
            "accuracy": float(accuracy_score(labels, predicted)),
            "precision": float(precision),
            "recall": float(recall),
            "macro_f1": float(macro_f1),
        },
        "classification_report": classification_report(
            labels,
            predicted,
            labels=list(range(3)),
            target_names=CANONICAL_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=list(range(3))).tolist(),
        "labels": list(CANONICAL_LABELS),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    prepared, split_manifest = load_splits(args.data_dir.resolve(), args.split_manifest.resolve())
    evaluate_test = test_evaluation_allowed(split_manifest, args.allow_test_evaluation)
    if args.max_train_rows is not None:
        if args.max_train_rows < 1:
            raise ValueError("--max-train-rows must be positive")
        prepared["train"] = prepared["train"][: args.max_train_rows]

    # Heavy optional dependencies begin here; importing this module alone stays lightweight.
    import numpy as np
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.max_length < 8:
        raise ValueError("--max-length must be at least 8")
    seed = args.seed
    random.seed(seed); np.random.seed(seed); set_seed(seed)
    output_dir = args.output_dir.resolve()
    model_dir = output_dir / "model-v1"
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, num_labels=3,
        id2label=ID_TO_LABEL, label2id=LABEL_TO_ID)
    encoded: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        rows = prepared[split]
        encoded[split] = Dataset.from_dict({"text": [r.text for r in rows], "labels": [LABEL_TO_ID[r.label] for r in rows]})
        encoded[split] = encoded[split].map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_length), batched=True, remove_columns=["text"])

    def metrics(eval_pred: Any) -> dict[str, float]:
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predicted = np.argmax(predictions, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, labels=list(range(3)), average="macro", zero_division=0)
        return {"accuracy": float(accuracy_score(labels, predicted)), "precision": float(precision), "recall": float(recall), "macro_f1": float(f1)}

    ta_kwargs: dict[str, Any] = dict(output_dir=str(checkpoint_dir), num_train_epochs=args.epochs, learning_rate=2e-5, weight_decay=.01,
        per_device_train_batch_size=4, per_device_eval_batch_size=8, gradient_accumulation_steps=2, dataloader_num_workers=0,
        seed=seed, report_to=[], logging_strategy="steps", eval_accumulation_steps=1, save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True)
    ta_params = inspect.signature(TrainingArguments).parameters
    ta_kwargs["eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"] = "epoch"
    if "save_strategy" in ta_params:
        ta_kwargs["save_strategy"] = "epoch"
    if "use_cpu" in ta_params: ta_kwargs["use_cpu"] = True
    elif "no_cuda" in ta_params: ta_kwargs["no_cuda"] = True
    training_args = TrainingArguments(**ta_kwargs)
    trainer_kwargs: dict[str, Any] = dict(model=model, args=training_args, train_dataset=encoded["train"], eval_dataset=encoded["validation"],
        tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
    if "processing_class" in inspect.signature(Trainer).parameters:
        trainer_kwargs["processing_class"] = tokenizer; trainer_kwargs.pop("tokenizer")
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    validation_evaluation = trainer.evaluate(encoded["validation"])
    metrics_api = {
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "precision_recall_fscore_support": precision_recall_fscore_support,
    }
    validation_predictions = trainer.predict(encoded["validation"])
    validation_artifacts = _prediction_artifacts(prepared["validation"], validation_predictions, np, metrics_api)
    test_artifacts = None
    if evaluate_test:
        test_predictions = trainer.predict(encoded["test"])
        test_artifacts = _prediction_artifacts(prepared["test"], test_predictions, np, metrics_api)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    artifact_paths: dict[str, str] = {}
    validation_matrix_paths = _write_matrix(output_dir, validation_artifacts["confusion_matrix"], split="validation")
    artifact_paths.update({f"confusion_matrix_{key}": path for key, path in validation_matrix_paths.items()})
    report_path = output_dir / "classification_report.json"
    report_path.write_text(json.dumps(validation_artifacts["classification_report"], indent=2, default=float) + "\n", encoding="utf-8")
    artifact_paths["classification_report"] = str(report_path)
    metrics_path = output_dir / "metrics.json"
    result = {
        "split": "validation",
        "metrics": validation_artifacts["metrics"],
        "trainer_evaluation": validation_evaluation,
        **validation_artifacts,
    }
    if test_artifacts is not None:
        test_matrix_paths = _write_matrix(output_dir, test_artifacts["confusion_matrix"], split="test")
        artifact_paths.update({f"test_confusion_matrix_{key}": path for key, path in test_matrix_paths.items()})
        test_report_path = output_dir / "classification_report_test.json"
        test_report_path.write_text(json.dumps(test_artifacts["classification_report"], indent=2, default=float) + "\n", encoding="utf-8")
        artifact_paths["test_classification_report"] = str(test_report_path)
        result["test"] = test_artifacts
    metrics_path.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    artifact_paths["metrics"] = str(metrics_path)
    artifact_paths["model"] = str(model_dir)
    checksums = {
        path: sha256_file(Path(path))
        for path in artifact_paths.values()
        if Path(path).is_file()
    }
    checksums["model-v1/*"] = sha256_tree(model_dir)
    best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    experiment = {"model": MODEL_NAME, "revision": MODEL_REVISION, "preprocessing_version": PREPROCESSING_VERSION,
        "label_mapping": LABEL_TO_ID, "config": {"seed": seed, "max_length": args.max_length, "train_batch_size": 4, "eval_batch_size": 8, "gradient_accumulation_steps": 2, "epochs": args.epochs, "learning_rate": 2e-5, "weight_decay": .01, "workers": 0, "use_cpu": True},
        "split_manifest_sha256": sha256_file(args.split_manifest.resolve()), "row_counts": {key: len(value) for key, value in prepared.items()},
        "checkpoint_identifier": best_checkpoint,
        "input_split_sha256": {split: split_manifest["splits"][split]["sha256"] for split in SPLITS},
        "artifact_paths": artifact_paths, "artifact_checksums": checksums, "test_evaluation": {"enabled": evaluate_test, "status": "evaluated" if evaluate_test else "not_evaluated"},
        "igar_external_only": True}
    (output_dir / "experiment_manifest.json").write_text(json.dumps(experiment, indent=2) + "\n", encoding="utf-8")
    return experiment


if __name__ == "__main__":
    try:
        train(build_parser().parse_args())
    except (OSError, ValueError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
