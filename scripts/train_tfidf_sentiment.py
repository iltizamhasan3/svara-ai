#!/usr/bin/env python3
"""Train the non-IGAR word TF-IDF sentiment baseline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.ai.preprocessing import CANONICAL_LABELS, PREPROCESSING_VERSION, prepare_labeled_rows  # noqa: E402


def assert_not_igar_training_path(path: Path) -> None:
    resolved = path.resolve()
    if "igar" in {part.casefold() for part in resolved.parts} or "rating_labeled" in resolved.name.casefold():
        raise ValueError("IGAR is sealed external-test data and cannot be a training input")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_training_tsv(path: Path):
    assert_not_igar_training_path(path)
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, values in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) != 2:
                raise ValueError(f"{path}: line {line_number} must have text and label fields")
            rows.append({"text": values[0], "label": values[1]})
    prepared, report = prepare_labeled_rows(rows, text_column="text", label_column="label")
    if not prepared or {row.label for row in prepared} != set(CANONICAL_LABELS):
        raise ValueError("training input must contain every canonical label")
    return prepared, report


def train(input_path: Path, model_path: Path, manifest_path: Path, *, c: float = 4.0) -> dict:
    assert_not_igar_training_path(input_path)
    if c <= 0:
        raise ValueError("C must be positive")
    prepared, report = read_training_tsv(input_path)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, max_features=100000
    )
    classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=500, random_state=0)
    classifier.fit(vectorizer.fit_transform([row.text for row in prepared]), [row.label for row in prepared])
    model_path = model_path.resolve(); manifest_path = manifest_path.resolve()
    assert_not_igar_training_path(model_path); assert_not_igar_training_path(manifest_path)
    model_path.parent.mkdir(parents=True, exist_ok=True); manifest_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, model_path)
    manifest = {"artifact": "tfidf-sentiment-v1", "model_file": model_path.name, "model_sha256": sha256_file(model_path), "labels": list(CANONICAL_LABELS), "preprocessing_version": PREPROCESSING_VERSION, "training_input": {"path": str(input_path), "sha256": sha256_file(input_path), "rows": len(prepared), "preparation": report.__dict__}, "source": {"dataset": "Indonesian Google Play Review", "license": "CC BY 4.0", "role": "training_only"}, "vectorizer": {"analyzer": "word", "ngram_range": [1, 2], "sublinear_tf": True, "max_features": 100000}, "classifier": {"type": "LogisticRegression", "C": c, "class_weight": "balanced", "max_iter": 500}, "igar_forbidden": True, "igar_read": False, "igar_labels_used": False, "igar_metrics_used": False}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--C", type=float, default=4.0)
    return parser


if __name__ == "__main__":
    try:
        args = build_parser().parse_args()
        print(json.dumps(train(args.input, args.model, args.manifest, c=args.C)))
    except (OSError, ValueError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr); raise SystemExit(1)
