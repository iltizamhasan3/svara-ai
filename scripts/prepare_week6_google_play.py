#!/usr/bin/env python3
"""Prepare Indonesian Google Play Review rows for non-IGAR adaptation.

The published Google Play source exposes binary labels and a five-point
rating. This preparer derives the project's three-class target from that
rating, removes ambiguous normalized texts, and writes a deterministic
stratified split. IGAR is read only for exact-text overlap exclusion; its
labels, scores, and predictions are never consumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.ai.preprocessing import (  # noqa: E402
    PREPROCESSING_VERSION,
    PreparedRow,
    normalize_text,
)
from prepare_week6_data import (  # noqa: E402
    IGAR_SCHEMA,
    SPLITS,
    _default_igar_guard_paths,
    _display_path,
    _remove_protected_overlaps,
    _write_tsv,
    load_igar_text_keys,
    load_smsa_splits,
    sha256_file,
    stratified_split,
)


SOURCE_KEY = "google_play_review"
SCHEMA = ["text", "label", "stars"]


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def assert_not_igar_training_path(path: Path) -> None:
    resolved = path.resolve()
    if "igar" in {part.casefold() for part in resolved.parts} or "rating_labeled" in resolved.name.casefold():
        raise ValueError("IGAR is sealed external-test data and cannot be a training input")


def _load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("training_policy", {}).get("igar_forbidden") is not True:
        raise ValueError("training manifest must explicitly forbid IGAR")
    source = manifest.get("sources", {}).get(SOURCE_KEY)
    if not isinstance(source, dict) or source.get("role") != "additional_training":
        raise ValueError("training manifest must identify Google Play Review as additional_training")
    train_file = next(
        (details for details in source.get("files", []) if details.get("name") == "train.csv"),
        None,
    )
    if not isinstance(train_file, dict):
        raise ValueError("Google Play Review training file is missing from the manifest")
    return manifest, source, train_file


def _derived_label(raw_label: str, stars_value: str, *, row_number: int) -> str:
    try:
        stars = int(stars_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"source row {row_number} has a non-integer stars value") from error
    if stars not in {1, 2, 3, 4, 5}:
        raise ValueError(f"source row {row_number} has stars outside 1..5")
    normalized_raw_label = raw_label.strip().casefold()
    expected_raw_label = "pos" if stars in {4, 5} else "neg"
    if normalized_raw_label != expected_raw_label:
        raise ValueError(
            f"source row {row_number} has inconsistent published label/rating: "
            f"{normalized_raw_label!r} with {stars} stars"
        )
    if stars in {1, 2}:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


def load_google_play_rows(path: Path, training_manifest_path: Path) -> tuple[list[PreparedRow], dict[str, Any]]:
    """Load raw train.csv and derive a conflict-free, three-class view."""

    assert_not_igar_training_path(path)
    _, source, train_file = _load_manifest(training_manifest_path.resolve())
    if sha256_file(path) != train_file["expected_sha256"]:
        raise ValueError(f"{path}: SHA-256 does not match the Google Play training manifest")

    grouped: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    missing_text_rows = 0
    input_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SCHEMA:
            raise ValueError(f"{path}: schema mismatch: {reader.fieldnames}")
        for source_row_number, row in enumerate(reader, start=1):
            input_rows += 1
            text = normalize_text(row.get("text"))
            if not text:
                missing_text_rows += 1
                continue
            label = _derived_label(str(row.get("label", "")), str(row.get("stars", "")), row_number=source_row_number)
            grouped[text.casefold()].append((source_row_number, text, label))

    if input_rows != train_file["expected_rows"]:
        raise ValueError(f"{path}: expected {train_file['expected_rows']} rows, got {input_rows}")

    prepared: list[PreparedRow] = []
    duplicate_rows = 0
    ambiguous_rows_removed = 0
    ambiguous_text_groups = 0
    for entries in grouped.values():
        labels = {entry[2] for entry in entries}
        if len(labels) > 1:
            ambiguous_text_groups += 1
            ambiguous_rows_removed += len(entries)
            continue
        first_row, text, label = entries[0]
        duplicate_rows += len(entries) - 1
        prepared.append(PreparedRow(first_row, text, label))
    prepared.sort(key=lambda row: row.source_row_number)
    return prepared, {
        "input_rows": input_rows,
        "usable_rows": len(prepared),
        "missing_text_rows": missing_text_rows,
        "duplicate_rows": duplicate_rows,
        "ambiguous_text_groups": ambiguous_text_groups,
        "ambiguous_rows_removed": ambiguous_rows_removed,
        "sha256": sha256_file(path),
        "label_counts": dict(sorted(Counter(row.label for row in prepared).items())),
        "label_derivation": source["derived_label_rule"],
    }


def prepare(
    *,
    google_play_train_path: Path,
    smsa_data_dir: Path,
    smsa_split_manifest_path: Path,
    training_manifest_path: Path,
    output_dir: Path,
    igar_paths: list[Path] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    rows, source_report = load_google_play_rows(
        google_play_train_path.resolve(), training_manifest_path.resolve()
    )
    smsa_splits, smsa_manifest = load_smsa_splits(
        smsa_data_dir.resolve(), smsa_split_manifest_path.resolve()
    )
    smsa_keys = {
        normalize_text(row.text).casefold(): row.label
        for split_rows in smsa_splits.values()
        for row in split_rows
    }
    rows, smsa_overlap_removed = _remove_protected_overlaps(
        rows, smsa_keys, source_name="SmSA"
    )

    effective_igar_paths = [
        path.resolve()
        for path in (igar_paths if igar_paths is not None else _default_igar_guard_paths())
    ]
    igar_keys: set[str] = set()
    igar_details: list[dict[str, Any]] = []
    for path in effective_igar_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        keys = load_igar_text_keys(path)
        igar_keys.update(keys)
        igar_details.append(
            {"path": _display_path(path), "sha256": sha256_file(path), "text_keys": len(keys)}
        )
    rows, igar_overlap_removed = _remove_protected_overlaps(
        rows, {key: "" for key in igar_keys}, source_name="IGAR"
    )

    split_rows = stratified_split(rows, seed=seed)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_details: dict[str, Any] = {}
    for split in SPLITS:
        path = output_dir / f"google_play_review_{split}.tsv"
        _write_tsv(path, split_rows[split])
        split_details[split] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "rows": len(split_rows[split]),
            "label_counts": dict(sorted(Counter(row.label for row in split_rows[split]).items())),
        }

    result = {
        "dataset": "Indonesian Google Play Review",
        "source_key": SOURCE_KEY,
        "preprocessing_version": PREPROCESSING_VERSION,
        "source_manifest": _display_path(training_manifest_path),
        "source_manifest_sha256": sha256_file(training_manifest_path.resolve()),
        "source_file": _display_path(google_play_train_path),
        "source_sha256": source_report["sha256"],
        "split_strategy": "stratified_random_per_derived_label",
        "seed": seed,
        "split_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "training_policy": {
            "igar_forbidden": True,
            "igar_role": "sealed_external_test_only",
            "igar_labels_used": False,
            "igar_metrics_used": False,
            "igar_text_used_for_overlap_guard_only": True,
        },
        "source_report": source_report,
        "overlap_guard": {
            "smsa_rows_removed": smsa_overlap_removed,
            "igar_rows_removed": igar_overlap_removed,
            "igar_sources": igar_details,
        },
        "splits": split_details,
        "smsa_split_manifest_sha256": sha256_file(smsa_split_manifest_path.resolve()),
        "smsa_rows": {split: len(rows) for split, rows in smsa_splits.items()},
        "smsa_leakage_check_passed": True,
    }
    _atomic_write_text(output_dir / "google_play_review_split_manifest.json", json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--google-play-train-path",
        type=Path,
        default=ROOT / "data/raw/google_play_review/train.csv",
    )
    parser.add_argument("--smsa-data-dir", type=Path, default=ROOT / "data/raw/smsa")
    parser.add_argument(
        "--smsa-split-manifest",
        type=Path,
        default=ROOT / "artifacts/week2/smsa_split_manifest.json",
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=ROOT / "data/manifests/week6_training_sources.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/week6/google_play_review",
    )
    parser.add_argument("--igar-test-path", type=Path, action="append")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    try:
        args = build_parser().parse_args()
        payload = prepare(
            google_play_train_path=args.google_play_train_path,
            smsa_data_dir=args.smsa_data_dir,
            smsa_split_manifest_path=args.smsa_split_manifest,
            training_manifest_path=args.training_manifest,
            output_dir=args.output_dir,
            igar_paths=args.igar_test_path,
            seed=args.seed,
        )
        print(
            "Prepared Google Play Review splits: "
            f"train={payload['splits']['train']['rows']}, "
            f"validation={payload['splits']['validation']['rows']}, "
            f"test={payload['splits']['test']['rows']}; "
            f"IGAR overlaps removed={payload['overlap_guard']['igar_rows_removed']}"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
