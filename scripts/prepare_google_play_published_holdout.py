#!/usr/bin/env python3
"""Prepare the published Google Play validation file as a confirmation holdout.

The source labels are deliberately ignored for the target: stars determine the
three canonical sentiment labels.  This script has no external-test source
dependency; protected overlap inputs are only the existing non-external-test
Google Play and SmSA processed splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.preprocessing import (  # noqa: E402
    CANONICAL_LABELS,
    PREPROCESSING_VERSION,
    normalize_label,
    normalize_text,
)


SOURCE_KEY = "google_play_review"
SCHEMA = ["text", "label", "stars"]
MISSING_TEXT_VALUES = {"", "na", "n/a", "nan", "none", "null"}
DEFAULT_PROTECTED_PATHS = (
    ROOT / "data/processed/week6/google_play_review/google_play_review_train.tsv",
    ROOT / "data/processed/week6/google_play_review/google_play_review_validation.tsv",
    ROOT / "data/processed/week6/google_play_review/google_play_review_test.tsv",
    ROOT / "data/raw/smsa/train_preprocess.tsv",
    ROOT / "data/raw/smsa/valid_preprocess.tsv",
    ROOT / "data/raw/smsa/test_preprocess.tsv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def assert_not_igar_input_path(path: Path) -> None:
    """Reject paths that could identify the sealed external-test dataset."""

    resolved = path.resolve()
    parts = {part.casefold() for part in resolved.parts}
    if "igar" in parts or "rating_labeled" in resolved.name.casefold():
        raise ValueError("sealed external-test input paths are not allowed")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _derived_label(raw_label: str, stars_value: str, *, row_number: int) -> str:
    """Derive the canonical target and verify the published binary label."""

    try:
        stars = int(stars_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"source row {row_number} has a non-integer stars value") from error
    if stars not in {1, 2, 3, 4, 5}:
        raise ValueError(f"source row {row_number} has stars outside 1..5")
    published = normalize_text(raw_label).casefold()
    expected = "pos" if stars in {4, 5} else "neg"
    if published != expected:
        raise ValueError(
            f"source row {row_number} has inconsistent published label/rating: "
            f"{published!r} with {stars} stars"
        )
    return "negative" if stars <= 2 else "neutral" if stars == 3 else "positive"


def _validation_file(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("sources", {}).get(SOURCE_KEY)
    if not isinstance(source, dict):
        raise ValueError("training manifest is missing the Google Play source")
    if source.get("schema") != SCHEMA:
        raise ValueError("Google Play source schema is not pinned as expected")
    entry = next(
        (item for item in source.get("files", []) if item.get("name") == "validation.csv"),
        None,
    )
    if not isinstance(entry, dict):
        raise ValueError("Google Play validation file is missing from the manifest")
    return manifest, entry


def load_published_rows(path: Path, training_manifest_path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Validate and load the pinned validation CSV, filtering source defects."""

    path = path.resolve()
    training_manifest_path = training_manifest_path.resolve()
    assert_not_igar_input_path(path)
    assert_not_igar_input_path(training_manifest_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    _, entry = _validation_file(training_manifest_path)
    checksum = sha256_file(path)
    if checksum != entry.get("expected_sha256"):
        raise ValueError(f"{path}: SHA-256 does not match the validation manifest")

    grouped: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    input_rows = 0
    missing_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SCHEMA:
            raise ValueError(f"{path}: schema mismatch: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=1):
            input_rows += 1
            text = normalize_text(row.get("text"))
            if text.casefold() in MISSING_TEXT_VALUES:
                missing_rows += 1
                continue
            label = _derived_label(str(row.get("label", "")), str(row.get("stars", "")), row_number=row_number)
            grouped[text.casefold()].append((row_number, text, label))

    if input_rows != entry.get("expected_rows"):
        raise ValueError(f"{path}: expected {entry.get('expected_rows')} rows, got {input_rows}")

    rows: list[dict[str, str]] = []
    duplicate_rows = 0
    conflicting_groups = 0
    conflicting_rows = 0
    for entries in grouped.values():
        labels = {item[2] for item in entries}
        if len(labels) > 1:
            conflicting_groups += 1
            conflicting_rows += len(entries)
            continue
        duplicate_rows += len(entries) - 1
        _, text, label = entries[0]
        rows.append({"text": text, "label": label})
    rows.sort(key=lambda row: next(item[0] for item in grouped[row["text"].casefold()] if item[1] == row["text"]))
    return rows, {
        "input_rows": input_rows,
        "missing_text_rows": missing_rows,
        "duplicate_rows": duplicate_rows,
        "conflicting_text_groups": conflicting_groups,
        "conflicting_rows_removed": conflicting_rows,
        "usable_rows": len(rows),
        "sha256": checksum,
    }


def load_protected_text_keys(paths: Iterable[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    """Read only text/label pairs from non-external-test protected TSVs."""

    keys: set[str] = set()
    details: list[dict[str, Any]] = []
    for candidate in paths:
        path = candidate.resolve()
        assert_not_igar_input_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, values in enumerate(csv.reader(handle, delimiter="\t"), start=1):
                if not values or all(not value.strip() for value in values):
                    continue
                if line_number == 1 and values == ["text", "label"]:
                    continue
                if len(values) != 2:
                    raise ValueError(f"{path}: line {line_number} must contain text and label")
                text = normalize_text(values[0])
                if text.casefold() in MISSING_TEXT_VALUES:
                    continue
                normalize_label(values[1])
                keys.add(text.casefold())
                count += 1
        details.append({"path": _display_path(path), "sha256": sha256_file(path), "rows": count})
    return keys, details


def _write_tsv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    content = "".join(
        f"{row['text'].replace(chr(9), ' ')}\t{row['label']}\n" for row in rows
    )
    _atomic_write_text(path, content)


def prepare(*, validation_path: Path, training_manifest_path: Path, output_tsv: Path,
            output_manifest: Path, protected_paths: Iterable[Path] | None = None,
            exclude_overlaps: bool = True) -> dict[str, Any]:
    rows, source_report = load_published_rows(validation_path, training_manifest_path)
    protected = list(DEFAULT_PROTECTED_PATHS if protected_paths is None else protected_paths)
    overlap_count = 0
    protected_details: list[dict[str, Any]] = []
    if exclude_overlaps:
        keys, protected_details = load_protected_text_keys(protected)
        kept = []
        for row in rows:
            if row["text"].casefold() in keys:
                overlap_count += 1
            else:
                kept.append(row)
        rows = kept

    _write_tsv(output_tsv.resolve(), rows)
    payload = {
        "dataset": "Indonesian Google Play Review published validation",
        "source_file": _display_path(validation_path),
        "source_manifest": _display_path(training_manifest_path),
        "source_sha256": source_report["sha256"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "label_derivation": {"negative_stars": [1, 2], "neutral_stars": [3], "positive_stars": [4, 5]},
        "policy": {"confirmation_only": True, "for_training": False, "for_selection": False},
        "source_report": source_report,
        "exclusion": {"enabled": exclude_overlaps, "protected_overlap_rows_removed": overlap_count,
                       "protected_inputs": protected_details},
        "rows": len(rows),
        "class_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "output": {"file": _display_path(output_tsv), "sha256": sha256_file(output_tsv.resolve())},
    }
    _atomic_write_text(output_manifest.resolve(), json.dumps(payload, indent=2) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-path", type=Path, default=ROOT / "data/raw/google_play_review/validation.csv")
    parser.add_argument("--training-manifest", type=Path, default=ROOT / "data/manifests/week6_training_sources.json")
    parser.add_argument("--output-tsv", type=Path, default=ROOT / "data/processed/week6/google_play_published_holdout.tsv")
    parser.add_argument("--output-manifest", type=Path, default=ROOT / "data/processed/week6/google_play_published_holdout_manifest.json")
    parser.add_argument("--protected-path", type=Path, action="append", dest="protected_paths")
    parser.add_argument("--no-overlap-exclusion", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        result = prepare(validation_path=args.validation_path, training_manifest_path=args.training_manifest,
                         output_tsv=args.output_tsv, output_manifest=args.output_manifest,
                         protected_paths=args.protected_paths, exclude_overlaps=not args.no_overlap_exclusion)
        print(f"Prepared confirmation holdout: rows={result['rows']}")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
