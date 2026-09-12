#!/usr/bin/env python3
"""Prepare ID-SMSA splits without allowing IGAR into training data.

The prepared ID-SMSA validation and test splits are holdouts for development
reporting. IGAR text is read only to remove exact overlaps from the additional
training source; IGAR labels, scores, and predictions are never consumed by
this preparation step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.preprocessing import (  # noqa: E402
    PREPROCESSING_VERSION,
    PreparedRow,
    normalize_text,
    prepare_labeled_rows,
)


SPLITS = ("train", "validation", "test")
IGAR_SCHEMA = ["", "app", "content", "translation", "score", "at", "appVersion", "labelScoreBase"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def assert_not_igar_training_path(path: Path) -> None:
    """Fail closed if a purported training input looks like an IGAR payload."""

    lowered_parts = {part.casefold() for part in path.resolve().parts}
    lowered_name = path.name.casefold()
    if "igar" in lowered_parts or "rating_labeled" in lowered_name:
        raise ValueError("IGAR is sealed external-test data and cannot be a training input")


def _read_smsa(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, values in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) != 2:
                raise ValueError(f"{path}: line {line_number} must have two TSV fields")
            rows.append({"sentence": values[0], "label": values[1]})
    return rows


def load_smsa_splits(data_dir: Path, split_manifest_path: Path) -> tuple[dict[str, list[PreparedRow]], dict[str, Any]]:
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "SmSA":
        raise ValueError("SmSA split manifest must describe SmSA")
    if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("SmSA preprocessing version does not match the code")

    prepared: dict[str, list[PreparedRow]] = {}
    for split in SPLITS:
        details = manifest["splits"][split]
        candidate = Path(str(details["file"]))
        path = candidate if candidate.is_absolute() else data_dir / candidate.name
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_sha256 = details.get("sha256")
        if sha256_file(path).casefold() != str(expected_sha256).casefold():
            raise ValueError(f"{path}: SHA-256 does not match SmSA split manifest")
        rows, _ = prepare_labeled_rows(
            _read_smsa(path),
            text_column="sentence",
            label_column="label",
        )
        prepared[split] = rows

    protected = {
        normalize_text(row.text).casefold()
        for split in ("validation", "test")
        for row in prepared[split]
    }
    prepared["train"] = [
        row for row in prepared["train"] if normalize_text(row.text).casefold() not in protected
    ]
    _assert_disjoint(prepared)
    return prepared, manifest


def _assert_disjoint(splits: Mapping[str, Iterable[PreparedRow]]) -> None:
    seen: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            key = normalize_text(row.text).casefold()
            previous = seen.get(key)
            if previous is not None and previous != split:
                raise ValueError(f"normalized text overlaps {previous} and {split}: {row.text[:80]!r}")
            seen[key] = split


def load_idsmsa(path: Path, training_manifest_path: Path) -> tuple[list[PreparedRow], dict[str, Any]]:
    """Load and validate the pinned ID-SMSA source for preparation."""

    assert_not_igar_training_path(path)
    manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("training_policy", {}).get("igar_forbidden") is not True:
        raise ValueError("training manifest must explicitly forbid IGAR")
    source = manifest.get("sources", {}).get("idsmsa")
    if not isinstance(source, dict) or source.get("role") != "additional_training":
        raise ValueError("training manifest must identify ID-SMSA as additional_training")
    if sha256_file(path).casefold() != str(source["expected_sha256"]).casefold():
        raise ValueError(f"{path}: SHA-256 does not match ID-SMSA training manifest")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != source["schema"]:
            raise ValueError(f"{path}: ID-SMSA schema mismatch: {reader.fieldnames}")
        raw_rows = list(reader)
    if len(raw_rows) != source["expected_rows"]:
        raise ValueError(f"{path}: expected {source['expected_rows']} rows, got {len(raw_rows)}")
    prepared, report = prepare_labeled_rows(
        raw_rows,
        text_column="Sentence",
        label_column="Sentiment",
    )
    return prepared, {
        "input_rows": report.input_rows,
        "usable_rows": report.output_rows,
        "missing_text_rows": report.missing_text_rows,
        "duplicate_rows": report.duplicate_rows,
        "sha256": sha256_file(path),
    }


def load_igar_text_keys(path: Path) -> set[str]:
    """Read only IGAR text for a leakage guard; labels are deliberately ignored."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != IGAR_SCHEMA:
            raise ValueError(f"{path}: IGAR schema mismatch: {reader.fieldnames}")
        return {
            normalize_text(row.get("content")).casefold()
            for row in reader
            if normalize_text(row.get("content"))
        }


def _default_igar_guard_paths() -> list[Path]:
    full_path = ROOT / "data/raw/igar/Rating_labeled.csv"
    sample_path = ROOT / "data/samples/igar/Rating_labeled_sample.csv"
    if full_path.is_file():
        return [full_path]
    if sample_path.is_file():
        return [sample_path]
    return []


def _remove_protected_overlaps(
    rows: Iterable[PreparedRow],
    protected_keys: Mapping[str, str],
    *,
    source_name: str,
) -> tuple[list[PreparedRow], int]:
    kept: list[PreparedRow] = []
    removed = 0
    for row in rows:
        key = normalize_text(row.text).casefold()
        if key in protected_keys:
            removed += 1
            if source_name == "SmSA" and protected_keys[key] != row.label:
                raise ValueError(f"{source_name} has a conflicting label for {row.text[:80]!r}")
            continue
        kept.append(row)
    return kept, removed


def stratified_split(rows: Iterable[PreparedRow], *, seed: int) -> dict[str, list[PreparedRow]]:
    """Create a deterministic 80/10/10 split independently per label."""

    by_label: dict[str, list[PreparedRow]] = {}
    for row in rows:
        by_label.setdefault(row.label, []).append(row)
    randomizer = random.Random(seed)
    split_rows = {split: [] for split in SPLITS}
    for label in sorted(by_label):
        group = sorted(
            by_label[label],
            key=lambda row: (normalize_text(row.text).casefold(), row.source_row_number),
        )
        randomizer.shuffle(group)
        train_count = max(1, int(len(group) * 0.8))
        validation_count = max(1, int(len(group) * 0.1))
        if train_count + validation_count >= len(group):
            validation_count = 1
            train_count = len(group) - 2
        split_rows["train"].extend(group[:train_count])
        split_rows["validation"].extend(group[train_count : train_count + validation_count])
        split_rows["test"].extend(group[train_count + validation_count :])
    for split in SPLITS:
        split_rows[split].sort(key=lambda row: (row.label, normalize_text(row.text).casefold()))
    _assert_disjoint(split_rows)
    return split_rows


def _write_tsv(path: Path, rows: Iterable[PreparedRow]) -> None:
    output = "".join(f"{row.text}\t{row.label}\n" for row in rows)
    _atomic_write_text(path, output)


def prepare(
    *,
    idsmsa_path: Path,
    smsa_data_dir: Path,
    smsa_split_manifest_path: Path,
    training_manifest_path: Path,
    output_dir: Path,
    igar_paths: list[Path] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    idsmsa_rows, idsmsa_report = load_idsmsa(idsmsa_path.resolve(), training_manifest_path.resolve())
    smsa_splits, smsa_manifest = load_smsa_splits(smsa_data_dir.resolve(), smsa_split_manifest_path.resolve())
    smsa_keys = {
        normalize_text(row.text).casefold(): row.label
        for rows in smsa_splits.values()
        for row in rows
    }
    idsmsa_rows, smsa_overlap_removed = _remove_protected_overlaps(
        idsmsa_rows,
        smsa_keys,
        source_name="SmSA",
    )

    effective_igar_paths = [path.resolve() for path in (igar_paths if igar_paths is not None else _default_igar_guard_paths())]
    igar_keys: set[str] = set()
    igar_details: list[dict[str, Any]] = []
    for path in effective_igar_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        keys = load_igar_text_keys(path)
        igar_keys.update(keys)
        igar_details.append({"path": _display_path(path), "sha256": sha256_file(path), "text_keys": len(keys)})
    idsmsa_rows, igar_overlap_removed = _remove_protected_overlaps(
        idsmsa_rows,
        {key: "" for key in igar_keys},
        source_name="IGAR",
    )

    split_rows = stratified_split(idsmsa_rows, seed=seed)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_details: dict[str, Any] = {}
    for split in SPLITS:
        path = output_dir / f"idsmsa_{split}.tsv"
        _write_tsv(path, split_rows[split])
        split_details[split] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "rows": len(split_rows[split]),
            "label_counts": dict(sorted(Counter(row.label for row in split_rows[split]).items())),
        }

    result = {
        "dataset": "ID-SMSA",
        "preprocessing_version": PREPROCESSING_VERSION,
        "source_manifest": _display_path(training_manifest_path),
        "source_manifest_sha256": sha256_file(training_manifest_path.resolve()),
        "source_file": _display_path(idsmsa_path),
        "source_sha256": idsmsa_report["sha256"],
        "split_strategy": "stratified_random_per_label",
        "seed": seed,
        "split_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "training_policy": {
            "igar_forbidden": True,
            "igar_role": "sealed_external_test_only",
            "igar_labels_used": False,
            "igar_metrics_used": False,
        },
        "source_report": idsmsa_report,
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
    _atomic_write_text(output_dir / "idsmsa_split_manifest.json", json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idsmsa-path", type=Path, default=ROOT / "data/raw/idsmsa/IDSMSA.csv")
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/week6/idsmsa")
    parser.add_argument(
        "--igar-test-path",
        type=Path,
        action="append",
        help="optional IGAR test file(s) used only for text-overlap exclusion",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    try:
        args = build_parser().parse_args()
        payload = prepare(
            idsmsa_path=args.idsmsa_path,
            smsa_data_dir=args.smsa_data_dir,
            smsa_split_manifest_path=args.smsa_split_manifest,
            training_manifest_path=args.training_manifest,
            output_dir=args.output_dir,
            igar_paths=args.igar_test_path,
            seed=args.seed,
        )
        print(
            "Prepared ID-SMSA splits: "
            f"train={payload['splits']['train']['rows']}, "
            f"validation={payload['splits']['validation']['rows']}, "
            f"test={payload['splits']['test']['rows']}; "
            f"IGAR overlaps removed={payload['overlap_guard']['igar_rows_removed']}"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
