#!/usr/bin/env python3
"""Build a non-IGAR tuning source with Google reviews and SmSA neutral rows.

The Google Play source is the app-review domain proxy.  A deterministic sample
of neutral SmSA training rows is added to improve neutral coverage without
reading any external-test payload.  Google validation/test and SmSA
validation/test remain protected from the mixed training view.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.ai.preprocessing import PREPROCESSING_VERSION, PreparedRow, normalize_text  # noqa: E402
from train_indobert import (  # noqa: E402
    assert_disjoint_splits,
    assert_not_igar_training_path,
    load_additional_splits,
    load_splits,
    sha256_file,
)


def _write_tsv(path: Path, rows: Iterable[PreparedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows((row.text, row.label) for row in rows)


def mix_training_rows(
    google_rows: list[PreparedRow],
    smsa_rows: list[PreparedRow],
    *,
    smsa_neutral_limit: int | None,
    seed: int,
) -> tuple[list[PreparedRow], dict[str, int]]:
    """Add a deterministic SmSA-neutral sample to Google training rows."""

    if smsa_neutral_limit is not None and smsa_neutral_limit < 1:
        raise ValueError("smsa_neutral_limit must be positive when provided")
    neutral_rows = [row for row in smsa_rows if row.label == "neutral"]
    if smsa_neutral_limit is not None:
        if smsa_neutral_limit > len(neutral_rows):
            raise ValueError(
                f"smsa_neutral_limit {smsa_neutral_limit} exceeds available neutral rows {len(neutral_rows)}"
            )
        neutral_rows = random.Random(seed).sample(neutral_rows, smsa_neutral_limit)
        neutral_rows.sort(key=lambda row: row.source_row_number)

    accepted: list[PreparedRow] = []
    seen: dict[str, str] = {}
    duplicate_rows = 0
    conflicting_rows = 0
    for row in [*google_rows, *neutral_rows]:
        key = normalize_text(row.text).casefold()
        if not key:
            continue
        previous_label = seen.get(key)
        if previous_label is not None:
            duplicate_rows += 1
            if previous_label != row.label:
                conflicting_rows += 1
            continue
        seen[key] = row.label
        accepted.append(row)
    if conflicting_rows:
        raise ValueError("Google and SmSA neutral training rows contain conflicting labels")
    return accepted, {
        "google_train_rows": len(google_rows),
        "smsa_neutral_available": len([row for row in smsa_rows if row.label == "neutral"]),
        "smsa_neutral_selected": len(neutral_rows),
        "accepted_train_rows": len(accepted),
        "duplicate_rows_removed": duplicate_rows,
        "conflicting_rows": conflicting_rows,
    }


def prepare(
    *,
    google_manifest_path: Path,
    smsa_data_dir: Path,
    smsa_split_manifest_path: Path,
    output_dir: Path,
    smsa_neutral_limit: int | None = 500,
    seed: int = 42,
) -> dict[str, Any]:
    assert_not_igar_training_path(google_manifest_path)
    assert_not_igar_training_path(smsa_data_dir)
    assert_not_igar_training_path(smsa_split_manifest_path)
    google_splits, google_manifest = load_additional_splits(google_manifest_path.resolve())
    smsa_splits, smsa_manifest = load_splits(
        smsa_data_dir.resolve(), smsa_split_manifest_path.resolve()
    )
    mixed_train, mix_report = mix_training_rows(
        google_splits["train"],
        smsa_splits["train"],
        smsa_neutral_limit=smsa_neutral_limit,
        seed=seed,
    )
    split_rows = {
        "train": mixed_train,
        "validation": google_splits["validation"],
        "test": google_splits["test"],
    }
    assert_disjoint_splits(split_rows)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_details: dict[str, Any] = {}
    for split, rows in split_rows.items():
        path = output_dir / f"google_play_smsa_neutral_{split}.tsv"
        _write_tsv(path, rows)
        split_details[split] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "rows": len(rows),
            "label_counts": dict(sorted(Counter(row.label for row in rows).items())),
        }

    payload: dict[str, Any] = {
        "dataset": "Google Play Review + SmSA neutral augmentation",
        "source_key": "google_play_smsa_neutral_mix",
        "preprocessing_version": PREPROCESSING_VERSION,
        "seed": seed,
        "smsa_neutral_limit": smsa_neutral_limit,
        "training_policy": {
            "igar_forbidden": True,
            "igar_role": "sealed_external_test_only",
            "igar_labels_used": False,
            "igar_metrics_used": False,
            "igar_read": False,
        },
        "sources": {
            "google_play_review": {
                "role": "domain_training",
                "manifest": str(google_manifest_path.resolve().relative_to(ROOT)),
                "manifest_sha256": sha256_file(google_manifest_path.resolve()),
                "train_rows": len(google_splits["train"]),
                "license": "CC BY 4.0",
            },
            "smsa": {
                "role": "neutral_augmentation_training",
                "manifest": str(smsa_split_manifest_path.resolve().relative_to(ROOT)),
                "manifest_sha256": sha256_file(smsa_split_manifest_path.resolve()),
                "train_rows": len(smsa_splits["train"]),
                "license": "CC BY-SA 4.0",
            },
        },
        "mix_report": mix_report,
        "splits": split_details,
        "holdouts": {
            "selection": "Google Play validation and SmSA validation only",
            "google_play_test_used": False,
            "smsa_test_used": False,
        },
    }
    (output_dir / "google_play_smsa_neutral_mix_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--google-manifest",
        type=Path,
        default=ROOT / "data/processed/week6/google_play_review/google_play_review_split_manifest.json",
    )
    parser.add_argument("--smsa-data-dir", type=Path, default=ROOT / "data/raw/smsa")
    parser.add_argument(
        "--smsa-split-manifest",
        type=Path,
        default=ROOT / "artifacts/week2/smsa_split_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/sentiment_tuning/google_play_smsa_neutral",
    )
    parser.add_argument("--smsa-neutral-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    try:
        args = build_parser().parse_args()
        result = prepare(
            google_manifest_path=args.google_manifest,
            smsa_data_dir=args.smsa_data_dir,
            smsa_split_manifest_path=args.smsa_split_manifest,
            output_dir=args.output_dir,
            smsa_neutral_limit=args.smsa_neutral_limit,
            seed=args.seed,
        )
        print(
            "Prepared sentiment tuning mix: "
            f"train={result['splits']['train']['rows']}, "
            f"neutral_added={result['mix_report']['smsa_neutral_selected']}"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
