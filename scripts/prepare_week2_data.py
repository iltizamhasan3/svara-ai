"""Prepare and profile the official SmSA splits for Week 2 experiments.

This script intentionally uses only the Python standard library plus the
repository's lightweight preprocessing module. Heavy model dependencies stay
optional and are only needed by the notebook training cell.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.preprocessing import (  # noqa: E402
    CANONICAL_LABELS,
    PREPROCESSING_VERSION,
    PreparedRow,
    PreparationReport,
    normalize_text,
    prepare_labeled_rows,
)


SMSA_SPLITS = {
    "train": "train_preprocess.tsv",
    "validation": "valid_preprocess.tsv",
    "test": "test_preprocess.tsv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_smsa_rows(path: Path) -> list[dict[str, str]]:
    """Read the headerless SmSA TSV format and fail on malformed records."""

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, values in enumerate(reader, start=1):
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) != 2:
                raise ValueError(
                    f"{path}: line {line_number} must contain text and label separated by one tab"
                )
            rows.append({"sentence": values[0], "label": values[1]})
    return rows


def length_summary(rows: list[PreparedRow]) -> dict[str, float | int]:
    character_lengths = [len(row.text) for row in rows]
    token_lengths = [len(row.text.split()) for row in rows]
    if not character_lengths:
        return {
            "characters_min": 0,
            "characters_mean": 0,
            "characters_median": 0,
            "characters_max": 0,
            "tokens_min": 0,
            "tokens_mean": 0,
            "tokens_median": 0,
            "tokens_max": 0,
        }

    return {
        "characters_min": min(character_lengths),
        "characters_mean": round(statistics.fmean(character_lengths), 2),
        "characters_median": statistics.median(character_lengths),
        "characters_max": max(character_lengths),
        "tokens_min": min(token_lengths),
        "tokens_mean": round(statistics.fmean(token_lengths), 2),
        "tokens_median": statistics.median(token_lengths),
        "tokens_max": max(token_lengths),
    }


def split_summary(
    split: str,
    path: Path,
    rows: list[dict[str, str]],
) -> tuple[dict[str, Any], list[PreparedRow], PreparationReport]:
    prepared, report = prepare_labeled_rows(
        rows,
        text_column="sentence",
        label_column="label",
    )
    counts = Counter(row.label for row in prepared)
    unknown_labels = sorted(set(counts) - set(CANONICAL_LABELS))
    if unknown_labels:
        raise ValueError(f"{split}: unsupported labels after normalization: {unknown_labels}")

    total = len(prepared)
    distribution = {
        label: {
            "count": counts.get(label, 0),
            "percentage": round((counts.get(label, 0) / total) * 100, 4) if total else 0,
        }
        for label in CANONICAL_LABELS
    }
    summary = {
        "file": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "input_rows": report.input_rows,
        "usable_rows": report.output_rows,
        "missing_text_rows": report.missing_text_rows,
        "duplicate_rows_removed": report.duplicate_rows,
        "label_distribution": distribution,
        "text_length": length_summary(prepared),
    }
    return summary, prepared, report


def rebuild_summary(
    summary: dict[str, Any],
    rows: list[PreparedRow],
    *,
    cross_split_duplicate_rows_removed: int = 0,
) -> dict[str, Any]:
    """Recalculate metrics after a deterministic leakage-cleaning pass."""

    counts = Counter(row.label for row in rows)
    total = len(rows)
    return {
        **summary,
        "usable_rows": total,
        "cross_split_duplicate_rows_removed": cross_split_duplicate_rows_removed,
        "label_distribution": {
            label: {
                "count": counts.get(label, 0),
                "percentage": round((counts.get(label, 0) / total) * 100, 4) if total else 0,
            }
            for label in CANONICAL_LABELS
        },
        "text_length": length_summary(rows),
    }


def normalized_text_set(rows: list[dict[str, str]]) -> set[str]:
    return {
        normalized
        for row in rows
        if (normalized := normalize_text(row.get("sentence")).casefold())
        not in {"", "na", "n/a", "nan", "none", "null"}
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_distribution_csv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "label", "count", "percentage"],
            lineterminator="\n",
        )
        writer.writeheader()
        for split, summary in summaries.items():
            for label in CANONICAL_LABELS:
                values = summary["label_distribution"][label]
                writer.writerow(
                    {
                        "split": split,
                        "label": label,
                        "count": values["count"],
                        "percentage": values["percentage"],
                    }
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smsa-dir",
        type=Path,
        default=ROOT / "data/raw/smsa",
        help="directory containing the official SmSA TSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/week2",
        help="directory for EDA JSON/CSV artifacts",
    )
    parser.add_argument(
        "--fail-on-cross-split-overlap",
        action="store_true",
        help="fail instead of removing overlapping training rows from the official validation/test sets",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    smsa_dir = args.smsa_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    raw_rows_by_split: dict[str, list[dict[str, str]]] = {}
    prepared_rows_by_split: dict[str, list[PreparedRow]] = {}
    reports: dict[str, PreparationReport] = {}

    for split, filename in SMSA_SPLITS.items():
        path = smsa_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run scripts/download_week2_datasets.py first"
            )
        raw_rows = read_smsa_rows(path)
        summary, prepared, report = split_summary(split, path, raw_rows)
        summaries[split] = summary
        raw_rows_by_split[split] = raw_rows
        prepared_rows_by_split[split] = prepared
        reports[split] = report

    overlaps: dict[str, int] = {}
    text_sets = {split: normalized_text_set(rows) for split, rows in raw_rows_by_split.items()}
    split_names = list(SMSA_SPLITS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlaps[f"{left}__{right}"] = len(text_sets[left] & text_sets[right])
    has_overlap = any(overlaps.values())
    protected_texts = text_sets["validation"] | text_sets["test"]
    leakage_safe_train = [
        row
        for row in prepared_rows_by_split["train"]
        if normalize_text(row.text).casefold() not in protected_texts
    ]
    cross_split_train_rows_removed = len(prepared_rows_by_split["train"]) - len(leakage_safe_train)
    leakage_safe_summaries = {
        **summaries,
        "train": rebuild_summary(
            summaries["train"],
            leakage_safe_train,
            cross_split_duplicate_rows_removed=cross_split_train_rows_removed,
        ),
    }
    leakage_safe_text_sets = {
        "train": {normalize_text(row.text).casefold() for row in leakage_safe_train},
        "validation": text_sets["validation"],
        "test": text_sets["test"],
    }
    safe_overlaps = {
        f"{left}__{right}": len(leakage_safe_text_sets[left] & leakage_safe_text_sets[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    }
    safe_has_overlap = any(safe_overlaps.values())

    payload = {
        "dataset": "SmSA",
        "preprocessing_version": PREPROCESSING_VERSION,
        "format": "headerless TSV: sentence<TAB>label",
        "official_splits": summaries,
        "leakage_safe_splits": leakage_safe_summaries,
        "cross_split_duplicate_text": overlaps,
        "cross_split_train_rows_removed": cross_split_train_rows_removed,
        "leakage_safe_cross_split_duplicate_text": safe_overlaps,
        "leakage_check_passed": not safe_has_overlap,
        "notes": [
            "Official train/validation/test boundaries are preserved; overlapping training texts are removed from the prepared training view.",
            "Text is normalized for whitespace and Unicode only; punctuation, emoji, and slang are retained.",
            "Rows with missing text are excluded; same-label duplicates are counted and removed within each split.",
            "Conflicting normalized labels fail closed during preparation.",
        ],
    }
    write_json(output_dir / "smsa_eda.json", payload)
    leakage_safe_rows_by_split = {
        "train": leakage_safe_train,
        "validation": prepared_rows_by_split["validation"],
        "test": prepared_rows_by_split["test"],
    }
    write_json(
        output_dir / "smsa_split_manifest.json",
        {
            "dataset": "SmSA",
            "preprocessing_version": PREPROCESSING_VERSION,
            "splits": {
                split: {
                    "file": leakage_safe_summaries[split]["file"],
                    "sha256": leakage_safe_summaries[split]["sha256"],
                    "input_rows": leakage_safe_summaries[split]["input_rows"],
                    "usable_rows": leakage_safe_summaries[split]["usable_rows"],
                    "cross_split_duplicate_rows_removed": leakage_safe_summaries[split].get(
                        "cross_split_duplicate_rows_removed", 0
                    ),
                    "source_row_numbers": {
                        "first": prepared_rows[0].source_row_number if prepared_rows else None,
                        "last": prepared_rows[-1].source_row_number if prepared_rows else None,
                    },
                }
                for split, prepared_rows in leakage_safe_rows_by_split.items()
            },
            "cross_split_duplicate_text": overlaps,
            "leakage_safe_cross_split_duplicate_text": safe_overlaps,
            "leakage_check_passed": not safe_has_overlap,
        },
    )
    write_distribution_csv(output_dir / "smsa_label_distribution.csv", leakage_safe_summaries)

    if safe_has_overlap or (has_overlap and args.fail_on_cross_split_overlap):
        raise ValueError(
            "normalized text overlaps remain in the prepared SmSA splits; inspect "
            f"{output_dir / 'smsa_eda.json'}"
        )

    total_input = sum(report.input_rows for report in reports.values())
    total_usable = sum(summary["usable_rows"] for summary in leakage_safe_summaries.values())
    print(
        f"SmSA EDA written to {output_dir} "
        f"({total_usable}/{total_input} usable rows; "
        f"removed_cross_split_train_rows={cross_split_train_rows_removed}; "
        f"leakage_check_passed={not safe_has_overlap})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
