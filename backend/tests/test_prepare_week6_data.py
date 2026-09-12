import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from app.ai.preprocessing import PreparedRow


ROOT = Path(__file__).parents[2]


def load_preparer():
    spec = importlib.util.spec_from_file_location(
        "week6_data_preparer", ROOT / "scripts/prepare_week6_data.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_igar_path_is_rejected_as_training_input():
    preparer = load_preparer()

    with pytest.raises(ValueError, match="sealed external-test"):
        preparer.assert_not_igar_training_path(ROOT / "data/raw/igar/Rating_labeled.csv")


def test_stratified_split_is_deterministic_and_disjoint():
    preparer = load_preparer()
    rows = [
        PreparedRow(index, f"{label} example {index}", label)
        for label in ("positive", "neutral", "negative")
        for index in range(1, 11)
    ]

    first = preparer.stratified_split(rows, seed=42)
    second = preparer.stratified_split(rows, seed=42)

    assert first == second
    assert {split: len(values) for split, values in first.items()} == {
        "train": 24,
        "validation": 3,
        "test": 3,
    }
    assert not (
        {row.text for row in first["train"]}
        & {row.text for row in first["validation"]}
    )


def test_prepare_removes_cross_source_and_igar_text_overlaps(tmp_path):
    preparer = load_preparer()
    smsa_dir = tmp_path / "smsa"
    smsa_dir.mkdir()
    smsa_contents = {
        "train_preprocess.tsv": "SmSA training	positive\n",
        "valid_preprocess.tsv": "SmSA validation	neutral\n",
        "test_preprocess.tsv": "SmSA test	negative\n",
    }
    split_details = {}
    for filename, content in smsa_contents.items():
        path = smsa_dir / filename
        path.write_text(content, encoding="utf-8")
        split = "validation" if filename.startswith("valid") else filename.split("_")[0]
        split_details[split] = {
            "file": f"data/raw/smsa/{filename}",
            "sha256": preparer.sha256_file(path),
        }
    smsa_manifest = tmp_path / "smsa-manifest.json"
    smsa_manifest.write_text(
        json.dumps(
            {
                "dataset": "SmSA",
                "preprocessing_version": preparer.PREPROCESSING_VERSION,
                "splits": split_details,
            }
        ),
        encoding="utf-8",
    )

    idsmsa_path = tmp_path / "idsmsa.csv"
    fieldnames = [
        "Tweet Date",
        "Sentence",
        "Quote Count",
        "Reply Count",
        "Retweet Count",
        "Favorite Count",
        "Sentiment",
        "English Translation",
    ]
    rows = [
        ["2025-01-01", "SmSA validation", "0", "0", "0", "0", "Neutral", ""],
        ["2025-01-02", "IGAR overlap", "0", "0", "0", "0", "Negative", ""],
        ["2025-01-03", "Independent positive", "0", "0", "0", "0", "Positive", ""],
        ["2025-01-04", "Independent neutral", "0", "0", "0", "0", "Neutral", ""],
        ["2025-01-05", "Independent negative", "0", "0", "0", "0", "Negative", ""],
        ["2025-01-06", "Independent positive two", "0", "0", "0", "0", "Positive", ""],
    ]
    with idsmsa_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fieldnames)
        writer.writerows(rows)
    training_manifest = tmp_path / "training-manifest.json"
    training_manifest.write_text(
        json.dumps(
            {
                "training_policy": {"igar_forbidden": True},
                "sources": {
                    "idsmsa": {
                        "role": "additional_training",
                        "schema": fieldnames,
                        "expected_rows": len(rows),
                        "expected_sha256": hashlib.sha256(idsmsa_path.read_bytes()).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    igar_path = tmp_path / "external-test.csv"
    with igar_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=preparer.IGAR_SCHEMA, lineterminator="\n")
        writer.writeheader()
        writer.writerow({"": "1", "app": "test", "content": "IGAR overlap"})

    result = preparer.prepare(
        idsmsa_path=idsmsa_path,
        smsa_data_dir=smsa_dir,
        smsa_split_manifest_path=smsa_manifest,
        training_manifest_path=training_manifest,
        output_dir=tmp_path / "processed",
        igar_paths=[igar_path],
        seed=42,
    )

    assert result["overlap_guard"]["smsa_rows_removed"] == 1
    assert result["overlap_guard"]["igar_rows_removed"] == 1
    assert result["training_policy"]["igar_labels_used"] is False
    assert result["training_policy"]["igar_metrics_used"] is False
    assert sum(result["splits"][split]["rows"] for split in preparer.SPLITS) == 4
