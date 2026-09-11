import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
_UNSET = object()


def load_runner():
    spec = importlib.util.spec_from_file_location("indobert_runner", ROOT / "scripts/train_indobert.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_label_order_is_canonical():
    runner = load_runner()
    assert runner.CANONICAL_LABELS == ("positive", "neutral", "negative")
    assert runner.LABEL_TO_ID == {"positive": 0, "neutral": 1, "negative": 2}


def test_disjointness_guard():
    runner = load_runner()
    row = runner.PreparedRow(1, " Satu  teks ", "positive")
    with pytest.raises(ValueError, match="overlaps"):
        runner.assert_disjoint_splits({"train": [row], "validation": [runner.PreparedRow(2, "Satu teks", "positive")]})


def test_test_evaluation_requires_frozen_manifest():
    runner = load_runner()
    assert runner.test_evaluation_allowed({}, False) is False
    with pytest.raises(ValueError, match="test evaluation"):
        runner.test_evaluation_allowed({}, True)
    assert runner.test_evaluation_allowed({"test_evaluation_frozen": True}, True) is True


def _fixture_manifest(tmp_path: Path, runner, *, checksum_override=_UNSET):
    data_dir = tmp_path / "selected-smsa"
    data_dir.mkdir()
    contents = {
        "train_preprocess.tsv": "train positif\tPositive\ntrain netral\tNeutral\ntrain negatif\tNegative\n",
        "valid_preprocess.tsv": "valid positif\tPositive\nvalid netral\tNeutral\nvalid negatif\tNegative\n",
        "test_preprocess.tsv": "test positif\tPositive\ntest netral\tNeutral\ntest negatif\tNegative\n",
    }
    split_details = {}
    for filename, content in contents.items():
        path = data_dir / filename
        path.write_text(content, encoding="utf-8")
        split = filename.split("_")[0]
        split = "validation" if split == "valid" else split
        split_details[split] = {
            "file": f"data/raw/smsa/{filename}",
            "sha256": runner.sha256_file(path),
        }
    if checksum_override is not _UNSET:
        split_details["train"]["sha256"] = checksum_override
    manifest_path = tmp_path / "split-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "SmSA",
                "preprocessing_version": runner.PREPROCESSING_VERSION,
                "splits": split_details,
            }
        ),
        encoding="utf-8",
    )
    return data_dir, manifest_path


def test_loader_uses_selected_data_directory(tmp_path, monkeypatch):
    runner = load_runner()
    data_dir, manifest_path = _fixture_manifest(tmp_path, runner)
    monkeypatch.chdir(tmp_path)
    splits, _ = runner.load_splits(data_dir, manifest_path)
    assert {split: len(rows) for split, rows in splits.items()} == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }


@pytest.mark.parametrize("checksum", [None, "not-a-sha256", "0" * 63])
def test_loader_rejects_missing_or_malformed_checksum(tmp_path, checksum):
    runner = load_runner()
    data_dir, manifest_path = _fixture_manifest(tmp_path, runner, checksum_override=checksum)
    with pytest.raises(ValueError, match="64-character SHA-256"):
        runner.load_splits(data_dir, manifest_path)
