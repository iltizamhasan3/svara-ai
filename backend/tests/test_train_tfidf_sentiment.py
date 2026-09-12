import csv
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def load_script():
    spec = importlib.util.spec_from_file_location("train_tfidf_sentiment", ROOT / "scripts/train_tfidf_sentiment.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_path_rejects_igar_without_reading(tmp_path):
    script = load_script()
    with pytest.raises(ValueError, match="sealed"):
        script.assert_not_igar_training_path(tmp_path / "IGAR" / "labels.tsv")


def test_read_training_tsv_normalizes_and_requires_all_labels(tmp_path):
    script = load_script()
    path = tmp_path / "train.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows([[" bagus ", "Positif"], ["biasa", "netral"], ["buruk", "Negatif"]])
    rows, report = script.read_training_tsv(path)
    assert [row.label for row in rows] == ["positive", "neutral", "negative"]
    assert report.output_rows == 3

    path.write_text("bagus\tpositive\n", encoding="utf-8")
    with pytest.raises(ValueError, match="every canonical"):
        script.read_training_tsv(path)
