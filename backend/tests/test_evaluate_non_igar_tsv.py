import csv
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "non_igar_tsv_evaluator", ROOT / "scripts/evaluate_non_igar_tsv.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluator_rejects_igar_input_path():
    evaluator = load_evaluator()
    with pytest.raises(ValueError, match="sealed external-test"):
        evaluator.assert_not_igar_input_path(ROOT / "data/raw/igar/Rating_labeled.csv")


def test_read_labeled_tsv_uses_shared_preprocessing(tmp_path):
    evaluator = load_evaluator()
    path = tmp_path / "holdout.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows([["  bagus  ", "Positif"], ["N/A", "negative"], ["buruk", "negative"]])

    rows, report = evaluator.read_labeled_tsv(path)

    assert [(row.text, row.label) for row in rows] == [("bagus", "positive"), ("buruk", "negative")]
    assert report.missing_text_rows == 1


def test_classification_metrics_reports_canonical_order():
    evaluator = load_evaluator()

    result = evaluator.classification_metrics(
        ["positive", "neutral", "negative"],
        ["positive", "negative", "negative"],
    )

    assert result["labels"] == ["positive", "neutral", "negative"]
    assert result["metrics"]["accuracy"] == pytest.approx(2 / 3)
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
