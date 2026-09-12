import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def load_preparer():
    spec = importlib.util.spec_from_file_location(
        "prepare_sentiment_tuning_mix", ROOT / "scripts/prepare_sentiment_tuning_mix.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mix_training_rows_adds_a_deterministic_neutral_sample():
    preparer = load_preparer()
    rows, report = preparer.mix_training_rows(
        [
            preparer.PreparedRow(1, "bagus", "positive"),
            preparer.PreparedRow(2, "buruk", "negative"),
        ],
        [
            preparer.PreparedRow(10, "biasa satu", "neutral"),
            preparer.PreparedRow(11, "biasa dua", "neutral"),
            preparer.PreparedRow(12, "positif", "positive"),
        ],
        smsa_neutral_limit=1,
        seed=42,
    )

    assert len(rows) == 3
    assert [row.label for row in rows] == ["positive", "negative", "neutral"]
    assert report == {
        "google_train_rows": 2,
        "smsa_neutral_available": 2,
        "smsa_neutral_selected": 1,
        "accepted_train_rows": 3,
        "duplicate_rows_removed": 0,
        "conflicting_rows": 0,
    }


def test_mix_training_rows_rejects_conflicting_normalized_labels():
    preparer = load_preparer()

    with pytest.raises(ValueError, match="conflicting labels"):
        preparer.mix_training_rows(
            [preparer.PreparedRow(1, "campur", "positive")],
            [preparer.PreparedRow(2, " CAMPUR ", "neutral")],
            smsa_neutral_limit=None,
            seed=42,
        )


def test_mix_training_rows_rejects_an_oversized_neutral_sample():
    preparer = load_preparer()

    with pytest.raises(ValueError, match="exceeds available"):
        preparer.mix_training_rows(
            [],
            [preparer.PreparedRow(1, "biasa", "neutral")],
            smsa_neutral_limit=2,
            seed=42,
        )
