import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


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
