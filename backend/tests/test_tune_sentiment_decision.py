import csv
import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[2]


def load_tuner():
    spec = importlib.util.spec_from_file_location(
        "tune_sentiment_decision", ROOT / "scripts/tune_sentiment_decision.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_biases_uses_log_probability_and_canonical_tie_order():
    tuner = load_tuner()
    probabilities = [{"positive": 0.4, "neutral": 0.35, "negative": 0.25}]
    assert tuner.apply_biases(probabilities, {"positive": 0, "neutral": 0, "negative": 0}) == ["positive"]
    assert tuner.apply_biases(probabilities, {"positive": 0, "neutral": 1, "negative": 0}) == ["neutral"]


def test_search_biases_is_deterministic_and_prefers_lower_l1_on_tie():
    tuner = load_tuner()
    probabilities = [
        {"positive": 0.34, "neutral": 0.33, "negative": 0.33},
        {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
        {"positive": 0.33, "neutral": 0.33, "negative": 0.34},
    ]
    result = tuner.search_biases(
        probabilities, ["positive", "neutral", "negative"], bias_min=-0.1, bias_max=0.1, bias_step=0.1
    )
    assert result["biases"] == {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    assert result["metrics"]["macro_f1"] == pytest.approx(1.0)


def test_search_biases_enforces_primary_and_guard_floors():
    tuner = load_tuner()
    probabilities = [
        {"positive": 0.8, "neutral": 0.1, "negative": 0.1},
        {"positive": 0.1, "neutral": 0.8, "negative": 0.1},
        {"positive": 0.1, "neutral": 0.1, "negative": 0.8},
    ]
    result = tuner.search_biases(
        probabilities, ["positive", "neutral", "negative"],
        guard_probabilities=probabilities,
        guard_expected=["positive", "neutral", "negative"],
        primary_accuracy_floor=1.0,
        min_guard_accuracy=1.0,
        min_guard_macro_f1=1.0,
        primary_macro_f1_floor=1.0,
        selection_metric="accuracy",
        bias_min=-0.1, bias_max=0.1, bias_step=0.1,
    )
    assert result["metrics"]["accuracy"] >= 1.0
    assert result["guard_metrics"]["metrics"]["accuracy"] >= 1.0
    assert result["guard_metrics"]["metrics"]["macro_f1"] >= 1.0


def test_search_biases_raises_when_no_candidate_meets_floor():
    tuner = load_tuner()
    probabilities = [{"positive": 0.8, "neutral": 0.1, "negative": 0.1}]
    with pytest.raises(ValueError, match="no bias candidate satisfies"):
        tuner.search_biases(
            probabilities, ["negative"], primary_accuracy_floor=1.0,
            bias_min=0, bias_max=0, bias_step=1,
        )


def test_search_biases_without_guard_preserves_unconstrained_result():
    tuner = load_tuner()
    probabilities = [
        {"positive": 0.34, "neutral": 0.33, "negative": 0.33},
        {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
        {"positive": 0.33, "neutral": 0.33, "negative": 0.34},
    ]
    result = tuner.search_biases(
        probabilities, ["positive", "neutral", "negative"],
        bias_min=-0.1, bias_max=0.1, bias_step=0.1,
    )
    assert result["biases"] == {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    assert "guard_metrics" not in result


def test_tuner_rejects_igar_path():
    tuner = load_tuner()
    with pytest.raises(ValueError, match="sealed external-test"):
        tuner._evaluator_module().read_labeled_tsv(ROOT / "data/raw/igar/ratings.tsv")


def test_run_writes_calibration_artifact_shape(tmp_path, monkeypatch):
    tuner = load_tuner()
    input_path = tmp_path / "validation.tsv"
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(
            [["bagus", "positive"], ["biasa", "neutral"], ["buruk", "negative"]]
        )

    class ProbabilityMap:
        def __init__(self, **values):
            self.values = values

        def model_dump(self):
            return self.values

    class FakeInferencer:
        loaded_model = SimpleNamespace(
            bundle=SimpleNamespace(
                path=tmp_path / "model", model_version="v1", model_name="fake",
                model_revision="r1", preprocessing_version="p1",
                label_to_id={"positive": 0, "neutral": 1, "negative": 2},
            )
        )

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def predict_prepared_rows(self, rows):
            maps = [
                {"positive": 0.8, "neutral": 0.1, "negative": 0.1},
                {"positive": 0.1, "neutral": 0.8, "negative": 0.1},
                {"positive": 0.1, "neutral": 0.1, "negative": 0.8},
            ]
            return [SimpleNamespace(sentiment=label, probabilities=ProbabilityMap(**values))
                    for label, values in zip(("positive", "neutral", "negative"), maps)]

    monkeypatch.setattr(tuner, "SentimentBatchInferencer", FakeInferencer)
    output = tmp_path / "calibration.json"
    result = tuner.run(Namespace(
        input=input_path, model_dir=tmp_path, export_manifest=None, output=output,
        batch_size=2, max_length=16, torch_threads=1,
        bias_min=-0.1, bias_max=0.1, bias_step=0.1,
    ))
    artifact = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert artifact == result
    assert {"baseline", "calibrated", "grid", "model", "input", "evaluation_policy"} <= artifact.keys()
    assert artifact["calibrated"]["biases"]["positive"] == 0.0
    assert artifact["evaluation_policy"] == {
        "igar_read": False, "igar_labels_used": False,
        "igar_metrics_used": False, "training_or_tuning": True,
    }
