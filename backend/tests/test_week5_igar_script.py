import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

from app.ai.preprocessing import PreparationReport
from app.ai.sentiment_inference import SentimentPrediction
from app.schemas.ai import ProbabilityMap


ROOT = Path(__file__).parents[2]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "week5_igar_runner", ROOT / "scripts/run_week5_igar_inference.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prediction(source_row_number: int, text: str, sentiment: str) -> SentimentPrediction:
    probabilities = {
        "positive": 0.9 if sentiment == "positive" else 0.05,
        "neutral": 0.9 if sentiment == "neutral" else 0.05,
        "negative": 0.9 if sentiment == "negative" else 0.05,
    }
    return SentimentPrediction(
        source_row_number=source_row_number,
        text=text,
        sentiment=sentiment,
        confidence=0.9,
        probabilities=ProbabilityMap(**probabilities),
    )


def test_error_category_is_a_deterministic_review_bucket():
    runner = load_runner()

    assert runner.error_category("bagus tapi lambat") == "mixed_signal"
    assert runner.error_category("OTP") == "short_text"
    assert runner.error_category("bisa login?") == "question_or_request"
    assert runner.error_category("kata " * 30) == "long_text"
    assert runner.error_category("aplikasi ini memerlukan perbaikan") == "other"


def test_run_writes_igar_predictions_metrics_and_errors(tmp_path, monkeypatch):
    runner = load_runner()
    input_path = tmp_path / "igar.csv"
    input_path.write_text(
        "content,labelScoreBase\nBagus sekali,Positive\nOTP,Neutral\nCepat,Positive\n",
        encoding="utf-8",
    )

    class FakeInferencer:
        loaded_model = type(
            "Loaded",
            (),
            {
                "device": "cpu",
                "bundle": type(
                    "Bundle",
                    (),
                    {
                        "model_version": "sentiment-model-v1",
                        "model_name": "model",
                        "model_revision": "revision",
                        "preprocessing_version": "preprocessing-v1",
                        "label_to_id": {"positive": 0, "neutral": 1, "negative": 2},
                    },
                )()
            },
        )()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def predict_rows(self, rows, *, text_column):
            return [
                _prediction(1, "Bagus sekali", "positive"),
                _prediction(2, "OTP", "negative"),
                _prediction(3, "Cepat", "positive"),
            ], PreparationReport(3, 3, 0, 0)

    monkeypatch.setattr(runner, "SentimentBatchInferencer", FakeInferencer)
    output_dir = tmp_path / "output"
    payload = runner.run(
        Namespace(
            input=input_path,
            model_dir=tmp_path / "model",
            output_dir=output_dir,
            text_column="content",
            label_column="labelScoreBase",
            batch_size=2,
            max_length=64,
            torch_threads=1,
        )
    )

    assert payload["metrics"]["accuracy"] == pytest.approx(2 / 3)
    assert payload["input"]["evaluated_rows"] == 3
    assert payload["inference"]["batch_size"] == 2
    assert payload["inference"]["max_length"] == 64
    assert payload["inference"]["torch_threads"] == 1
    assert payload["inference"]["text_column"] == "content"
    assert payload["inference"]["label_column"] == "labelScoreBase"
    assert payload["inference"]["runtime_versions"]["python"]
    assert payload["artifacts"]["predictions"] == "igar_predictions.csv"
    assert (output_dir / "igar_predictions.csv").is_file()
    assert (output_dir / "igar_error_analysis.csv").is_file()
    error_report = json.loads((output_dir / "igar_error_analysis.json").read_text())
    assert error_report["error_count"] == 1
    assert error_report["category_counts"] == {"short_text": 1}


def test_read_csv_rows_rejects_empty_input(tmp_path):
    runner = load_runner()
    path = tmp_path / "empty.csv"
    path.write_text("content,labelScoreBase\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one data row"):
        runner.read_csv_rows(path)
