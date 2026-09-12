from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ai.model_bundle import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    LoadedSentimentModel,
    SentimentModelBundle,
)
from app.ai.sentiment_inference import SentimentBatchInferencer, SentimentInferenceError


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        import torch

        self.calls.append((list(texts), kwargs))
        return {
            "input_ids": torch.tensor([[len(text)] for text in texts]),
            "attention_mask": torch.ones((len(texts), 1), dtype=torch.long),
        }


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(num_labels=3, id2label=ID_TO_LABEL)
        self.eval_called = False

    def __call__(self, **encoded):
        import torch

        lengths = encoded["input_ids"][:, 0].tolist()
        logits = []
        for length in lengths:
            if length % 3 == 0:
                logits.append([0.0, 0.0, 4.0])
            elif length % 2 == 0:
                logits.append([4.0, 0.0, 0.0])
            else:
                logits.append([0.0, 4.0, 0.0])
        return SimpleNamespace(logits=torch.tensor(logits))

    def to(self, device):
        assert device == "cpu"
        return self

    def eval(self):
        self.eval_called = True
        return self


def _inferencer(*, batch_size=8):
    bundle = SentimentModelBundle(
        path=Path("/tmp/model-v1"),
        model_version="sentiment-model-v1",
        model_name="indobenchmark/indobert-base-p1",
        model_revision="revision",
        preprocessing_version="preprocessing-v1",
        label_to_id=LABEL_TO_ID,
        id_to_label=ID_TO_LABEL,
        required_files=(),
    )
    tokenizer = FakeTokenizer()
    model = FakeModel()
    loaded = LoadedSentimentModel(
        bundle=bundle,
        tokenizer=tokenizer,
        model=model,
        device="cpu",
    )
    return SentimentBatchInferencer(loaded, batch_size=batch_size), tokenizer, model


def test_predict_batch_preserves_order_duplicates_and_contract_fields():
    inferencer, tokenizer, model = _inferencer(batch_size=2)

    predictions = inferencer.predict_batch(["  Bagus  ", "OTP", "OTP"])

    assert [prediction.text for prediction in predictions] == ["Bagus", "OTP", "OTP"]
    assert [prediction.source_row_number for prediction in predictions] == [1, 2, 3]
    assert [prediction.sentiment for prediction in predictions] == [
        "neutral",
        "negative",
        "negative",
    ]
    assert all(0 <= prediction.confidence <= 1 for prediction in predictions)
    assert all(
        abs(
            prediction.probabilities.positive
            + prediction.probabilities.neutral
            + prediction.probabilities.negative
            - 1
        )
        <= 0.01
        for prediction in predictions
    )
    assert len(tokenizer.calls) == 2
    assert model.eval_called is False


def test_predict_rows_skips_missing_and_keeps_source_row_numbers():
    inferencer, _, _ = _inferencer()

    predictions, report = inferencer.predict_rows(
        [
            {"feedback": "Bagus"},
            {"feedback": None},
            {"feedback": "OTP"},
        ],
        text_column="feedback",
    )

    assert [prediction.source_row_number for prediction in predictions] == [1, 3]
    assert report.missing_text_rows == 1
    assert report.output_rows == 2


def test_predict_batch_rejects_empty_item_and_bad_batch_configuration():
    with pytest.raises(SentimentInferenceError, match="batch_size must be positive"):
        inferencer, _, _ = _inferencer()
        SentimentBatchInferencer(inferencer.loaded_model, batch_size=0)

    inferencer, _, _ = _inferencer()
    with pytest.raises(SentimentInferenceError, match="index 1 is empty"):
        inferencer.predict_batch(["valid", "  "])


def test_predict_batch_returns_empty_for_empty_sequence():
    inferencer, _, _ = _inferencer()

    assert inferencer.predict_batch([]) == []


def test_default_decision_matches_raw_argmax_and_confidence():
    inferencer, _, _ = _inferencer()

    prediction = inferencer.predict_batch(["OTP"])[0]

    assert prediction.sentiment == "negative"
    assert prediction.confidence == prediction.probabilities.negative


def test_decision_bias_can_select_non_maximum_raw_class_without_changing_map():
    default_inferencer, _, _ = _inferencer()
    biased_inferencer = SentimentBatchInferencer(
        default_inferencer.loaded_model,
        decision_bias={"positive": 5.0, "neutral": 0.0, "negative": 0.0},
    )

    default_prediction = default_inferencer.predict_batch(["Bagus"])[0]
    biased_prediction = biased_inferencer.predict_batch(["Bagus"])[0]

    assert default_prediction.sentiment == "neutral"
    assert biased_prediction.sentiment == "positive"
    assert biased_prediction.probabilities == default_prediction.probabilities
    assert biased_prediction.confidence == biased_prediction.probabilities.positive
    assert biased_prediction.confidence < max(
        biased_prediction.probabilities.neutral,
        biased_prediction.probabilities.negative,
    )


@pytest.mark.parametrize(
    "decision_bias",
    [
        {"positive": 0.0, "neutral": 0.0},
        {"positive": 0.0, "neutral": 0.0, "negative": 0.0, "other": 0.0},
        {"positive": float("nan"), "neutral": 0.0, "negative": 0.0},
        {"positive": 0.0, "neutral": float("inf"), "negative": 0.0},
        {"positive": "not-a-number", "neutral": 0.0, "negative": 0.0},
    ],
)
def test_decision_bias_rejects_malformed_or_non_finite_values(decision_bias):
    inferencer, _, _ = _inferencer()

    with pytest.raises(SentimentInferenceError, match="decision_bias"):
        SentimentBatchInferencer(inferencer.loaded_model, decision_bias=decision_bias)
