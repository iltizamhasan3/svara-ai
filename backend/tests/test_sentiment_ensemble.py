from pathlib import Path
import hashlib
import json

import pytest

from app.ai.sentiment_ensemble import (
    SentimentEnsembleError,
    blend_probability_maps,
    load_tfidf_model,
)


def test_blend_probability_and_bias_change_decision():
    sentiment, probabilities, confidence = blend_probability_maps(
        {"positive": 0.8, "neutral": 0.1, "negative": 0.1},
        {"positive": 0.1, "neutral": 0.8, "negative": 0.1},
        bert_weight=0.25,
    )
    assert sentiment == "neutral"
    assert probabilities["neutral"] == pytest.approx(0.625)
    assert confidence == pytest.approx(0.625)
    biased, _, _ = blend_probability_maps(
        {"positive": 0.8, "neutral": 0.1, "negative": 0.1},
        {"positive": 0.1, "neutral": 0.8, "negative": 0.1},
        bert_weight=0.25,
        decision_bias={"positive": 2.0, "neutral": 0.0, "negative": 0.0},
    )
    assert biased == "positive"


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), "bad"])
def test_blend_rejects_malformed_weight(weight):
    with pytest.raises((SentimentEnsembleError, ValueError)):
        blend_probability_maps(
            {"positive": 1, "neutral": 0, "negative": 0},
            {"positive": 0, "neutral": 1, "negative": 0},
            bert_weight=weight,
        )


def test_loader_rejects_bad_manifest_and_checksum(tmp_path):
    model = tmp_path / "model.joblib"
    manifest = tmp_path / "manifest.json"
    model.write_bytes(b"not-a-joblib")
    manifest.write_text(json.dumps({"labels": ["positive", "negative", "neutral"]}), encoding="utf-8")
    with pytest.raises(SentimentEnsembleError, match="labels"):
        load_tfidf_model(model, manifest)
    manifest.write_text(json.dumps({"labels": ["positive", "neutral", "negative"], "model_sha256": hashlib.sha256(b"wrong").hexdigest(), "igar_forbidden": True, "igar_read": False, "igar_labels_used": False, "igar_metrics_used": False}), encoding="utf-8")
    with pytest.raises(SentimentEnsembleError, match="checksum"):
        load_tfidf_model(model, manifest)


def test_loader_rejects_sealed_path(tmp_path):
    path = tmp_path / "igar" / "model.joblib"
    with pytest.raises(SentimentEnsembleError, match="sealed"):
        load_tfidf_model(path, tmp_path / "manifest.json")
