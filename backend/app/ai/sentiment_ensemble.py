"""Fail-closed TF-IDF sentiment artifacts and IndoBERT probability blending."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.ai.preprocessing import CANONICAL_LABELS, normalize_text
from app.ai.sentiment_inference import SentimentPrediction
from app.schemas.ai import ProbabilityMap


class SentimentEnsembleError(ValueError):
    """Raised when an ensemble artifact or probability vector is unsafe."""


def _sealed(path: Path) -> None:
    resolved = path.resolve()
    if "igar" in {part.casefold() for part in resolved.parts} or "rating_labeled" in resolved.name.casefold():
        raise SentimentEnsembleError("IGAR is sealed external-test data and cannot be read")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TfidfSentimentModel:
    vectorizer: Any
    classifier: Any
    manifest: Mapping[str, Any]

    def predict_proba(self, texts: Sequence[object]) -> list[dict[str, float]]:
        if isinstance(texts, (str, bytes)):
            raise SentimentEnsembleError("TF-IDF input texts must be a sequence")
        normalized = [normalize_text(text) for text in texts]
        if any(not text for text in normalized):
            raise SentimentEnsembleError("TF-IDF input texts must be non-empty")
        values = self.classifier.predict_proba(self.vectorizer.transform(normalized))
        classes = [str(value) for value in self.classifier.classes_]
        if set(classes) != set(CANONICAL_LABELS):
            raise SentimentEnsembleError("TF-IDF model classes must be canonical labels")
        return [
            _validate_probabilities(dict(zip(classes, row, strict=True)))
            for row in values
        ]


def _validate_probabilities(values: Mapping[str, object]) -> dict[str, float]:
    if set(values) != set(CANONICAL_LABELS):
        raise SentimentEnsembleError("probabilities must contain exactly canonical labels")
    result = {}
    for label in CANONICAL_LABELS:
        if isinstance(values[label], bool):
            raise SentimentEnsembleError("probabilities must be finite numbers")
        number = float(values[label])
        if not math.isfinite(number) or number < 0 or number > 1:
            raise SentimentEnsembleError("probabilities must be in [0, 1]")
        result[label] = number
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-5):
        raise SentimentEnsembleError("probabilities must sum to 1")
    return result


def load_tfidf_model(model_path: Path | str, manifest_path: Path | str) -> TfidfSentimentModel:
    """Load a joblib artifact only when its provenance and checksum are valid."""

    model_path, manifest_path = Path(model_path).resolve(), Path(manifest_path).resolve()
    _sealed(model_path)
    _sealed(manifest_path)
    if not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(model_path if not model_path.is_file() else manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SentimentEnsembleError("TF-IDF manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise SentimentEnsembleError("TF-IDF manifest must be an object")
    if manifest.get("labels") != list(CANONICAL_LABELS):
        raise SentimentEnsembleError("TF-IDF manifest labels must be canonical and ordered")
    for key in ("igar_forbidden", "igar_read", "igar_labels_used", "igar_metrics_used"):
        if manifest.get(key) is not (key == "igar_forbidden"):
            raise SentimentEnsembleError(f"TF-IDF manifest must prove {key}={key == 'igar_forbidden'}")
    expected = manifest.get("model_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or sha256_file(model_path).casefold() != expected.casefold():
        raise SentimentEnsembleError("TF-IDF model checksum does not match manifest")
    try:
        import joblib
        artifact = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001 - artifact loading must fail closed
        raise SentimentEnsembleError("TF-IDF model could not be loaded") from exc
    if not isinstance(artifact, dict) or not all(key in artifact for key in ("vectorizer", "classifier")):
        raise SentimentEnsembleError("TF-IDF model artifact is malformed")
    try:
        classes = {str(value) for value in artifact["classifier"].classes_}
        if classes != set(CANONICAL_LABELS):
            raise SentimentEnsembleError("TF-IDF model classes must be canonical labels")
        if tuple(artifact["vectorizer"].ngram_range) != (1, 2) or artifact["vectorizer"].analyzer != "word" or artifact["vectorizer"].sublinear_tf is not True or artifact["vectorizer"].max_features != 100000:
            raise SentimentEnsembleError("TF-IDF vectorizer configuration does not match manifest")
    except AttributeError as exc:
        raise SentimentEnsembleError("TF-IDF model artifact is malformed") from exc
    return TfidfSentimentModel(artifact["vectorizer"], artifact["classifier"], manifest)


def blend_probability_maps(
    bert_probabilities: Mapping[str, object],
    tfidf_probabilities: Mapping[str, object],
    *,
    bert_weight: float = 0.5,
    decision_bias: Mapping[str, float] | None = None,
) -> tuple[str, dict[str, float], float]:
    """Blend canonical probabilities and apply additive log-bias exactly once."""

    try:
        numeric_weight = float(bert_weight)
    except (TypeError, ValueError) as exc:
        raise SentimentEnsembleError("bert_weight must be in [0, 1]") from exc
    if isinstance(bert_weight, bool) or not math.isfinite(numeric_weight) or not 0 <= numeric_weight <= 1:
        raise SentimentEnsembleError("bert_weight must be in [0, 1]")
    bert = _validate_probabilities(bert_probabilities)
    tfidf = _validate_probabilities(tfidf_probabilities)
    combined = {label: numeric_weight * bert[label] + (1 - numeric_weight) * tfidf[label] for label in CANONICAL_LABELS}
    if decision_bias is not None:
        if set(decision_bias) != set(CANONICAL_LABELS):
            raise SentimentEnsembleError("decision_bias labels must exactly match canonical labels")
        biases = {}
        for label in CANONICAL_LABELS:
            if isinstance(decision_bias[label], bool):
                raise SentimentEnsembleError("decision_bias must contain finite numbers")
            try:
                biases[label] = float(decision_bias[label])
            except (TypeError, ValueError) as exc:
                raise SentimentEnsembleError("decision_bias must contain finite numbers") from exc
            if not math.isfinite(biases[label]):
                raise SentimentEnsembleError("decision_bias must contain finite numbers")
        scores = {label: math.log(combined[label]) + biases[label] if combined[label] > 0 else float("-inf") for label in CANONICAL_LABELS}
    else:
        scores = combined
    sentiment = max(CANONICAL_LABELS, key=lambda label: scores[label])
    return sentiment, combined, combined[sentiment]


def blend_prediction(prediction: SentimentPrediction, tfidf_probabilities: Mapping[str, object], *, bert_weight: float = 0.5, decision_bias: Mapping[str, float] | None = None) -> SentimentPrediction:
    bert_probabilities = prediction.probabilities.model_dump() if hasattr(prediction.probabilities, "model_dump") else dict(prediction.probabilities)
    sentiment, probabilities, confidence = blend_probability_maps(bert_probabilities, tfidf_probabilities, bert_weight=bert_weight, decision_bias=decision_bias)
    return SentimentPrediction(prediction.source_row_number, prediction.text, sentiment, confidence, ProbabilityMap(**probabilities))
