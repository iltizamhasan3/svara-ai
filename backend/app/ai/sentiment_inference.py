"""CPU batch inference for the exported IndoBERT sentiment model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.ai.model_bundle import LoadedSentimentModel, load_sentiment_model
from app.ai.preprocessing import (
    PreparedInferenceRow,
    PreparationReport,
    normalize_text,
    prepare_inference_rows,
)
from app.schemas.ai import ProbabilityMap


class SentimentInferenceError(ValueError):
    """Raised when a batch cannot produce contract-safe sentiment results."""


@dataclass(frozen=True)
class SentimentPrediction:
    """A traceable sentiment prediction without topic assignment."""

    source_row_number: int
    text: str
    sentiment: str
    confidence: float
    probabilities: ProbabilityMap


class SentimentBatchInferencer:
    """Run deterministic, ordered inference over fixed-size CPU batches."""

    def __init__(
        self,
        loaded_model: LoadedSentimentModel,
        *,
        batch_size: int = 8,
        max_length: int = 128,
    ) -> None:
        if batch_size < 1:
            raise SentimentInferenceError("batch_size must be positive")
        if max_length < 8:
            raise SentimentInferenceError("max_length must be at least 8")
        self.loaded_model = loaded_model
        self.batch_size = batch_size
        self.max_length = max_length

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        *,
        batch_size: int = 8,
        max_length: int = 128,
        device: str = "cpu",
        local_files_only: bool = True,
        torch_threads: int | None = 4,
        export_manifest_path: Path | str | None = None,
    ) -> "SentimentBatchInferencer":
        """Load a local export and construct the batch adapter."""

        loaded_model = load_sentiment_model(
            model_dir,
            device=device,
            local_files_only=local_files_only,
            torch_threads=torch_threads,
            export_manifest_path=export_manifest_path,
        )
        return cls(loaded_model, batch_size=batch_size, max_length=max_length)

    def predict_batch(self, texts: Sequence[object]) -> list[SentimentPrediction]:
        """Predict a non-empty sequence while preserving order and duplicates.

        Direct batch callers must supply usable text for every item. CSV-like
        inputs that contain missing values should use :meth:`predict_rows`,
        which skips only those rows and retains source-row traceability.
        """

        if isinstance(texts, (str, bytes)):
            raise SentimentInferenceError("texts must be a sequence of text values")
        normalized_texts = [normalize_text(value) for value in texts]
        if not normalized_texts:
            return []
        for index, text in enumerate(normalized_texts):
            if not text:
                raise SentimentInferenceError(f"batch text at index {index} is empty")

        return self._predict_normalized(
            normalized_texts,
            source_row_numbers=range(1, len(normalized_texts) + 1),
        )

    def predict_rows(
        self,
        rows: Iterable[Mapping[str, object]],
        *,
        text_column: str,
    ) -> tuple[list[SentimentPrediction], PreparationReport]:
        """Prepare and predict upload rows with a preprocessing report."""

        prepared, report = prepare_inference_rows(rows, text_column=text_column)
        predictions = self.predict_prepared_rows(prepared)
        return predictions, report

    def predict_prepared_rows(
        self,
        rows: Sequence[PreparedInferenceRow],
    ) -> list[SentimentPrediction]:
        """Predict already prepared rows and retain their source positions."""

        if not rows:
            return []
        return self._predict_normalized(
            [row.text for row in rows],
            source_row_numbers=[row.source_row_number for row in rows],
        )

    def _predict_normalized(
        self,
        texts: Sequence[str],
        *,
        source_row_numbers: Iterable[int],
    ) -> list[SentimentPrediction]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency boundary
            raise SentimentInferenceError(
                "install the backend AI extra before running sentiment inference"
            ) from exc

        source_numbers = list(source_row_numbers)
        if len(source_numbers) != len(texts):
            raise SentimentInferenceError("source row numbers must match the text batch")

        predictions: list[SentimentPrediction] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = list(texts[start : start + self.batch_size])
            encoded = self.loaded_model.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.loaded_model.device)
                if hasattr(value, "to")
                else value
                for key, value in encoded.items()
            }
            with torch.inference_mode():
                outputs = self.loaded_model.model(**encoded)
                logits = getattr(outputs, "logits", None)
                if logits is None and isinstance(outputs, (tuple, list)) and outputs:
                    logits = outputs[0]
                if logits is None:
                    raise SentimentInferenceError("model output does not contain logits")
                probabilities = torch.softmax(logits, dim=-1).detach().cpu()

            if probabilities.ndim != 2 or probabilities.shape[0] != len(batch_texts):
                raise SentimentInferenceError(
                    "model logits must have one probability row per input text"
                )
            if probabilities.shape[1] != len(self.loaded_model.bundle.id_to_label):
                raise SentimentInferenceError(
                    "model logits class count does not match the validated label mapping"
                )

            for offset, values in enumerate(probabilities.tolist()):
                prediction = self._prediction_from_probabilities(
                    text=batch_texts[offset],
                    source_row_number=source_numbers[start + offset],
                    values=values,
                )
                predictions.append(prediction)

        return predictions

    def _prediction_from_probabilities(
        self,
        *,
        text: str,
        source_row_number: int,
        values: Sequence[float],
    ) -> SentimentPrediction:
        label_by_id = self.loaded_model.bundle.id_to_label
        probabilities_by_label = {
            label_by_id[index]: float(values[index])
            for index in sorted(label_by_id)
        }
        predicted_id = max(range(len(values)), key=values.__getitem__)
        try:
            sentiment = label_by_id[predicted_id]
        except KeyError as exc:
            raise SentimentInferenceError(
                f"model predicted unmapped class id {predicted_id}"
            ) from exc
        probability_map = ProbabilityMap(**probabilities_by_label)
        return SentimentPrediction(
            source_row_number=source_row_number,
            text=text,
            sentiment=sentiment,
            confidence=max(probabilities_by_label.values()),
            probabilities=probability_map,
        )
