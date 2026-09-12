"""Deterministic text preparation shared by Week 2 AI experiments.

The first version deliberately keeps Indonesian text intact. It normalizes
Unicode and whitespace, but does not remove stopwords, punctuation, emoji, or
slang because those signals may be useful to the sentiment model.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


PREPROCESSING_VERSION = "preprocessing-v1"
CANONICAL_LABELS = ("positive", "neutral", "negative")

_WHITESPACE_RE = re.compile(r"\s+")
_MISSING_TEXT_VALUES = {"", "na", "n/a", "nan", "none", "null"}
_LABEL_ALIASES = {
    "positive": "positive",
    "positif": "positive",
    "pos": "positive",
    "neutral": "neutral",
    "netral": "neutral",
    "neu": "neutral",
    "negative": "negative",
    "negatif": "negative",
    "neg": "negative",
}


class PreprocessingError(ValueError):
    """Raised when a row cannot be safely prepared for model training."""


@dataclass(frozen=True)
class PreparedRow:
    """A normalized, traceable labeled text row."""

    source_row_number: int
    text: str
    label: str


@dataclass(frozen=True)
class PreparedInferenceRow:
    """A normalized, traceable unlabeled text row for inference."""

    source_row_number: int
    text: str


@dataclass(frozen=True)
class PreparationReport:
    """Counters describing deterministic row filtering decisions."""

    input_rows: int
    output_rows: int
    missing_text_rows: int
    duplicate_rows: int


def normalize_text(value: object) -> str:
    """Normalize Unicode/whitespace without deleting sentiment-bearing text."""

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\ufeff", "").replace("\u200b", "")
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_label(value: object) -> str:
    """Map supported dataset spellings to the AI contract labels.

    Unknown values fail closed. Silent guessing would make evaluation metrics
    impossible to trust.
    """

    label = normalize_text(value).casefold()
    if label in _MISSING_TEXT_VALUES:
        raise PreprocessingError("label is missing")
    try:
        return _LABEL_ALIASES[label]
    except KeyError as exc:
        allowed = ", ".join(CANONICAL_LABELS)
        raise PreprocessingError(f"unsupported label {label!r}; expected {allowed}") from exc


def prepare_labeled_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    text_column: str,
    label_column: str,
    drop_duplicates: bool = True,
) -> tuple[list[PreparedRow], PreparationReport]:
    """Prepare rows while preserving source row numbers.

    Empty text is skipped because there is no model input to classify. Exact
    duplicates after normalization are dropped deterministically, while the
    same normalized text carrying different labels raises an error: retaining
    such a row would make the official split vulnerable to label leakage.
    """

    if not text_column.strip() or not label_column.strip():
        raise PreprocessingError("text_column and label_column are required")

    prepared: list[PreparedRow] = []
    seen_labels: dict[str, str] = {}
    input_rows = 0
    missing_text_rows = 0
    duplicate_rows = 0

    for source_row_number, row in enumerate(rows, start=1):
        input_rows += 1
        text = normalize_text(row.get(text_column))
        if text.casefold() in _MISSING_TEXT_VALUES:
            missing_text_rows += 1
            continue

        label = normalize_label(row.get(label_column))
        text_key = text.casefold()
        existing_label = seen_labels.get(text_key)
        if existing_label is not None:
            if existing_label != label:
                raise PreprocessingError(
                    "conflicting labels for normalized text "
                    f"{text[:80]!r}: {existing_label!r} vs {label!r} "
                    f"(source row {source_row_number})"
                )
            if drop_duplicates:
                duplicate_rows += 1
                continue

        seen_labels[text_key] = label
        prepared.append(
            PreparedRow(
                source_row_number=source_row_number,
                text=text,
                label=label,
            )
        )

    return prepared, PreparationReport(
        input_rows=input_rows,
        output_rows=len(prepared),
        missing_text_rows=missing_text_rows,
        duplicate_rows=duplicate_rows,
    )


def prepare_inference_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    text_column: str,
) -> tuple[list[PreparedInferenceRow], PreparationReport]:
    """Prepare unlabeled rows without changing their order or multiplicity.

    Missing text is skipped, while every retained row keeps its original
    one-based source row number. The text column must be named and present in
    each input row so an upload cannot silently produce an empty inference
    batch.
    """

    if not isinstance(text_column, str) or not text_column.strip():
        raise PreprocessingError("text_column is required")

    prepared: list[PreparedInferenceRow] = []
    seen_texts: set[str] = set()
    input_rows = 0
    missing_text_rows = 0
    duplicate_rows = 0

    for source_row_number, row in enumerate(rows, start=1):
        input_rows += 1
        if text_column not in row:
            raise PreprocessingError(
                f"text column {text_column!r} is missing from source row "
                f"{source_row_number}"
            )

        text = normalize_text(row[text_column])
        if text.casefold() in _MISSING_TEXT_VALUES:
            missing_text_rows += 1
            continue

        text_key = text.casefold()
        if text_key in seen_texts:
            duplicate_rows += 1
        seen_texts.add(text_key)
        prepared.append(
            PreparedInferenceRow(
                source_row_number=source_row_number,
                text=text,
            )
        )

    return prepared, PreparationReport(
        input_rows=input_rows,
        output_rows=len(prepared),
        missing_text_rows=missing_text_rows,
        duplicate_rows=duplicate_rows,
    )
