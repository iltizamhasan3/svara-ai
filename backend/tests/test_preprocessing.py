import math

import pytest

from app.ai.preprocessing import (
    PREPROCESSING_VERSION,
    PreprocessingError,
    normalize_label,
    normalize_text,
    prepare_labeled_rows,
)


def test_normalize_text_preserves_emoji_slang_and_punctuation():
    assert normalize_text("\ufeff  OTP\u200b   nggak masuk 😭!! ") == "OTP nggak masuk 😭!!"


def test_normalize_label_maps_supported_spellings_and_rejects_unknown_values():
    assert normalize_label(" Positif ") == "positive"
    assert normalize_label("NETRAL") == "neutral"
    assert normalize_label("Negative") == "negative"

    with pytest.raises(PreprocessingError, match="unsupported label"):
        normalize_label("mixed")


def test_prepare_rows_drops_missing_text_and_same_label_duplicates():
    rows = [
        {"sentence": "Bagus  sekali", "label": "positive"},
        {"sentence": "  ", "label": "negative"},
        {"sentence": math.nan, "label": "neutral"},
        {"sentence": "bagus sekali", "label": "positif"},
        {"sentence": "OTP tidak masuk", "label": "Negative"},
    ]

    prepared, report = prepare_labeled_rows(rows, text_column="sentence", label_column="label")

    assert PREPROCESSING_VERSION == "preprocessing-v1"
    assert [(row.source_row_number, row.text, row.label) for row in prepared] == [
        (1, "Bagus sekali", "positive"),
        (5, "OTP tidak masuk", "negative"),
    ]
    assert report.input_rows == 5
    assert report.output_rows == 2
    assert report.missing_text_rows == 2
    assert report.duplicate_rows == 1


def test_prepare_rows_rejects_conflicting_normalized_labels():
    rows = [
        {"text": "Aplikasi lambat", "sentiment": "negative"},
        {"text": " aplikasi   lambat ", "sentiment": "positive"},
    ]

    with pytest.raises(PreprocessingError, match="conflicting labels"):
        prepare_labeled_rows(rows, text_column="text", label_column="sentiment")


def test_prepare_rows_rejects_missing_label():
    with pytest.raises(PreprocessingError, match="label is missing"):
        prepare_labeled_rows(
            [{"text": "feedback", "sentiment": None}],
            text_column="text",
            label_column="sentiment",
        )
