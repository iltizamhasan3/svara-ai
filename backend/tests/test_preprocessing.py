import math

import pytest

from app.ai.preprocessing import (
    PREPROCESSING_VERSION,
    PreprocessingError,
    normalize_label,
    normalize_text,
    prepare_inference_rows,
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


def test_prepare_inference_rows_normalizes_skips_missing_and_preserves_traceability():
    rows = [
        {"feedback": "  Bagus\nsekali  "},
        {"feedback": None},
        {"feedback": "bagus sekali"},
        {"feedback": "  OTP masuk 😭!! "},
    ]

    prepared, report = prepare_inference_rows(rows, text_column="feedback")

    assert [(row.source_row_number, row.text) for row in prepared] == [
        (1, "Bagus sekali"),
        (3, "bagus sekali"),
        (4, "OTP masuk 😭!!"),
    ]
    assert report.input_rows == 4
    assert report.output_rows == 3
    assert report.missing_text_rows == 1
    assert report.duplicate_rows == 1


def test_prepare_inference_rows_validates_text_column():
    with pytest.raises(PreprocessingError, match="text_column is required"):
        prepare_inference_rows([{"feedback": "teks"}], text_column=" ")

    with pytest.raises(PreprocessingError, match="text column 'feedback' is missing"):
        prepare_inference_rows([{"other": "teks"}], text_column="feedback")
