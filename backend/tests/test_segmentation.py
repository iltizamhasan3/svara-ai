from app.ai.segmentation import (
    SEGMENTATION_VERSION,
    segment_rows,
    segment_text,
    split_sentences,
)


def test_sentence_split_preserves_punctuation_and_avoids_decimal_and_abbreviation_splits():
    text = "Versi 2.1 stabil. Dr. Budi bilang cepat! Sangat bagus?"

    assert split_sentences(text) == [
        "Versi 2.1 stabil.",
        "Dr. Budi bilang cepat!",
        "Sangat bagus?",
    ]

    assert split_sentences("Disetujui a.n. Budi. Proses selesai.") == [
        "Disetujui a.n. Budi.",
        "Proses selesai.",
    ]


def test_clause_split_targets_contrastive_conjunctions_only():
    text = "Makanannya enak tetapi pelayanannya lambat dan tempatnya ramai."

    assert segment_text(text, split_clauses=True) == [
        "Makanannya enak",
        "pelayanannya lambat dan tempatnya ramai.",
    ]


def test_segment_rows_skips_missing_values_and_preserves_traceability():
    rows = [
        {"feedback": "Bagus. Cepat."},
        {"feedback": "  "},
        {"feedback": None},
        {"feedback": "Login gagal tetapi OTP juga lambat."},
    ]

    units = segment_rows(rows, "feedback", split_clauses=True)

    assert SEGMENTATION_VERSION == "segmentation-v1"
    assert [(unit.source_row_number, unit.unit_index, unit.text) for unit in units] == [
        (1, 1, "Bagus."),
        (1, 2, "Cepat."),
        (4, 1, "Login gagal"),
        (4, 2, "OTP juga lambat."),
    ]


def test_segment_rows_rejects_blank_feedback_column():
    try:
        segment_rows([], "   ")
    except ValueError as exc:
        assert str(exc) == "feedback_column is required"
    else:
        raise AssertionError("blank feedback columns must fail closed")
