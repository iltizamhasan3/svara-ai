import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[2]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "week4_topic_runner",
        ROOT / "scripts/run_week4_topic_discovery.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_input_rows_supports_headered_csv(tmp_path):
    runner = load_runner()
    path = tmp_path / "reviews.csv"
    path.write_text("content,score\nBagus,5\nLambat,1\n", encoding="utf-8")

    assert runner.read_input_rows(path, text_column="content") == [
        {"content": "Bagus", "score": "5"},
        {"content": "Lambat", "score": "1"},
    ]


def test_read_input_rows_supports_headerless_tsv(tmp_path):
    runner = load_runner()
    path = tmp_path / "reviews.tsv"
    path.write_text("Bagus\tpositive\nLambat\tnegative\n", encoding="utf-8")

    assert runner.read_input_rows(path, text_column="sentence", no_header=True) == [
        {"sentence": "Bagus"},
        {"sentence": "Lambat"},
    ]


def test_completed_manual_review_is_preserved_for_matching_topics(tmp_path):
    runner = load_runner()
    path = tmp_path / "manual_evaluation.csv"
    path.write_text(
        "topic_id,unit_count,keywords,representative_texts,manual_judgment,review_notes\n"
        "0,2,login,\"[]\",useful,coherent\n"
        "-1,1,,,outlier,traceable\n",
        encoding="utf-8",
    )

    assert runner.review_has_manual_entries(path) is True
    assert runner.should_preserve_review(path, {0}) is True


def test_completed_manual_review_rejects_stale_topics(tmp_path):
    runner = load_runner()
    path = tmp_path / "manual_evaluation.csv"
    path.write_text(
        "topic_id,unit_count,keywords,representative_texts,manual_judgment,review_notes\n"
        "0,2,login,\"[]\",useful,coherent\n",
        encoding="utf-8",
    )

    try:
        runner.should_preserve_review(path, {1})
    except ValueError as exc:
        assert "new output directory" in str(exc)
    else:
        raise AssertionError("stale manual reviews must fail closed")


def test_external_artifact_paths_are_serializable(tmp_path):
    runner = load_runner()
    external_path = (tmp_path / "week4" / "embeddings.npy").resolve()

    assert runner.display_path(external_path) == str(external_path)
