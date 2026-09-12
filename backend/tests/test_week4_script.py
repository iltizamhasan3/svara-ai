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
