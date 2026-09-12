import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def load_preparer():
    spec = importlib.util.spec_from_file_location(
        "week6_google_play_preparer", ROOT / "scripts/prepare_week6_google_play.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rating_to_three_class_mapping():
    preparer = load_preparer()
    assert preparer._derived_label("neg", "1", row_number=1) == "negative"
    assert preparer._derived_label("neg", "3", row_number=2) == "neutral"
    assert preparer._derived_label("pos", "5", row_number=3) == "positive"

    with pytest.raises(ValueError, match="inconsistent"):
        preparer._derived_label("pos", "3", row_number=4)


def test_loader_excludes_ambiguous_normalized_texts(tmp_path):
    preparer = load_preparer()
    path = tmp_path / "train.csv"
    rows = [
        ["bagus", "pos", "5"],
        [" bagus ", "pos", "4"],
        ["biasa", "neg", "3"],
        ["Biasa", "neg", "3"],
        ["campur", "neg", "1"],
        ["campur", "pos", "5"],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(preparer.SCHEMA)
        writer.writerows(rows)

    manifest = tmp_path / "manifest.json"
    payload = path.read_bytes()
    manifest.write_text(
        json.dumps(
            {
                "training_policy": {"igar_forbidden": True},
                "sources": {
                    "google_play_review": {
                        "role": "additional_training",
                        "derived_label_rule": {},
                        "files": [
                            {
                                "name": "train.csv",
                                "expected_rows": len(rows),
                                "expected_sha256": hashlib.sha256(payload).hexdigest(),
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    prepared, report = preparer.load_google_play_rows(path, manifest)

    assert [(row.text, row.label) for row in prepared] == [
        ("bagus", "positive"),
        ("biasa", "neutral"),
    ]
    assert report["duplicate_rows"] == 2
    assert report["ambiguous_text_groups"] == 1
    assert report["ambiguous_rows_removed"] == 2


def test_google_play_path_is_rejected_as_igar_training_input():
    preparer = load_preparer()
    with pytest.raises(ValueError, match="sealed external-test"):
        preparer.assert_not_igar_training_path(ROOT / "data/raw/igar/Rating_labeled.csv")
