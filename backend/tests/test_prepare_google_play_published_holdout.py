import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def load_preparer():
    spec = importlib.util.spec_from_file_location(
        "published_holdout_preparer", ROOT / "scripts/prepare_google_play_published_holdout.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_source(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["text", "label", "stars"])
        writer.writerows(rows)


def write_manifest(path, source_path, rows):
    path.write_text(json.dumps({"sources": {"google_play_review": {
        "schema": ["text", "label", "stars"], "files": [{
            "name": "validation.csv", "expected_rows": len(rows),
            "expected_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }]
    }}}), encoding="utf-8")


def test_derivation_and_deterministic_duplicate_conflict_filtering(tmp_path):
    p = load_preparer()
    rows = [["bagus", "pos", "5"], [" bagus ", "pos", "4"], ["biasa", "neg", "3"],
            ["Biasa", "neg", "3"], ["campur", "neg", "1"], ["campur", "pos", "5"], ["", "pos", "5"]]
    source = tmp_path / "validation.csv"
    write_source(source, rows)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, source, rows)
    prepared, report = p.load_published_rows(source, manifest)
    assert prepared == [{"text": "bagus", "label": "positive"}, {"text": "biasa", "label": "neutral"}]
    assert report["duplicate_rows"] == 2
    assert report["conflicting_rows_removed"] == 2
    assert report["missing_text_rows"] == 1


def test_checksum_and_row_count_are_enforced(tmp_path):
    p = load_preparer()
    source = tmp_path / "validation.csv"
    rows = [["ok", "pos", "5"]]
    write_source(source, rows)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, source, rows)
    source.write_text(source.read_text(encoding="utf-8").replace("ok", "changed"), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        p.load_published_rows(source, manifest)


def test_path_guard_rejects_external_test_looking_paths(tmp_path):
    p = load_preparer()
    with pytest.raises(ValueError):
        p.assert_not_igar_input_path(tmp_path / "igar" / "input.csv")
    with pytest.raises(ValueError):
        p.assert_not_igar_input_path(tmp_path / "Rating_labeled.csv")


def test_prepare_has_no_external_test_dependency_and_writes_policy_manifest(tmp_path):
    p = load_preparer()
    rows = [["ok", "pos", "5"], ["biasa", "neg", "3"]]
    source = tmp_path / "validation.csv"
    write_source(source, rows)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, source, rows)
    output = tmp_path / "holdout.tsv"
    output_manifest = tmp_path / "holdout.json"
    result = p.prepare(validation_path=source, training_manifest_path=manifest,
                       output_tsv=output, output_manifest=output_manifest,
                       protected_paths=[], exclude_overlaps=True)
    saved = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert result["rows"] == 2
    assert saved["policy"] == {"confirmation_only": True, "for_training": False, "for_selection": False}
    assert saved["exclusion"]["protected_inputs"] == []
    assert output.read_text(encoding="utf-8") == "ok\tpositive\nbiasa\tneutral\n"
