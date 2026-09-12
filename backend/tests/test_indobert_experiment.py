import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
_UNSET = object()


def load_runner():
    spec = importlib.util.spec_from_file_location("indobert_runner", ROOT / "scripts/train_indobert.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_label_order_is_canonical():
    runner = load_runner()
    assert runner.CANONICAL_LABELS == ("positive", "neutral", "negative")
    assert runner.LABEL_TO_ID == {"positive": 0, "neutral": 1, "negative": 2}


def test_disjointness_guard():
    runner = load_runner()
    row = runner.PreparedRow(1, " Satu  teks ", "positive")
    with pytest.raises(ValueError, match="overlaps"):
        runner.assert_disjoint_splits({"train": [row], "validation": [runner.PreparedRow(2, "Satu teks", "positive")]})


def test_test_evaluation_requires_frozen_manifest():
    runner = load_runner()
    assert runner.test_evaluation_allowed({}, False) is False
    with pytest.raises(ValueError, match="test evaluation"):
        runner.test_evaluation_allowed({}, True)
    assert runner.test_evaluation_allowed({"test_evaluation_frozen": True}, True) is True


def test_igar_training_path_is_rejected():
    runner = load_runner()

    with pytest.raises(ValueError, match="sealed external-test"):
        runner.assert_not_igar_training_path(ROOT / "data/raw/igar/Rating_labeled.csv")


def test_freeze_encoder_keeps_only_classifier_head_trainable():
    runner = load_runner()

    class Parameter:
        def __init__(self):
            self.requires_grad = True

    class Model:
        def __init__(self):
            self.parameters_by_name = {
                "bert.embeddings.weight": Parameter(),
                "bert.encoder.weight": Parameter(),
                "classifier.weight": Parameter(),
                "classifier.bias": Parameter(),
            }

        def named_parameters(self):
            return self.parameters_by_name.items()

    model = Model()
    assert runner.configure_trainable_parameters(model, freeze_encoder=True) == {
        "trainable": 2,
        "frozen": 2,
    }
    assert model.parameters_by_name["bert.encoder.weight"].requires_grad is False
    assert model.parameters_by_name["classifier.weight"].requires_grad is True


def test_additional_training_manifest_is_loaded_and_merged(tmp_path):
    runner = load_runner()
    extra_dir = tmp_path / "idsmsa"
    extra_dir.mkdir()
    contents = {
        "idsmsa_train.tsv": "tambahan positif\tpositive\n",
        "idsmsa_validation.tsv": "tambahan netral\tneutral\n",
        "idsmsa_test.tsv": "tambahan negatif\tnegative\n",
    }
    split_details = {}
    for filename, content in contents.items():
        path = extra_dir / filename
        path.write_text(content, encoding="utf-8")
        split = filename.removeprefix("idsmsa_").removesuffix(".tsv")
        split_details[split] = {
            "file": filename,
            "sha256": runner.sha256_file(path),
            "rows": 1,
        }
    manifest_path = extra_dir / "idsmsa_split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "ID-SMSA",
                "training_policy": {
                    "igar_forbidden": True,
                    "igar_labels_used": False,
                    "igar_metrics_used": False,
                },
                "splits": split_details,
            }
        ),
        encoding="utf-8",
    )

    additional, _ = runner.load_additional_splits(manifest_path)
    primary = {
        "train": [runner.PreparedRow(1, "primary", "positive")],
        "validation": [runner.PreparedRow(2, "validation", "neutral")],
        "test": [runner.PreparedRow(3, "test", "negative")],
    }
    report = runner.merge_additional_training(primary, additional)

    assert report == {
        "source_train_rows": 1,
        "accepted_train_rows": 1,
        "duplicate_rows_removed": 0,
        "conflicting_rows": 0,
    }
    assert [row.text for row in primary["train"]] == ["primary", "tambahan positif"]


def test_additional_only_requires_an_additional_manifest():
    runner = load_runner()
    args = runner.build_parser().parse_args(["--additional-only"])

    assert args.additional_only is True
    assert args.learning_rate == pytest.approx(2e-5)


def _fixture_manifest(tmp_path: Path, runner, *, checksum_override=_UNSET):
    data_dir = tmp_path / "selected-smsa"
    data_dir.mkdir()
    contents = {
        "train_preprocess.tsv": "train positif\tPositive\ntrain netral\tNeutral\ntrain negatif\tNegative\n",
        "valid_preprocess.tsv": "valid positif\tPositive\nvalid netral\tNeutral\nvalid negatif\tNegative\n",
        "test_preprocess.tsv": "test positif\tPositive\ntest netral\tNeutral\ntest negatif\tNegative\n",
    }
    split_details = {}
    for filename, content in contents.items():
        path = data_dir / filename
        path.write_text(content, encoding="utf-8")
        split = filename.split("_")[0]
        split = "validation" if split == "valid" else split
        split_details[split] = {
            "file": f"data/raw/smsa/{filename}",
            "sha256": runner.sha256_file(path),
        }
    if checksum_override is not _UNSET:
        split_details["train"]["sha256"] = checksum_override
    manifest_path = tmp_path / "split-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "SmSA",
                "preprocessing_version": runner.PREPROCESSING_VERSION,
                "splits": split_details,
            }
        ),
        encoding="utf-8",
    )
    return data_dir, manifest_path


def test_loader_uses_selected_data_directory(tmp_path, monkeypatch):
    runner = load_runner()
    data_dir, manifest_path = _fixture_manifest(tmp_path, runner)
    monkeypatch.chdir(tmp_path)
    splits, _ = runner.load_splits(data_dir, manifest_path)
    assert {split: len(rows) for split, rows in splits.items()} == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }


@pytest.mark.parametrize("checksum", [None, "not-a-sha256", "0" * 63])
def test_loader_rejects_missing_or_malformed_checksum(tmp_path, checksum):
    runner = load_runner()
    data_dir, manifest_path = _fixture_manifest(tmp_path, runner, checksum_override=checksum)
    with pytest.raises(ValueError, match="64-character SHA-256"):
        runner.load_splits(data_dir, manifest_path)
