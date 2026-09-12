import importlib.util
import json
from pathlib import Path

import pytest

from app.ai.model_bundle import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    REQUIRED_INFERENCE_FILES,
    ModelBundleError,
    validate_model_bundle,
)


ROOT = Path(__file__).parents[2]


def _write_fake_bundle(tmp_path: Path, *, mapping=None) -> Path:
    bundle = tmp_path / "model-v1"
    bundle.mkdir()
    for filename in REQUIRED_INFERENCE_FILES:
        (bundle / filename).write_bytes(b"placeholder")
    id_to_label = mapping or ID_TO_LABEL
    label_to_id = {label: identifier for identifier, label in id_to_label.items()}
    (bundle / "config.json").write_text(
        json.dumps(
            {
                "id2label": {str(identifier): label for identifier, label in id_to_label.items()},
                "label2id": label_to_id,
                "num_labels": len(id_to_label),
            }
        ),
        encoding="utf-8",
    )
    return bundle


def test_validate_model_bundle_requires_canonical_inference_files(tmp_path):
    bundle = _write_fake_bundle(tmp_path)

    validated = validate_model_bundle(bundle)

    assert validated.path == bundle.resolve()
    assert dict(validated.label_to_id) == LABEL_TO_ID
    assert dict(validated.id_to_label) == ID_TO_LABEL


def test_validate_model_bundle_rejects_missing_file(tmp_path):
    bundle = _write_fake_bundle(tmp_path)
    (bundle / "model.safetensors").unlink()

    with pytest.raises(ModelBundleError, match="model.safetensors"):
        validate_model_bundle(bundle)


def test_validate_model_bundle_rejects_noncanonical_mapping(tmp_path):
    bundle = _write_fake_bundle(tmp_path, mapping={0: "negative", 1: "neutral", 2: "positive"})

    with pytest.raises(ModelBundleError, match="canonical order"):
        validate_model_bundle(bundle)


def _load_exporter():
    spec = importlib.util.spec_from_file_location(
        "sentiment_exporter", ROOT / "scripts/export_sentiment_model.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_manifest_uses_relative_bundle_path(tmp_path):
    bundle = _write_fake_bundle(tmp_path)
    exporter = _load_exporter()

    manifest = exporter.build_export_manifest(bundle, root=tmp_path)

    assert manifest["bundle"]["path"] == "model-v1"
    assert manifest["label_mapping"] == LABEL_TO_ID
    assert set(manifest["bundle"]["files"]) == set(REQUIRED_INFERENCE_FILES)
    assert all(len(details["sha256"]) == 64 for details in manifest["bundle"]["files"].values())
