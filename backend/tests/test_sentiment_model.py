import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
import app.ai.model_bundle as model_bundle

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

    validated = validate_model_bundle(bundle, verify_export_manifest=False)

    assert validated.path == bundle.resolve()
    assert dict(validated.label_to_id) == LABEL_TO_ID
    assert dict(validated.id_to_label) == ID_TO_LABEL


def test_validate_model_bundle_rejects_missing_file(tmp_path):
    bundle = _write_fake_bundle(tmp_path)
    (bundle / "model.safetensors").unlink()

    with pytest.raises(ModelBundleError, match="model.safetensors"):
        validate_model_bundle(bundle, verify_export_manifest=False)


def test_validate_model_bundle_rejects_noncanonical_mapping(tmp_path):
    bundle = _write_fake_bundle(tmp_path, mapping={0: "negative", 1: "neutral", 2: "positive"})

    with pytest.raises(ModelBundleError, match="canonical order"):
        validate_model_bundle(bundle, verify_export_manifest=False)


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


def test_validate_model_bundle_rejects_modified_file_against_trusted_manifest(tmp_path):
    bundle = _write_fake_bundle(tmp_path)
    exporter = _load_exporter()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(exporter.build_export_manifest(bundle, root=tmp_path)),
        encoding="utf-8",
    )

    validate_model_bundle(bundle, manifest_path=manifest_path)
    (bundle / "model.safetensors").write_bytes(b"replacement")

    with pytest.raises(ModelBundleError, match="checksum mismatch"):
        validate_model_bundle(bundle, manifest_path=manifest_path)


def test_validate_model_bundle_accepts_explicit_v2_manifest(tmp_path):
    bundle = _write_fake_bundle(tmp_path)
    exporter = _load_exporter()
    manifest_path = tmp_path / "manifest-v2.json"
    manifest_path.write_text(
        json.dumps(
            exporter.build_export_manifest(
                bundle,
                root=tmp_path,
                model_version=model_bundle.SENTIMENT_MODEL_V2_VERSION,
                release_tag="ai-model-v2.0.0",
                release_url="https://example.test/releases/ai-model-v2.0.0",
                release_asset="svara-ai-sentiment-model-v2.tar.gz",
            )
        ),
        encoding="utf-8",
    )

    validated = validate_model_bundle(bundle, manifest_path=manifest_path)

    assert validated.model_version == "sentiment-model-v2"


def test_export_manifest_keeps_v1_defaults_and_accepts_release_overrides(tmp_path):
    bundle = _write_fake_bundle(tmp_path)
    exporter = _load_exporter()

    defaults = exporter.build_export_manifest(bundle, root=tmp_path)
    custom = exporter.build_export_manifest(
        bundle,
        root=tmp_path,
        model_version=model_bundle.SENTIMENT_MODEL_V2_VERSION,
        release_tag="v2",
        release_url="https://example.test/v2",
        release_asset="model-v2.tar.gz",
    )

    assert defaults["model_version"] == "sentiment-model-v1"
    assert defaults["release"]["tag"] == exporter.MODEL_RELEASE_TAG
    assert custom["model_version"] == "sentiment-model-v2"
    assert custom["release"] == {
        "tag": "v2",
        "url": "https://example.test/v2",
        "asset": "model-v2.tar.gz",
    }

    v2_defaults = exporter.build_export_manifest(
        bundle,
        root=tmp_path,
        model_version=model_bundle.SENTIMENT_MODEL_V2_VERSION,
    )
    assert v2_defaults["release"] == {
        "tag": exporter.MODEL_V2_RELEASE_TAG,
        "url": exporter.MODEL_V2_RELEASE_URL,
        "asset": exporter.MODEL_V2_RELEASE_ASSET,
    }


def test_exporter_model_version_only_override_uses_matching_release_defaults(tmp_path):
    bundle = _write_fake_bundle(tmp_path)
    exporter = _load_exporter()
    output = tmp_path / "manifest-v2.json"

    manifest = exporter.main(
        Namespace(
            model_dir=bundle,
            output=output,
            replace_trust_manifest=False,
            model_version=model_bundle.SENTIMENT_MODEL_V2_VERSION,
        )
    )

    assert manifest["model_version"] == model_bundle.SENTIMENT_MODEL_V2_VERSION
    assert manifest["release"]["tag"] == exporter.MODEL_V2_RELEASE_TAG
    assert manifest["release"]["url"] == exporter.MODEL_V2_RELEASE_URL
    assert manifest["release"]["asset"] == exporter.MODEL_V2_RELEASE_ASSET


def test_validate_model_bundle_rejects_malformed_manifest_version(tmp_path):
    bundle = _write_fake_bundle(tmp_path)
    exporter = _load_exporter()
    manifest = exporter.build_export_manifest(
        bundle,
        root=tmp_path,
        model_version=model_bundle.SENTIMENT_MODEL_V2_VERSION,
    )
    manifest_path = tmp_path / "malformed-manifest.json"

    for malformed_version in ([], {}):
        manifest["model_version"] = malformed_version
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ModelBundleError, match="model_version.*string"):
            validate_model_bundle(bundle, manifest_path=manifest_path)


def test_default_trust_anchor_does_not_opt_into_v2(tmp_path, monkeypatch):
    bundle = _write_fake_bundle(tmp_path)
    exporter = _load_exporter()
    manifest_path = tmp_path / "manifest-v2.json"
    manifest_path.write_text(
        json.dumps(
            exporter.build_export_manifest(
                bundle,
                root=tmp_path,
                model_version=model_bundle.SENTIMENT_MODEL_V2_VERSION,
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_bundle, "DEFAULT_EXPORT_MANIFEST", manifest_path)

    with pytest.raises(ModelBundleError, match="unsupported"):
        validate_model_bundle(bundle)


def test_validate_model_bundle_rejects_unsupported_explicit_version(tmp_path):
    bundle = _write_fake_bundle(tmp_path)
    exporter = _load_exporter()
    manifest = exporter.build_export_manifest(bundle, root=tmp_path)
    manifest["model_version"] = "sentiment-model-v3"
    manifest_path = tmp_path / "manifest-v3.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelBundleError, match="unsupported"):
        validate_model_bundle(bundle, manifest_path=manifest_path)


def test_validate_model_bundle_rejects_tampered_explicit_v2_manifest(tmp_path):
    bundle = _write_fake_bundle(tmp_path)
    exporter = _load_exporter()
    manifest_path = tmp_path / "manifest-v2.json"
    manifest_path.write_text(
        json.dumps(
            exporter.build_export_manifest(
                bundle,
                root=tmp_path,
                model_version=model_bundle.SENTIMENT_MODEL_V2_VERSION,
            )
        ),
        encoding="utf-8",
    )
    (bundle / "tokenizer.json").write_bytes(b"tampered!!!")

    with pytest.raises(ModelBundleError, match="checksum mismatch"):
        validate_model_bundle(bundle, manifest_path=manifest_path)


def test_exporter_requires_explicit_trust_manifest_replacement(tmp_path):
    bundle = _write_fake_bundle(tmp_path)
    exporter = _load_exporter()
    output = tmp_path / "manifest.json"
    output.write_text("existing trust anchor\n", encoding="utf-8")

    args = Namespace(model_dir=bundle, output=output, replace_trust_manifest=False)
    with pytest.raises(ValueError, match="replace-trust-manifest"):
        exporter.main(args)

    args.replace_trust_manifest = True
    exporter.main(args)
    assert json.loads(output.read_text(encoding="utf-8"))["manifest_version"] == 1
