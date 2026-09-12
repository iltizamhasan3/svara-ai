"""Validated loading metadata for the exported Week 3 sentiment model."""

from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.ai.preprocessing import CANONICAL_LABELS, PREPROCESSING_VERSION


SENTIMENT_MODEL_VERSION = "sentiment-model-v1"
MODEL_NAME = "indobenchmark/indobert-base-p1"
MODEL_REVISION = "c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2"
LABEL_TO_ID = {label: index for index, label in enumerate(CANONICAL_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
MODEL_RELEASE_TAG = "ai-model-v1.0.0"
MODEL_RELEASE_URL = (
    "https://github.com/iltizamhasan3/svara-ai/releases/tag/ai-model-v1.0.0"
)
MODEL_RELEASE_ASSET = "svara-ai-sentiment-model-v1.tar.gz"
DEFAULT_EXPORT_MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "artifacts/week5/sentiment_model_export_manifest.json"
)

# These are the files required by Transformers for local CPU inference. The
# training-only ``training_args.bin`` file is intentionally not required.
REQUIRED_INFERENCE_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


class ModelBundleError(ValueError):
    """Raised when a sentiment model export is incomplete or incompatible."""


@dataclass(frozen=True)
class SentimentModelBundle:
    """Immutable metadata for a validated local inference bundle."""

    path: Path
    model_version: str
    model_name: str
    model_revision: str
    preprocessing_version: str
    label_to_id: Mapping[str, int]
    id_to_label: Mapping[int, str]
    required_files: tuple[str, ...]


@dataclass(frozen=True)
class LoadedSentimentModel:
    """Loaded tokenizer/model pair ready for the batch inference adapter."""

    bundle: SentimentModelBundle
    tokenizer: Any
    model: Any
    device: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_label_mapping(config: Mapping[str, Any]) -> tuple[dict[str, int], dict[int, str]]:
    raw_id_to_label = config.get("id2label")
    raw_label_to_id = config.get("label2id")
    if not isinstance(raw_id_to_label, Mapping) or not isinstance(raw_label_to_id, Mapping):
        raise ModelBundleError("config.json must include id2label and label2id mappings")

    try:
        id_to_label = {
            int(identifier): str(label).casefold()
            for identifier, label in raw_id_to_label.items()
        }
        label_to_id = {
            str(label).casefold(): int(identifier)
            for label, identifier in raw_label_to_id.items()
        }
    except (TypeError, ValueError) as exc:
        raise ModelBundleError("config.json contains malformed label mappings") from exc

    return label_to_id, id_to_label


def _verify_export_manifest(bundle_path: Path, manifest_path: Path) -> None:
    """Verify bundle identity and bytes against a trusted export manifest."""

    if not manifest_path.is_file():
        raise ModelBundleError(f"trusted export manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleError(f"could not read trusted export manifest: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise ModelBundleError("trusted export manifest must contain a JSON object")

    expected_identity = {
        "model_version": SENTIMENT_MODEL_VERSION,
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "label_mapping": LABEL_TO_ID,
    }
    for field, expected in expected_identity.items():
        if manifest.get(field) != expected:
            raise ModelBundleError(
                f"trusted export manifest {field!r} does not match the Week 5 model identity"
            )

    bundle_metadata = manifest.get("bundle")
    manifest_files = bundle_metadata.get("files") if isinstance(bundle_metadata, Mapping) else None
    if not isinstance(manifest_files, Mapping):
        raise ModelBundleError("trusted export manifest must include bundle file checksums")

    for filename in REQUIRED_INFERENCE_FILES:
        file_metadata = manifest_files.get(filename)
        if not isinstance(file_metadata, Mapping):
            raise ModelBundleError(
                f"trusted export manifest has no checksum for {filename}"
            )
        expected_size = file_metadata.get("size_bytes")
        expected_sha256 = file_metadata.get("sha256")
        actual_path = bundle_path / filename
        if expected_size != actual_path.stat().st_size:
            raise ModelBundleError(
                f"model bundle size mismatch for {filename}; trusted export does not match"
            )
        if expected_sha256 != _sha256_file(actual_path):
            raise ModelBundleError(
                f"model bundle checksum mismatch for {filename}; trusted export does not match"
            )


def validate_model_bundle(
    model_dir: Path | str,
    *,
    manifest_path: Path | str | None = None,
    verify_export_manifest: bool = True,
) -> SentimentModelBundle:
    """Validate files, labels, and (by default) the trusted export manifest.

    ``verify_export_manifest=False`` is reserved for the exporter while it is
    creating a new manifest and for isolated unit fixtures. Runtime loading
    keeps verification enabled by default.
    """

    path = Path(model_dir).expanduser().resolve()
    if not path.is_dir():
        raise ModelBundleError(f"model bundle directory does not exist: {path}")

    missing = [filename for filename in REQUIRED_INFERENCE_FILES if not (path / filename).is_file()]
    if missing:
        raise ModelBundleError(
            "model bundle is missing required inference files: " + ", ".join(missing)
        )

    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleError(f"could not read model config: {path / 'config.json'}") from exc
    if not isinstance(config, Mapping):
        raise ModelBundleError("config.json must contain a JSON object")

    label_to_id, id_to_label = _config_label_mapping(config)
    if label_to_id != LABEL_TO_ID or id_to_label != ID_TO_LABEL:
        raise ModelBundleError(
            "model label mapping does not match canonical order "
            f"{LABEL_TO_ID}"
        )
    if verify_export_manifest:
        trusted_manifest = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path is not None
            else DEFAULT_EXPORT_MANIFEST
        )
        _verify_export_manifest(path, trusted_manifest)

    return SentimentModelBundle(
        path=path,
        model_version=SENTIMENT_MODEL_VERSION,
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        preprocessing_version=PREPROCESSING_VERSION,
        label_to_id=dict(LABEL_TO_ID),
        id_to_label=dict(ID_TO_LABEL),
        required_files=REQUIRED_INFERENCE_FILES,
    )


def load_sentiment_model(
    model_dir: Path | str,
    *,
    device: str = "cpu",
    local_files_only: bool = True,
    torch_threads: int | None = 4,
    export_manifest_path: Path | str | None = None,
) -> LoadedSentimentModel:
    """Load a validated Transformers bundle without implicit model downloads."""

    if device != "cpu":
        raise ModelBundleError("Week 5 sentiment inference supports device='cpu' only")
    if torch_threads is not None and torch_threads < 1:
        raise ModelBundleError("torch_threads must be positive when provided")

    bundle = validate_model_bundle(model_dir, manifest_path=export_manifest_path)

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ModelBundleError(
            "install the backend AI extra before loading the sentiment model"
        ) from exc

    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(bundle.path),
            local_files_only=local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(bundle.path),
            local_files_only=local_files_only,
        )
    except (OSError, ValueError) as exc:
        raise ModelBundleError(f"could not load sentiment model bundle: {bundle.path}") from exc

    runtime_id_to_label = {
        int(identifier): str(label).casefold()
        for identifier, label in getattr(model.config, "id2label", {}).items()
    }
    if getattr(model.config, "num_labels", None) != len(CANONICAL_LABELS):
        raise ModelBundleError(
            "loaded sentiment model must expose exactly three labels; "
            f"got {getattr(model.config, 'num_labels', None)!r}"
        )
    if runtime_id_to_label != ID_TO_LABEL:
        raise ModelBundleError("loaded sentiment model label mapping is incompatible")

    model = model.to(device)
    model.eval()
    return LoadedSentimentModel(
        bundle=bundle,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
