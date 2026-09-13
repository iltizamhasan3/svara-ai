#!/usr/bin/env python3
"""Validate and describe the Week 5 local sentiment inference export."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ai.model_bundle import (  # noqa: E402
    MODEL_NAME,
    MODEL_RELEASE_ASSET,
    MODEL_RELEASE_TAG,
    MODEL_RELEASE_URL,
    MODEL_V2_RELEASE_ASSET,
    MODEL_V2_RELEASE_TAG,
    MODEL_V2_RELEASE_URL,
    MODEL_REVISION,
    REQUIRED_INFERENCE_FILES,
    SENTIMENT_MODEL_VERSION,
    SENTIMENT_MODEL_V2_VERSION,
    SUPPORTED_MODEL_VERSIONS,
    validate_model_bundle,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _release_defaults(model_version: str) -> tuple[str, str, str]:
    if model_version == SENTIMENT_MODEL_VERSION:
        return MODEL_RELEASE_TAG, MODEL_RELEASE_URL, MODEL_RELEASE_ASSET
    if model_version == SENTIMENT_MODEL_V2_VERSION:
        return MODEL_V2_RELEASE_TAG, MODEL_V2_RELEASE_URL, MODEL_V2_RELEASE_ASSET
    raise ValueError(
        "model_version must be one of: "
        + ", ".join(sorted(SUPPORTED_MODEL_VERSIONS))
    )


def build_export_manifest(
    model_dir: Path,
    *,
    root: Path = ROOT,
    model_version: str = SENTIMENT_MODEL_VERSION,
    release_tag: str | None = None,
    release_url: str | None = None,
    release_asset: str | None = None,
) -> dict[str, Any]:
    """Return portable metadata and checksums for a validated model bundle."""

    default_tag, default_url, default_asset = _release_defaults(model_version)
    release_tag = default_tag if release_tag is None else release_tag
    release_url = default_url if release_url is None else release_url
    release_asset = default_asset if release_asset is None else release_asset
    bundle = validate_model_bundle(model_dir, verify_export_manifest=False)
    files = {
        filename: {
            "size_bytes": (bundle.path / filename).stat().st_size,
            "sha256": sha256_file(bundle.path / filename),
        }
        for filename in REQUIRED_INFERENCE_FILES
    }
    return {
        "manifest_version": 1,
        "model_version": model_version,
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "preprocessing_version": bundle.preprocessing_version,
        "label_mapping": dict(bundle.label_to_id),
        "inference": {
            "device": "cpu",
            "local_files_only": True,
            "required_files": list(REQUIRED_INFERENCE_FILES),
        },
        "bundle": {
            "path": _display_path(bundle.path, root=root),
            "files": files,
        },
        "release": {
            "tag": release_tag,
            "url": release_url,
            "asset": release_asset,
        },
        "external_evaluation": {
            "igar_only": True,
            "training_or_tuning": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "artifacts/week3/model-v1",
        help="validated local Transformers model directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/week5/sentiment_model_export_manifest.json",
    )
    parser.add_argument(
        "--replace-trust-manifest",
        action="store_true",
        help="explicitly replace an existing manifest during a deliberate release operation",
    )
    parser.add_argument("--model-version", default=SENTIMENT_MODEL_VERSION)
    parser.add_argument("--release-tag", default=None)
    parser.add_argument("--release-url", default=None)
    parser.add_argument("--release-asset", default=None)
    return parser


def main(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_export_manifest(
        args.model_dir.resolve(),
        root=ROOT,
        model_version=getattr(args, "model_version", SENTIMENT_MODEL_VERSION),
        release_tag=getattr(args, "release_tag", None),
        release_url=getattr(args, "release_url", None),
        release_asset=getattr(args, "release_asset", None),
    )
    output = args.output.resolve()
    if output.exists() and not getattr(args, "replace_trust_manifest", False):
        raise ValueError(
            f"{output}: trust manifest already exists; pass --replace-trust-manifest "
            "only for a deliberate release operation"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    try:
        main(build_parser().parse_args())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
