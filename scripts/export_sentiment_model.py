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
    MODEL_REVISION,
    REQUIRED_INFERENCE_FILES,
    SENTIMENT_MODEL_VERSION,
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


def build_export_manifest(model_dir: Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Return portable metadata and checksums for a validated model bundle."""

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
        "model_version": SENTIMENT_MODEL_VERSION,
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
            "tag": MODEL_RELEASE_TAG,
            "url": MODEL_RELEASE_URL,
            "asset": MODEL_RELEASE_ASSET,
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
    return parser


def main(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_export_manifest(args.model_dir.resolve(), root=ROOT)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    try:
        main(build_parser().parse_args())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
