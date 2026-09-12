#!/usr/bin/env python3
"""Download and verify the pinned Indonesian Google Play Review source.

The published source has binary labels. Preparation derives the project's
three-class target from the published star rating; this downloader validates
only the raw source contract and never reads IGAR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/manifests/week6_training_sources.json"
USER_AGENT = "svara-ai-week6-google-play-review-training/1.0"
SOURCE_KEY = "google_play_review"
SCHEMA = ["text", "label", "stars"]


def request(url: str):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
        timeout=60,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(manifest: dict[str, Any]) -> dict[str, Any]:
    source = manifest.get("sources", {}).get(SOURCE_KEY)
    if not isinstance(source, dict) or source.get("role") != "additional_training":
        raise ValueError("training manifest must identify Google Play Review as additional_training")
    if source.get("license") != "CC BY 4.0":
        raise ValueError("Google Play Review source must retain its CC BY 4.0 license")
    if source.get("schema") != SCHEMA:
        raise ValueError("Google Play Review schema is not pinned to text, label, stars")
    return source


def validate_google_play_review(path: Path, *, file_details: dict[str, Any], source: dict[str, Any]) -> int:
    """Validate one raw CSV's schema, labels, ratings, and row count."""

    raw_labels = {str(label).casefold() for label in source["raw_labels"]}
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SCHEMA:
            raise ValueError(f"{path}: schema mismatch: {reader.fieldnames}")
        for row in reader:
            count += 1
            if not row["text"].strip():
                raise ValueError(f"{path}: row {count} has empty text")
            if row["label"].strip().casefold() not in raw_labels:
                raise ValueError(f"{path}: row {count} has an unknown label")
            try:
                stars = int(row["stars"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: row {count} has a non-integer stars value") from error
            if stars not in {1, 2, 3, 4, 5}:
                raise ValueError(f"{path}: row {count} has stars outside 1..5")
    if count != file_details["expected_rows"]:
        raise ValueError(f"{path}: expected {file_details['expected_rows']} rows, got {count}")
    return count


def _download_file(url: str, destination: Path, *, file_details: dict[str, Any], source: dict[str, Any]) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".part",
        dir=destination.parent,
    )
    digest = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(temporary_fd, "wb") as output, request(url) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                total += len(chunk)
        if total != file_details["expected_size_bytes"]:
            raise ValueError(f"{destination}: expected {file_details['expected_size_bytes']} bytes, got {total}")
        if digest.hexdigest() != file_details["expected_sha256"]:
            raise ValueError(f"{destination}: downloaded SHA-256 does not match the pinned manifest")
        validate_google_play_review(
            Path(temporary_name),
            file_details=file_details,
            source=source,
        )
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def download_google_play_review(
    output_root: Path,
    *,
    manifest_path: Path = MANIFEST_PATH,
) -> list[Path]:
    """Download both pinned source files atomically after validation."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("training_policy", {}).get("igar_forbidden") is not True:
        raise ValueError("training manifest must explicitly forbid IGAR")
    source = _source(manifest)
    destinations: list[Path] = []
    for details in source["files"]:
        destination = output_root / str(details["name"])
        destinations.append(
            _download_file(str(details["url"]), destination, file_details=details, source=source)
        )
    return destinations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/raw/google_play_review",
        help="ignored directory for the raw source files",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return parser


if __name__ == "__main__":
    try:
        args = build_parser().parse_args()
        files = download_google_play_review(
            args.output_root,
            manifest_path=args.manifest,
        )
        for path in files:
            print(f"Downloaded and verified Google Play Review at {path}")
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
