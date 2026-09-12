#!/usr/bin/env python3
"""Download and verify the pinned ID-SMSA training source.

ID-SMSA is an additional training source. IGAR is intentionally absent from
this manifest and must remain a sealed external test source.
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


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/manifests/week6_training_sources.json"
USER_AGENT = "svara-ai-week6-idsmsa-training/1.0"


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


def _source_details(manifest: dict[str, object]) -> dict[str, object]:
    source = manifest["sources"]["idsmsa"]
    if not isinstance(source, dict) or source.get("role") != "additional_training":
        raise ValueError("training manifest must identify ID-SMSA as additional_training")

    metadata_request = request(str(source["metadata_url"]))
    with metadata_request as response:
        metadata = json.loads(response.read().decode("utf-8"))
    metadata_doi = metadata.get("doi")
    if isinstance(metadata_doi, dict):
        metadata_doi = metadata_doi.get("id")
    if metadata_doi != source["doi"]:
        raise ValueError("ID-SMSA DOI metadata mismatch")
    license_details = metadata.get("data_licence")
    if not isinstance(license_details, dict) or license_details.get("short_name") != source["license"]:
        raise ValueError("ID-SMSA license metadata mismatch")

    files_request = request(str(source["files_url"]))
    with files_request as response:
        files = json.loads(response.read().decode("utf-8"))
    source_file = next(
        (item for item in files if item.get("filename") == source["file_name"]),
        None,
    )
    if source_file is None:
        raise ValueError(f"{source['file_name']} not found in Mendeley file list")
    details = source_file.get("content_details")
    if not isinstance(details, dict):
        raise ValueError("ID-SMSA file metadata is malformed")
    if (
        details.get("size") != source["expected_size_bytes"]
        or details.get("sha256_hash") != source["expected_sha256"]
    ):
        raise ValueError("ID-SMSA file provenance mismatch")
    return details


def validate_idsmsa(path: Path, *, manifest: dict[str, object]) -> int:
    """Validate the source schema, labels, and expected row count."""

    source = manifest["sources"]["idsmsa"]
    assert isinstance(source, dict)
    expected_schema = source["schema"]
    labels = {str(label).casefold() for label in source["labels"]}
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_schema:
            raise ValueError(f"{path}: schema mismatch: {reader.fieldnames}")
        for row in reader:
            count += 1
            if not row["Sentence"].strip():
                raise ValueError(f"{path}: row {count} has an empty Sentence")
            if row["Sentiment"].strip().casefold() not in labels:
                raise ValueError(f"{path}: row {count} has an unknown Sentiment")
    if count != source["expected_rows"]:
        raise ValueError(f"{path}: expected {source['expected_rows']} rows, got {count}")
    return count


def download_idsmsa(destination: Path, *, manifest_path: Path = MANIFEST_PATH) -> Path:
    """Stream ID-SMSA to an atomic destination after provenance checks."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    details = _source_details(manifest)
    source = manifest["sources"]["idsmsa"]
    assert isinstance(source, dict)
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
        with os.fdopen(temporary_fd, "wb") as output, request(str(details["download_url"])) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                total += len(chunk)
        if total != source["expected_size_bytes"]:
            raise ValueError(f"expected {source['expected_size_bytes']} bytes, got {total}")
        if digest.hexdigest() != source["expected_sha256"]:
            raise ValueError("downloaded ID-SMSA SHA-256 does not match the pinned manifest")
        validate_idsmsa(Path(temporary_name), manifest=manifest)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/raw/idsmsa/IDSMSA.csv",
        help="ID-SMSA CSV destination (ignored by Git)",
    )
    return parser


if __name__ == "__main__":
    try:
        destination = download_idsmsa(build_parser().parse_args().output)
        print(f"Downloaded and verified ID-SMSA at {destination}")
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
