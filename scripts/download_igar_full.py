#!/usr/bin/env python3
"""Download and verify the pinned full IGAR source for external testing only.

The full IGAR payload is deliberately written below ``data/raw/igar/``. That
directory is ignored by Git and must never be passed to a training or tuning
workflow. The tracked 30-row sample remains the default Week 5 evaluation
input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/manifests/week2_sources.json"
USER_AGENT = "svara-ai-igar-external-test/1.0"
IGAR_TEST_ROOT = ROOT / "data/raw/igar"


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


def _assert_test_destination(destination: Path) -> Path:
    """Reject destinations outside the ignored full-IGAR test directory."""

    resolved = destination.resolve()
    try:
        resolved.relative_to(IGAR_TEST_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            "full IGAR is test-only and must be stored below data/raw/igar/"
        ) from error
    return resolved


def _source_details(manifest: dict[str, object]) -> dict[str, object]:
    source = manifest["sources"]["igar"]
    assert isinstance(source, dict)
    metadata_url = source["metadata_url"]
    files_url = source["files_url"]
    metadata_request = request(str(metadata_url))
    with metadata_request as response:
        metadata = json.loads(response.read().decode("utf-8"))
    metadata_doi = metadata.get("doi")
    if isinstance(metadata_doi, dict):
        metadata_doi = metadata_doi.get("id")
    if metadata_doi != source["doi"]:
        raise ValueError("IGAR DOI metadata mismatch")

    files_request = request(str(files_url))
    with files_request as response:
        files = json.loads(response.read().decode("utf-8"))
    source_file = next(
        (item for item in files if item.get("filename") == source["file_name"]),
        None,
    )
    if source_file is None:
        raise ValueError(f"{source['file_name']} not found in Mendeley file list")
    details = source_file["content_details"]
    if (
        details.get("size") != source["expected_size_bytes"]
        or details.get("sha256_hash") != source["expected_sha256"]
    ):
        raise ValueError("IGAR file provenance mismatch")
    return details


def download_full_igar(destination: Path, *, manifest_path: Path = MANIFEST_PATH) -> Path:
    """Stream the full source to an atomic destination and verify its checksum."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    details = _source_details(manifest)
    destination = _assert_test_destination(destination)
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
        expected_size = manifest["sources"]["igar"]["expected_size_bytes"]
        expected_sha256 = manifest["sources"]["igar"]["expected_sha256"]
        if total != expected_size:
            raise ValueError(f"expected {expected_size} bytes, got {total}")
        if digest.hexdigest() != expected_sha256:
            raise ValueError("downloaded IGAR SHA-256 does not match the pinned manifest")
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
        default=ROOT / "data/raw/igar/Rating_labeled.csv",
        help="full IGAR CSV destination (ignored by Git)",
    )
    return parser


if __name__ == "__main__":
    try:
        download_full_igar(build_parser().parse_args().output)
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
