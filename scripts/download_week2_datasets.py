#!/usr/bin/env python3
"""Download pinned Week 2 sources and make a small deterministic IGAR sample."""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/week2_sources.json"
USER_AGENT = "svara-ai-week2-dataset-acquisition/1.0"
IGAR_LABELS = ("Negative", "Neutral", "Positive")


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60)


def atomic_download(url, destination, expected_size=None, expected_sha256=None, expected_git_sha=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part", dir=destination.parent)
    sha256 = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as output, request(url) as response:
            total = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk); sha256.update(chunk); total += len(chunk)
        if expected_size is not None and total != expected_size:
            raise ValueError(f"{destination.name}: expected {expected_size} bytes, got {total}")
        if expected_sha256 and sha256.hexdigest() != expected_sha256:
            raise ValueError(f"{destination.name}: SHA-256 mismatch")
        if expected_git_sha:
            git_hash_obj = hashlib.sha1(f"blob {total}\0".encode())
            with open(temporary, "rb") as downloaded:
                while chunk := downloaded.read(1024 * 1024): git_hash_obj.update(chunk)
            git_hash = git_hash_obj.hexdigest()
            if git_hash != expected_git_sha:
                raise ValueError(f"{destination.name}: Git blob hash mismatch")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def validate_smsa(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream, delimiter="\t")
        count = 0
        for row in rows:
            if len(row) != 2 or not row[0].strip() or row[1] not in {"positive", "neutral", "negative"}:
                raise ValueError(f"{path}: invalid SmSA row {count + 1}")
            count += 1
    if not count: raise ValueError(f"{path}: empty SmSA file")


def fetch_igar_sample(manifest, destination, sample_size):
    if sample_size < 3: raise ValueError("--igar-sample-size must be at least 3")
    metadata = json.loads(request(manifest["metadata_url"]).read().decode("utf-8"))
    metadata_doi = metadata.get("doi")
    if isinstance(metadata_doi, dict):
        metadata_doi = metadata_doi.get("id")
    if metadata_doi != manifest["doi"]: raise ValueError("IGAR DOI metadata mismatch")
    files = json.loads(request(manifest["files_url"]).read().decode("utf-8"))
    source = next((item for item in files if item.get("filename") == manifest["file_name"]), None)
    if not source: raise ValueError("Rating_labeled.csv not found in Mendeley file list")
    details = source["content_details"]
    if details.get("size") != manifest["expected_size_bytes"] or details.get("sha256_hash") != manifest["expected_sha256"]:
        raise ValueError("IGAR file provenance mismatch")
    quotas = {label: sample_size // 3 + (index < sample_size % 3) for index, label in enumerate(IGAR_LABELS)}
    selected = {label: [] for label in IGAR_LABELS}
    with request(details["download_url"]) as response:
        reader = csv.DictReader(io.TextIOWrapper(response, encoding="utf-8-sig", newline=""))
        if reader.fieldnames != manifest["schema"]: raise ValueError(f"IGAR schema mismatch: {reader.fieldnames}")
        for row in reader:
            label = row["labelScoreBase"]
            if label not in selected: raise ValueError(f"unexpected IGAR label: {label!r}")
            if len(selected[label]) < quotas[label]: selected[label].append(row)
            if all(len(selected[label]) == quotas[label] for label in IGAR_LABELS): break
    if any(not selected[label] for label in IGAR_LABELS): raise ValueError("IGAR stream did not contain all labels")
    rows = [row for label in IGAR_LABELS for row in selected[label]]
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=manifest["schema"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download pinned SmSA files and a deterministic stratified IGAR sample.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data", help="Dataset output root (default: repo data/)")
    parser.add_argument("--igar-sample-size", type=int, default=300, help="IGAR rows, split deterministically across labels (default: 300)")
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in manifest["sources"]["smsa"]["files"]:
        target = args.output_root / "raw/smsa" / item["name"]
        atomic_download(item["source_url"], target, item["expected_size_bytes"], expected_git_sha=item["expected_git_blob_sha1"])
        validate_smsa(target)
    fetch_igar_sample(manifest["sources"]["igar"], args.output_root / "samples/igar/Rating_labeled_sample.csv", args.igar_sample_size)
    print(f"Downloaded SmSA and wrote IGAR sample to {args.output_root}")


if __name__ == "__main__":
    try: main()
    except (OSError, ValueError, urllib.error.URLError) as error:
        print(f"error: {error}", file=sys.stderr); raise SystemExit(1)
