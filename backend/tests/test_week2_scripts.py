import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def test_igar_sample_rejects_incomplete_label_quotas(tmp_path, monkeypatch):
    downloader = load_script_module(
        "week2_downloader_for_test",
        ROOT / "scripts/download_week2_datasets.py",
    )
    schema = ["", "app", "content", "translation", "score", "at", "appVersion", "labelScoreBase"]
    source_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(source_buffer, fieldnames=schema, lineterminator="\n")
    writer.writeheader()
    for index, label in enumerate(("Negative", "Neutral", "Positive")):
        writer.writerow(
            {
                "": str(index),
                "app": "demo",
                "content": "sample",
                "translation": "sample",
                "score": "3",
                "at": "2026-01-01 00:00:00",
                "appVersion": "",
                "labelScoreBase": label,
            }
        )
    source = source_buffer.getvalue().encode("utf-8")
    metadata_url = "https://example.test/metadata"
    files_url = "https://example.test/files"
    download_url = "https://example.test/download"
    manifest = {
        "doi": "10.17632/example.1",
        "metadata_url": metadata_url,
        "files_url": files_url,
        "file_name": "Rating_labeled.csv",
        "expected_size_bytes": len(source),
        "expected_sha256": hashlib.sha256(source).hexdigest(),
        "schema": schema,
    }

    def fake_request(url):
        if url == metadata_url:
            return FakeResponse(json.dumps({"doi": {"id": manifest["doi"]}}).encode())
        if url == files_url:
            return FakeResponse(
                json.dumps(
                    [
                        {
                            "filename": manifest["file_name"],
                            "content_details": {
                                "size": len(source),
                                "sha256_hash": manifest["expected_sha256"],
                                "download_url": download_url,
                            },
                        }
                    ]
                ).encode()
            )
        assert url == download_url
        return FakeResponse(source)

    monkeypatch.setattr(downloader, "request", fake_request)
    destination = tmp_path / "igar-sample.csv"

    with pytest.raises(ValueError, match="quotas"):
        downloader.fetch_igar_sample(manifest, destination, sample_size=6)

    assert not destination.exists()
