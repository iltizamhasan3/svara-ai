import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def load_downloader():
    spec = importlib.util.spec_from_file_location(
        "igar_full_downloader", ROOT / "scripts/download_igar_full.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_full_igar_verifies_size_and_checksum(tmp_path, monkeypatch):
    downloader = load_downloader()
    monkeypatch.setattr(downloader, "IGAR_TEST_ROOT", tmp_path / "data" / "raw" / "igar")
    source = tmp_path / "source.csv"
    source.write_bytes(b"content,labelScoreBase\ntext,Positive\n")
    expected_size = source.stat().st_size
    expected_sha = downloader.sha256_file(source)
    manifest = {
        "sources": {
            "igar": {
                "metadata_url": "https://metadata.invalid",
                "files_url": "https://files.invalid",
                "doi": "10.17632/test.1",
                "file_name": "Rating_labeled.csv",
                "expected_size_bytes": expected_size,
                "expected_sha256": expected_sha,
            }
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            payload, self.payload = self.payload, b""
            return payload

    def fake_request(url):
        if url == "https://metadata.invalid":
            return Response(json.dumps({"doi": "10.17632/test.1"}).encode())
        if url == "https://files.invalid":
            return Response(
                json.dumps(
                    [
                        {
                            "filename": "Rating_labeled.csv",
                            "content_details": {
                                "size": expected_size,
                                "sha256_hash": expected_sha,
                                "download_url": "https://download.invalid",
                            },
                        }
                    ]
                ).encode()
            )
        if url == "https://download.invalid":
            return Response(source.read_bytes())
        raise AssertionError(url)

    monkeypatch.setattr(downloader, "request", fake_request)
    destination = tmp_path / "data" / "raw" / "igar" / "downloaded.csv"

    assert downloader.download_full_igar(destination, manifest_path=manifest_path) == destination.resolve()
    assert destination.read_bytes() == source.read_bytes()


def test_download_full_igar_rejects_checksum_mismatch(tmp_path, monkeypatch):
    downloader = load_downloader()
    monkeypatch.setattr(downloader, "IGAR_TEST_ROOT", tmp_path / "data" / "raw" / "igar")
    manifest = {
        "sources": {
            "igar": {
                "metadata_url": "https://metadata.invalid",
                "files_url": "https://files.invalid",
                "doi": "10.17632/test.1",
                "file_name": "Rating_labeled.csv",
                "expected_size_bytes": 1,
                "expected_sha256": "0" * 64,
            }
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            if size == -1:
                return json.dumps({"doi": "10.17632/test.1"}).encode()
            if getattr(self, "sent", False):
                return b""
            self.sent = True
            return b"x"

    def fake_request(url):
        if url == "https://metadata.invalid":
            return Response()
        if url == "https://files.invalid":
            response = Response()
            response.read = lambda size=-1: json.dumps(
                [
                    {
                        "filename": "Rating_labeled.csv",
                        "content_details": {
                            "size": 1,
                            "sha256_hash": "0" * 64,
                            "download_url": "https://download.invalid",
                        },
                    }
                ]
            ).encode()
            return response
        if url == "https://download.invalid":
            return Response()
        raise AssertionError(url)

    monkeypatch.setattr(downloader, "request", fake_request)
    with pytest.raises(ValueError, match="SHA-256"):
        downloader.download_full_igar(
            tmp_path / "data" / "raw" / "igar" / "bad.csv",
            manifest_path=manifest_path,
        )


def test_download_full_igar_rejects_non_test_destination(tmp_path):
    downloader = load_downloader()

    with pytest.raises(ValueError, match="test-only"):
        downloader.download_full_igar(tmp_path / "training.csv")
