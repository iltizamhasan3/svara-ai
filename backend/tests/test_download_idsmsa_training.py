import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def load_downloader():
    spec = importlib.util.spec_from_file_location(
        "idsmsa_training_downloader", ROOT / "scripts/download_idsmsa_training.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def fixture_manifest(payload: bytes) -> dict[str, object]:
    schema = [
        "Tweet Date",
        "Sentence",
        "Quote Count",
        "Reply Count",
        "Retweet Count",
        "Favorite Count",
        "Sentiment",
        "English Translation",
    ]
    return {
        "sources": {
            "idsmsa": {
                "doi": "10.17632/test.3",
                "metadata_url": "https://metadata.invalid",
                "files_url": "https://files.invalid",
                "file_name": "IDSMSA.csv",
                "license": "CC BY 4.0",
                "schema": schema,
                "labels": ["Negative", "Neutral", "Positive"],
                "expected_rows": 2,
                "expected_size_bytes": len(payload),
                "expected_sha256": hashlib.sha256(payload).hexdigest(),
                "role": "additional_training",
            }
        }
    }


def test_download_idsmsa_verifies_provenance_schema_and_labels(tmp_path, monkeypatch):
    downloader = load_downloader()
    source = (
        "Tweet Date,Sentence,Quote Count,Reply Count,Retweet Count,Favorite Count,Sentiment,English Translation\n"
        "2025-01-01,bagus,0,0,0,0,Positive,good\n"
        "2025-01-02,buruk,0,0,0,0,Negative,bad\n"
    ).encode()
    manifest = fixture_manifest(source)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    details = manifest["sources"]["idsmsa"]
    assert isinstance(details, dict)

    def fake_request(url):
        if url == details["metadata_url"]:
            return FakeResponse(
                json.dumps(
                    {
                        "doi": {"id": details["doi"]},
                        "data_licence": {"short_name": details["license"]},
                    }
                ).encode()
            )
        if url == details["files_url"]:
            return FakeResponse(
                json.dumps(
                    [
                        {
                            "filename": details["file_name"],
                            "content_details": {
                                "size": details["expected_size_bytes"],
                                "sha256_hash": details["expected_sha256"],
                                "download_url": "https://download.invalid",
                            },
                        }
                    ]
                ).encode()
            )
        assert url == "https://download.invalid"
        return FakeResponse(source)

    monkeypatch.setattr(downloader, "request", fake_request)
    destination = tmp_path / "idsmsa.csv"

    assert downloader.download_idsmsa(destination, manifest_path=manifest_path) == destination.resolve()
    assert destination.read_bytes() == source
    assert downloader.validate_idsmsa(destination, manifest=manifest) == 2


def test_download_idsmsa_rejects_download_checksum_mismatch(tmp_path, monkeypatch):
    downloader = load_downloader()
    expected = (
        "Tweet Date,Sentence,Quote Count,Reply Count,Retweet Count,Favorite Count,Sentiment,English Translation\n"
        "2025-01-01,bagus,0,0,0,0,Positive,good\n"
        "2025-01-02,buruk,0,0,0,0,Negative,bad\n"
    ).encode()
    manifest = fixture_manifest(expected)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    details = manifest["sources"]["idsmsa"]
    assert isinstance(details, dict)

    def fake_request(url):
        if url == details["metadata_url"]:
            return FakeResponse(
                json.dumps(
                    {
                        "doi": details["doi"],
                        "data_licence": {"short_name": details["license"]},
                    }
                ).encode()
            )
        if url == details["files_url"]:
            return FakeResponse(
                json.dumps(
                    [
                        {
                            "filename": details["file_name"],
                            "content_details": {
                                "size": details["expected_size_bytes"],
                                "sha256_hash": details["expected_sha256"],
                                "download_url": "https://download.invalid",
                            },
                        }
                    ]
                ).encode()
            )
        assert url == "https://download.invalid"
        return FakeResponse(b"x" * len(expected))

    monkeypatch.setattr(downloader, "request", fake_request)
    with pytest.raises(ValueError, match="SHA-256"):
        downloader.download_idsmsa(tmp_path / "idsmsa.csv", manifest_path=manifest_path)
