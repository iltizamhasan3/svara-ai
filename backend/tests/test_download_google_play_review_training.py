import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def load_downloader():
    spec = importlib.util.spec_from_file_location(
        "google_play_review_training_downloader",
        ROOT / "scripts/download_google_play_review_training.py",
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


def fixture_manifest(payloads: dict[str, bytes]) -> dict[str, object]:
    files = []
    for name, payload in payloads.items():
        files.append(
            {
                "name": name,
                "url": f"https://download.invalid/{name}",
                "role": "additional_training_source" if name == "train.csv" else "source_holdout_reference",
                "expected_rows": 2,
                "expected_size_bytes": len(payload),
                "expected_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "training_policy": {"igar_forbidden": True},
        "sources": {
            "google_play_review": {
                "dataset": "Indonesian Google Play Review",
                "license": "CC BY 4.0",
                "schema": ["text", "label", "stars"],
                "raw_labels": ["neg", "pos"],
                "role": "additional_training",
                "files": files,
            }
        },
    }


def test_download_google_play_review_validates_raw_contract(tmp_path, monkeypatch):
    downloader = load_downloader()
    payloads = {
        "train.csv": b"text,label,stars\nbagus,pos,5\nburuk,neg,1\n",
        "validation.csv": b"text,label,stars\nnetral,neg,3\noke,pos,4\n",
    }
    manifest = fixture_manifest(payloads)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_request(url):
        name = url.rsplit("/", 1)[-1]
        return FakeResponse(payloads[name])

    monkeypatch.setattr(downloader, "request", fake_request)
    paths = downloader.download_google_play_review(tmp_path / "raw", manifest_path=manifest_path)

    assert [path.name for path in paths] == ["train.csv", "validation.csv"]
    assert [path.read_bytes() for path in paths] == list(payloads.values())


def test_download_google_play_review_rejects_bad_star_value(tmp_path):
    downloader = load_downloader()
    source = {
        "raw_labels": ["neg", "pos"],
    }
    details = {"expected_rows": 1}
    path = tmp_path / "bad.csv"
    path.write_text("text,label,stars\ncontoh,pos,6\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside 1..5"):
        downloader.validate_google_play_review(path, file_details=details, source=source)
