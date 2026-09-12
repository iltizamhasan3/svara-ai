import numpy as np
import pytest

from app.ai.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    EMBEDDING_VERSION,
    EmbeddingConfig,
    SentenceTransformerEncoder,
)


class FakeSentenceEncoder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, sentences, **kwargs):
        self.calls.append({"sentences": sentences, **kwargs})
        return np.asarray([[float(index), 1.0] for index, _ in enumerate(sentences)])


def test_default_embedding_config_is_pinned_and_cpu_practical():
    config = EmbeddingConfig()

    assert config.model_name == DEFAULT_EMBEDDING_MODEL
    assert config.revision == DEFAULT_EMBEDDING_REVISION
    assert config.device == "cpu"
    assert config.batch_size == 32


def test_encoder_preserves_order_and_records_metadata():
    fake_model = FakeSentenceEncoder()
    encoder = SentenceTransformerEncoder(
        EmbeddingConfig(batch_size=2),
        model=fake_model,
    )

    batch = encoder.encode([" OTP gagal ", "Makanan enak"])

    assert batch.version == EMBEDDING_VERSION
    assert batch.texts == ("OTP gagal", "Makanan enak")
    assert batch.vectors.dtype == np.float32
    assert batch.vectors.shape == (2, 2)
    assert fake_model.calls == [
        {
            "sentences": ["OTP gagal", "Makanan enak"],
            "batch_size": 2,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
    ]


def test_empty_batch_has_stable_shape():
    batch = SentenceTransformerEncoder(model=FakeSentenceEncoder()).encode([])

    assert batch.texts == ()
    assert batch.vectors.shape == (0, 0)


def test_encoder_rejects_blank_text():
    with pytest.raises(ValueError, match="non-empty"):
        SentenceTransformerEncoder(model=FakeSentenceEncoder()).encode(["valid", " "])


@pytest.mark.parametrize(
    "field,value",
    [("batch_size", 0), ("device", ""), ("model_name", ""), ("revision", "not-a-sha")],
)
def test_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        EmbeddingConfig(**{field: value})
