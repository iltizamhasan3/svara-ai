"""Sentence Transformer embeddings with pinned, CPU-friendly configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


EMBEDDING_VERSION = "embedding-v1"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


class SentenceEncoder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Any: ...


@dataclass(frozen=True)
class EmbeddingConfig:
    """Reproducibility settings for the Week 4 embedding stage."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    revision: str = DEFAULT_EMBEDDING_REVISION
    batch_size: int = 32
    device: str = "cpu"
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name is required")
        if len(self.revision) != 40 or any(character not in "0123456789abcdef" for character in self.revision):
            raise ValueError("revision must be a 40-character lowercase commit SHA")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.device.strip():
            raise ValueError("device is required")


@dataclass(frozen=True)
class EmbeddingBatch:
    """Embedding matrix and metadata aligned to the input text order."""

    texts: tuple[str, ...]
    vectors: np.ndarray
    config: EmbeddingConfig
    version: str = EMBEDDING_VERSION

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError("embedding vectors must be a 2D matrix")
        if self.vectors.shape[0] != len(self.texts):
            raise ValueError("embedding row count must match input text count")
        if not np.isfinite(self.vectors).all():
            raise ValueError("embedding vectors must contain only finite values")


class SentenceTransformerEncoder:
    """Lazy-loading Sentence Transformer adapter for deterministic experiments."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        model: SentenceEncoder | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig()
        self._model = model

    def _load_model(self) -> SentenceEncoder:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.config.model_name,
                revision=self.config.revision,
                device=self.config.device,
            )
        return self._model

    def encode(self, texts: list[str] | tuple[str, ...]) -> EmbeddingBatch:
        """Encode texts while preserving their order and returning float32 vectors."""

        normalized_texts = tuple(str(text).strip() for text in texts)
        if any(not text for text in normalized_texts):
            raise ValueError("texts must contain non-empty values")
        if not normalized_texts:
            return EmbeddingBatch(
                texts=normalized_texts,
                vectors=np.empty((0, 0), dtype=np.float32),
                config=self.config,
            )

        encoded = self._load_model().encode(
            list(normalized_texts),
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        vectors = np.asarray(encoded, dtype=np.float32)
        return EmbeddingBatch(texts=normalized_texts, vectors=vectors, config=self.config)
