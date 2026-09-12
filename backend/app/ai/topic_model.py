"""BERTopic topic discovery with explicit UMAP, HDBSCAN, and c-TF-IDF config."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.schemas.ai import TopicKeyword


TOPIC_VERSION = "topic-v1"


@dataclass(frozen=True)
class TopicModelConfig:
    """Versioned CPU-friendly settings for the Week 4 topic experiment."""

    version: str = TOPIC_VERSION
    random_state: int = 42
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    hdbscan_min_cluster_size: int = 15
    hdbscan_min_samples: int = 5
    hdbscan_metric: str = "euclidean"
    min_topic_size: int = 15
    top_n_words: int = 10
    representative_texts: int = 3
    calculate_probabilities: bool = True

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("version is required")
        if self.umap_n_neighbors < 2:
            raise ValueError("umap_n_neighbors must be at least 2")
        if self.umap_n_components < 2:
            raise ValueError("umap_n_components must be at least 2")
        if not 0 <= self.umap_min_dist <= 1:
            raise ValueError("umap_min_dist must be between 0 and 1")
        if self.hdbscan_min_cluster_size < 2:
            raise ValueError("hdbscan_min_cluster_size must be at least 2")
        if self.hdbscan_min_samples < 1:
            raise ValueError("hdbscan_min_samples must be positive")
        if self.min_topic_size < 2:
            raise ValueError("min_topic_size must be at least 2")
        if self.min_topic_size != self.hdbscan_min_cluster_size:
            raise ValueError(
                "min_topic_size must match hdbscan_min_cluster_size when using a custom HDBSCAN model"
            )
        if self.top_n_words < 1:
            raise ValueError("top_n_words must be positive")
        if self.representative_texts < 0:
            raise ValueError("representative_texts cannot be negative")


@dataclass(frozen=True)
class TopicCluster:
    """Human-reviewable summary of one non-outlier BERTopic cluster."""

    topic_id: int
    unit_count: int
    keywords: tuple[TopicKeyword, ...]
    representative_texts: tuple[str, ...]

    @property
    def label(self) -> str:
        return " / ".join(keyword.keyword for keyword in self.keywords)


@dataclass(frozen=True)
class TopicModelResult:
    """Stable, serializable result of one topic-discovery run."""

    topic_ids: tuple[int, ...]
    probabilities: tuple[float | None, ...]
    clusters: tuple[TopicCluster, ...]
    config: TopicModelConfig
    warnings: tuple[str, ...] = ()

    @property
    def outlier_count(self) -> int:
        return sum(topic_id == -1 for topic_id in self.topic_ids)

    @property
    def topic_count(self) -> int:
        return len(self.clusters)


def _effective_umap_neighbors(config: TopicModelConfig, document_count: int) -> int:
    return min(config.umap_n_neighbors, max(2, document_count - 1))


def _effective_umap_components(config: TopicModelConfig, document_count: int) -> int:
    """Keep spectral initialization within UMAP's small-corpus boundary."""

    return min(config.umap_n_components, max(2, document_count - 2))


def _effective_hdbscan_min_samples(config: TopicModelConfig, document_count: int) -> int:
    return min(config.hdbscan_min_samples, document_count)


def _build_bertopic(config: TopicModelConfig, document_count: int) -> Any:
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=_effective_umap_neighbors(config, document_count),
        n_components=_effective_umap_components(config, document_count),
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=config.random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=_effective_hdbscan_min_samples(config, document_count),
        metric=config.hdbscan_metric,
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b\w+\b",
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    return BERTopic(
        top_n_words=config.top_n_words,
        calculate_probabilities=config.calculate_probabilities,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        verbose=False,
    )


def _probabilities_for_topics(
    topics: list[int],
    probabilities: Any,
) -> tuple[float | None, ...]:
    if probabilities is None:
        return tuple(None for _ in topics)

    matrix = np.asarray(probabilities)
    if matrix.ndim == 1:
        if matrix.shape[0] != len(topics):
            return tuple(None for _ in topics)
        values = matrix.tolist()
        return tuple(
            None if topic_id == -1 else float(values[index])
            for index, topic_id in enumerate(topics)
        )
    if matrix.ndim != 2 or matrix.shape[0] != len(topics):
        return tuple(None for _ in topics)
    topic_order = sorted({topic_id for topic_id in topics if topic_id >= 0})
    topic_to_column = {topic_id: column for column, topic_id in enumerate(topic_order)}
    return tuple(
        None
        if topic_id == -1 or topic_to_column.get(topic_id, matrix.shape[1]) >= matrix.shape[1]
        else float(matrix[index, topic_to_column[topic_id]])
        for index, topic_id in enumerate(topics)
    )


def _topic_cluster(model: Any, topic_id: int, unit_count: int, config: TopicModelConfig) -> TopicCluster:
    raw_keywords = model.get_topic(topic_id) or []
    usable_keywords = [
        (str(keyword).strip(), float(weight))
        for keyword, weight in raw_keywords
        if str(keyword).strip()
    ]
    keywords = tuple(
        TopicKeyword(keyword=keyword, weight=max(0.0, weight), rank=rank)
        for rank, (keyword, weight) in enumerate(
            sorted(usable_keywords, key=lambda item: (-item[1], item[0]))[: config.top_n_words],
            start=1,
        )
    )
    raw_representatives = []
    if config.representative_texts:
        raw_representatives = model.get_representative_docs(topic_id) or []
    return TopicCluster(
        topic_id=topic_id,
        unit_count=unit_count,
        keywords=keywords,
        representative_texts=tuple(str(text) for text in raw_representatives[: config.representative_texts]),
    )


def _empty_result(
    document_count: int,
    config: TopicModelConfig,
    warning: str,
) -> TopicModelResult:
    return TopicModelResult(
        topic_ids=tuple(-1 for _ in range(document_count)),
        probabilities=tuple(None for _ in range(document_count)),
        clusters=(),
        config=config,
        warnings=(warning,),
    )


def fit_topic_model(
    texts: list[str] | tuple[str, ...],
    embeddings: np.ndarray,
    *,
    config: TopicModelConfig | None = None,
) -> TopicModelResult:
    """Fit BERTopic and return assignments, c-TF-IDF keywords, and summaries."""

    selected_config = config or TopicModelConfig()
    normalized_texts = tuple(str(text).strip() for text in texts)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if len(normalized_texts) != matrix.shape[0]:
        raise ValueError("text and embedding counts must match")
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")
    if any(not text for text in normalized_texts):
        raise ValueError("texts must contain non-empty values")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain only finite values")
    if len(normalized_texts) < 4:
        return _empty_result(len(normalized_texts), selected_config, "insufficient_documents")

    model = _build_bertopic(selected_config, len(normalized_texts))
    topics, probabilities = model.fit_transform(list(normalized_texts), matrix)
    topic_ids = [int(topic_id) for topic_id in topics]
    counts = Counter(topic_id for topic_id in topic_ids if topic_id >= 0)
    clusters = tuple(
        _topic_cluster(model, topic_id, counts[topic_id], selected_config)
        for topic_id in sorted(counts)
    )
    warnings: list[str] = []
    if not clusters:
        warnings.append("all_documents_are_outliers")
    return TopicModelResult(
        topic_ids=tuple(topic_ids),
        probabilities=_probabilities_for_topics(topic_ids, probabilities),
        clusters=clusters,
        config=selected_config,
        warnings=tuple(warnings),
    )
