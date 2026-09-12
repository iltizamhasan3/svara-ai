import numpy as np
import pytest

from app.ai.topic_model import TOPIC_VERSION, TopicModelConfig, fit_topic_model


def _synthetic_inputs():
    texts = [
        "otp login tidak masuk",
        "verifikasi akun gagal",
        "login aplikasi mudah",
        "makanan enak dan segar",
        "menu makanan sangat lengkap",
        "rasa makanan memuaskan",
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    return texts, embeddings


def test_topic_config_has_explicit_umap_hdbscan_and_ctfidf_settings():
    config = TopicModelConfig()

    assert config.version == TOPIC_VERSION
    assert config.umap_n_neighbors == 15
    assert config.hdbscan_min_cluster_size == 15
    assert config.top_n_words == 10
    assert config.calculate_probabilities is True


def test_topic_model_returns_clusters_keywords_and_probabilities():
    texts, embeddings = _synthetic_inputs()
    config = TopicModelConfig(
        umap_n_neighbors=2,
        umap_n_components=2,
        hdbscan_min_cluster_size=2,
        hdbscan_min_samples=1,
        min_topic_size=2,
        top_n_words=3,
    )

    result = fit_topic_model(texts, embeddings, config=config)

    assert len(result.topic_ids) == len(texts)
    assert len(result.probabilities) == len(texts)
    assert result.topic_count == 2
    assert result.outlier_count == 0
    for cluster in result.clusters:
        assert cluster.unit_count == 3
        assert len(cluster.keywords) == 3
        assert [keyword.rank for keyword in cluster.keywords] == [1, 2, 3]
        assert all(keyword.weight >= 0 for keyword in cluster.keywords)
        assert cluster.representative_texts


def test_too_small_dataset_is_safe_and_marks_outliers():
    result = fit_topic_model(
        ["satu", "dua"],
        np.ones((2, 4), dtype=np.float32),
    )

    assert result.topic_ids == (-1, -1)
    assert result.probabilities == (None, None)
    assert result.clusters == ()
    assert result.warnings == ("insufficient_documents",)


@pytest.mark.parametrize(
    "texts,embeddings,error",
    [
        (["satu"], np.ones((2, 2), dtype=np.float32), "counts"),
        (["", "dua", "tiga"], np.ones((3, 2), dtype=np.float32), "non-empty"),
        (["satu", "dua", "tiga"], np.full((3, 2), np.nan), "finite"),
    ],
)
def test_topic_model_rejects_malformed_inputs(texts, embeddings, error):
    with pytest.raises(ValueError, match=error):
        fit_topic_model(texts, embeddings)
