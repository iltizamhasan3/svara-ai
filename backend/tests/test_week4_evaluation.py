from app.ai.evaluation import manual_review_rows, topic_metrics
from app.ai.topic_model import TopicCluster, TopicKeyword, TopicModelConfig, TopicModelResult


def _result():
    return TopicModelResult(
        topic_ids=(0, 0, 1, -1),
        probabilities=(0.9, 0.8, 0.7, None),
        clusters=(
            TopicCluster(
                topic_id=0,
                unit_count=2,
                keywords=(TopicKeyword(keyword="login", weight=0.4, rank=1),),
                representative_texts=("login gagal",),
            ),
            TopicCluster(
                topic_id=1,
                unit_count=1,
                keywords=(TopicKeyword(keyword="makanan", weight=0.3, rank=1),),
                representative_texts=("makanan enak",),
            ),
        ),
        config=TopicModelConfig(),
        warnings=(),
    )


def test_topic_metrics_reports_unsupervised_diagnostics():
    metrics = topic_metrics(_result())

    assert metrics == {
        "unit_count": 4,
        "topic_count": 2,
        "outlier_count": 1,
        "outlier_rate": 0.25,
        "keyword_slots": 2,
        "unique_keyword_count": 2,
        "topic_diversity": 1.0,
        "warnings": [],
        "manual_review_required": True,
    }


def test_manual_review_rows_include_clusters_and_outlier_traceability():
    rows = manual_review_rows(_result())

    assert rows[0]["topic_id"] == 0
    assert rows[0]["manual_judgment"] == ""
    assert rows[-1]["topic_id"] == -1
    assert rows[-1]["manual_judgment"] == "outlier"
