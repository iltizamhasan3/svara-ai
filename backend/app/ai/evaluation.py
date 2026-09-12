"""Qualitative and descriptive evaluation helpers for Week 4 topic runs."""

from __future__ import annotations

import json
from typing import Any

from app.ai.topic_model import TopicModelResult


def topic_metrics(result: TopicModelResult) -> dict[str, Any]:
    """Calculate unsupervised diagnostics without pretending to have labels."""

    unit_count = len(result.topic_ids)
    keyword_values = [
        keyword.keyword
        for cluster in result.clusters
        for keyword in cluster.keywords
    ]
    keyword_slots = sum(len(cluster.keywords) for cluster in result.clusters)
    return {
        "unit_count": unit_count,
        "topic_count": result.topic_count,
        "outlier_count": result.outlier_count,
        "outlier_rate": round(result.outlier_count / unit_count, 4) if unit_count else 0.0,
        "keyword_slots": keyword_slots,
        "unique_keyword_count": len(set(keyword_values)),
        "topic_diversity": round(len(set(keyword_values)) / keyword_slots, 4)
        if keyword_slots
        else 0.0,
        "warnings": list(result.warnings),
        "manual_review_required": bool(result.clusters),
    }


def manual_review_rows(result: TopicModelResult) -> list[dict[str, object]]:
    """Build a CSV-friendly worksheet for human cluster review."""

    rows: list[dict[str, object]] = []
    for cluster in result.clusters:
        rows.append(
            {
                "topic_id": cluster.topic_id,
                "unit_count": cluster.unit_count,
                "keywords": " | ".join(keyword.keyword for keyword in cluster.keywords),
                "representative_texts": json.dumps(
                    list(cluster.representative_texts), ensure_ascii=False
                ),
                "manual_judgment": "",
                "review_notes": "",
            }
        )
    if result.outlier_count:
        rows.append(
            {
                "topic_id": -1,
                "unit_count": result.outlier_count,
                "keywords": "",
                "representative_texts": "[]",
                "manual_judgment": "outlier",
                "review_notes": "HDBSCAN outlier units are retained for traceability.",
            }
        )
    return rows
