from collections import Counter, defaultdict
from typing import Any
from uuid import UUID

from app.ai.mock_pipeline import MockAnalysisPipeline, keyword_counts
from app.repositories.in_memory import InMemoryRepository, repository
from app.schemas.ai import AIResult
from app.schemas.dashboard import (
    DashboardAnalysis,
    DashboardResponse,
    Insight,
    IssueMetric,
    OverviewMetrics,
    SentimentBreakdown,
    TopicMetric,
)


def _percentage(value: int, total: int) -> float:
    return round(value / total * 100, 2) if total else 0.0


class AnalysisService:
    def __init__(self, repository: InMemoryRepository, pipeline: MockAnalysisPipeline) -> None:
        self.repository = repository
        self.pipeline = pipeline

    def run(self, analysis_id: UUID) -> None:
        analysis = self.repository.get_analysis(analysis_id)
        if analysis is None:
            return
        dataset = self.repository.get_dataset_for_user(analysis.dataset_id, analysis.user_id)
        if dataset is None:
            self.repository.fail_analysis(analysis_id, "dataset_not_found")
            return

        try:
            units = self.pipeline.run(dataset.rows, analysis.feedback_column)
            dashboard = self._build_dashboard(analysis, len(dataset.rows), units)
            self.repository.complete_analysis(analysis_id, units=units, dashboard=dashboard)
        except Exception as exc:  # pragma: no cover - defensive boundary for background work
            self.repository.fail_analysis(analysis_id, str(exc))

    def _build_dashboard(self, analysis: Any, total_feedback: int, units: list[AIResult]) -> DashboardResponse:
        sentiment_counts = Counter(unit.sentiment for unit in units)
        total_units = len(units)
        overview = OverviewMetrics(
            total_feedback=total_feedback,
            total_units=total_units,
            positive=sentiment_counts["positive"],
            neutral=sentiment_counts["neutral"],
            negative=sentiment_counts["negative"],
            positive_percentage=_percentage(sentiment_counts["positive"], total_units),
            neutral_percentage=_percentage(sentiment_counts["neutral"], total_units),
            negative_percentage=_percentage(sentiment_counts["negative"], total_units),
        )

        topic_units: dict[int, list[AIResult]] = defaultdict(list)
        for unit in units:
            if unit.topic_id >= 0:
                topic_units[unit.topic_id].append(unit)

        topics: list[TopicMetric] = []
        for topic_id, topic_results in sorted(topic_units.items()):
            counts = Counter(unit.sentiment for unit in topic_results)
            keywords = Counter(
                keyword.keyword for unit in topic_results for keyword in unit.keywords
            )
            label = " / ".join(word.upper() for word, _ in keywords.most_common(3)) or f"Topic {topic_id}"
            topics.append(
                TopicMetric(
                    topic_id=topic_id,
                    label=label,
                    unit_count=len(topic_results),
                    sentiment=SentimentBreakdown(
                        positive=_percentage(counts["positive"], len(topic_results)),
                        neutral=_percentage(counts["neutral"], len(topic_results)),
                        negative=_percentage(counts["negative"], len(topic_results)),
                    ),
                )
            )

        issues = [
            IssueMetric(text=text, frequency=frequency, topic_id=topic_id)
            for text, (frequency, topic_id) in sorted(
                keyword_counts(units).items(), key=lambda item: (-item[1][0], item[0])
            )[:10]
        ]

        return DashboardResponse(
            analysis=DashboardAnalysis(id=analysis.id, name=analysis.name, status="completed"),
            overview=overview,
            topics=topics,
            top_issues=issues,
            trend=[],
            insight=Insight(status="skipped", summary=None),
        )


analysis_service = AnalysisService(repository, MockAnalysisPipeline())
