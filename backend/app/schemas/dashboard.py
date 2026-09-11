from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SentimentBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positive: float = Field(ge=0, le=100)
    neutral: float = Field(ge=0, le=100)
    negative: float = Field(ge=0, le=100)


class DashboardAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    name: str
    status: Literal["pending", "processing", "completed", "failed"]


class OverviewMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_feedback: int = Field(ge=0)
    total_units: int = Field(ge=0)
    positive: int = Field(ge=0)
    neutral: int = Field(ge=0)
    negative: int = Field(ge=0)
    positive_percentage: float = Field(ge=0, le=100)
    neutral_percentage: float = Field(ge=0, le=100)
    negative_percentage: float = Field(ge=0, le=100)


class TopicMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_id: int = Field(ge=0)
    label: str
    unit_count: int = Field(ge=0)
    sentiment: SentimentBreakdown


class IssueMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    frequency: int = Field(ge=0)
    topic_id: int | None = Field(default=None, ge=0)


class TrendMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    total_units: int = Field(ge=0)
    sentiment: SentimentBreakdown


class Insight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pending", "completed", "failed", "skipped"]
    summary: str | None = None


class DashboardResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis: DashboardAnalysis
    overview: OverviewMetrics
    topics: list[TopicMetric]
    top_issues: list[IssueMetric]
    trend: list[TrendMetric]
    insight: Insight
