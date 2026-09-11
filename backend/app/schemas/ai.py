from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


SentimentLabel = Literal["positive", "neutral", "negative"]


class ProbabilityMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positive: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    negative: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def probabilities_should_sum_to_one(self) -> "ProbabilityMap":
        if abs((self.positive + self.neutral + self.negative) - 1.0) > 0.01:
            raise ValueError("probabilities must sum to approximately 1")
        return self


class TopicKeyword(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(min_length=1)
    weight: float = Field(ge=0)
    rank: int = Field(ge=1)


class AIResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_id: UUID
    source_row_number: int = Field(ge=1)
    text: str = Field(min_length=1)
    sentiment: SentimentLabel
    confidence: float = Field(ge=0, le=1)
    probabilities: ProbabilityMap
    topic_id: int = Field(ge=-1)
    keywords: list[TopicKeyword] = Field(default_factory=list)
