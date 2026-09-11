from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


AnalysisStatus = Literal["pending", "processing", "completed", "failed"]


class CreateAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: UUID
    name: str = Field(min_length=1, max_length=120)
    feedback_column: str = Field(min_length=1)
    date_column: str | None = None

    @field_validator("name", "feedback_column", "date_column")
    @classmethod
    def strip_text_fields(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else value


class CreateAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: UUID
    status: AnalysisStatus


class StartAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: UUID
    status: AnalysisStatus


class AnalysisStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: UUID
    status: AnalysisStatus
    progress: int | None = Field(default=None, ge=0, le=100)
    stage: str | None = None
    error_message: str | None = None


class AnalysisHistoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: UUID
    name: str
    dataset_id: UUID
    status: AnalysisStatus
    created_at: datetime


class AnalysisHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[AnalysisHistoryItem]
    page: int = Field(ge=1)
    limit: int = Field(ge=1, le=100)
    total: int = Field(ge=0)
