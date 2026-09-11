from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DatasetUploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: UUID
    filename: str = Field(min_length=1)
    file_type: Literal["csv", "xlsx"]
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[str]
    preview: list[dict[str, Any]]
