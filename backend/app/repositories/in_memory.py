from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import UUID, uuid4

from app.schemas.ai import AIResult
from app.schemas.dashboard import DashboardResponse


@dataclass(slots=True)
class DatasetRecord:
    id: UUID
    user_id: UUID
    filename: str
    file_type: str
    storage_path: str
    columns: list[str]
    rows: list[dict[str, Any]]
    preview: list[dict[str, Any]]
    created_at: datetime


@dataclass(slots=True)
class AnalysisRecord:
    id: UUID
    user_id: UUID
    dataset_id: UUID
    name: str
    feedback_column: str
    date_column: str | None
    status: str = "pending"
    progress: int | None = 0
    stage: str | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    units: list[AIResult] = field(default_factory=list)
    dashboard: DashboardResponse | None = None


class InMemoryRepository:
    """Development repository used until Supabase repositories are implemented."""

    def __init__(self) -> None:
        self._datasets: dict[UUID, DatasetRecord] = {}
        self._analyses: dict[UUID, AnalysisRecord] = {}
        self._lock = RLock()

    def clear(self) -> None:
        with self._lock:
            self._datasets.clear()
            self._analyses.clear()

    def create_dataset(
        self,
        *,
        user_id: UUID,
        filename: str,
        file_type: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        preview: list[dict[str, Any]],
    ) -> DatasetRecord:
        dataset = DatasetRecord(
            id=uuid4(),
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            storage_path=f"dev/{user_id}/{uuid4()}/{filename}",
            columns=columns,
            rows=rows,
            preview=preview,
            created_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._datasets[dataset.id] = dataset
        return dataset

    def get_dataset_for_user(self, dataset_id: UUID, user_id: UUID) -> DatasetRecord | None:
        with self._lock:
            dataset = self._datasets.get(dataset_id)
            return dataset if dataset and dataset.user_id == user_id else None

    def create_analysis(
        self,
        *,
        user_id: UUID,
        dataset_id: UUID,
        name: str,
        feedback_column: str,
        date_column: str | None,
    ) -> AnalysisRecord:
        analysis = AnalysisRecord(
            id=uuid4(),
            user_id=user_id,
            dataset_id=dataset_id,
            name=name,
            feedback_column=feedback_column,
            date_column=date_column,
        )
        with self._lock:
            self._analyses[analysis.id] = analysis
        return analysis

    def get_analysis(self, analysis_id: UUID) -> AnalysisRecord | None:
        with self._lock:
            return self._analyses.get(analysis_id)

    def get_analysis_for_user(self, analysis_id: UUID, user_id: UUID) -> AnalysisRecord | None:
        with self._lock:
            analysis = self._analyses.get(analysis_id)
            return analysis if analysis and analysis.user_id == user_id else None

    def list_analyses(self, user_id: UUID, page: int, limit: int) -> tuple[list[AnalysisRecord], int]:
        with self._lock:
            all_items = sorted(
                (item for item in self._analyses.values() if item.user_id == user_id),
                key=lambda item: item.created_at,
                reverse=True,
            )
            start = (page - 1) * limit
            return all_items[start : start + limit], len(all_items)

    def mark_processing(self, analysis_id: UUID) -> AnalysisRecord | None:
        with self._lock:
            analysis = self._analyses.get(analysis_id)
            if analysis:
                analysis.status = "processing"
                analysis.progress = 5
                analysis.stage = "preprocessing"
                analysis.error_message = None
            return analysis

    def claim_analysis_for_user(
        self, analysis_id: UUID, user_id: UUID
    ) -> tuple[AnalysisRecord | None, bool]:
        """Atomically verify ownership and claim a pending/failed analysis."""
        with self._lock:
            analysis = self._analyses.get(analysis_id)
            if analysis is None or analysis.user_id != user_id:
                return None, False
            if analysis.status in {"processing", "completed"}:
                return analysis, False
            analysis.status = "processing"
            analysis.progress = 5
            analysis.stage = "preprocessing"
            analysis.error_message = None
            return analysis, True

    def complete_analysis(
        self,
        analysis_id: UUID,
        *,
        units: list[AIResult],
        dashboard: DashboardResponse,
    ) -> AnalysisRecord | None:
        with self._lock:
            analysis = self._analyses.get(analysis_id)
            if analysis:
                analysis.units = units
                analysis.dashboard = dashboard
                analysis.status = "completed"
                analysis.progress = 100
                analysis.stage = "completed"
            return analysis

    def fail_analysis(self, analysis_id: UUID, error_message: str) -> AnalysisRecord | None:
        with self._lock:
            analysis = self._analyses.get(analysis_id)
            if analysis:
                analysis.status = "failed"
                analysis.progress = None
                analysis.stage = "failed"
                analysis.error_message = error_message
            return analysis


repository = InMemoryRepository()
