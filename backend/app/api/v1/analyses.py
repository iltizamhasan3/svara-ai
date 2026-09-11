from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status

from app.api.deps import current_user_id
from app.repositories.in_memory import repository
from app.schemas.analysis import (
    AnalysisHistoryItem,
    AnalysisHistoryResponse,
    AnalysisStatusResponse,
    CreateAnalysisRequest,
    CreateAnalysisResponse,
    StartAnalysisResponse,
)
from app.schemas.dashboard import DashboardResponse
from app.services.analysis import analysis_service


router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("", response_model=CreateAnalysisResponse, status_code=status.HTTP_201_CREATED)
def create_analysis(
    request: CreateAnalysisRequest,
    user_id=Depends(current_user_id),
) -> CreateAnalysisResponse:
    dataset = repository.get_dataset_for_user(request.dataset_id, user_id)
    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "dataset_not_found", "message": "Dataset not found"},
        )
    if request.feedback_column not in dataset.columns:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_feedback_column", "message": "feedback_column is not present in dataset"},
        )
    if request.date_column is not None and request.date_column not in dataset.columns:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_date_column", "message": "date_column is not present in dataset"},
        )

    analysis = repository.create_analysis(
        user_id=user_id,
        dataset_id=dataset.id,
        name=request.name,
        feedback_column=request.feedback_column,
        date_column=request.date_column,
    )
    return CreateAnalysisResponse(analysis_id=analysis.id, status="pending")


@router.post(
    "/{analysis_id}/start",
    response_model=StartAnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_analysis(
    analysis_id: UUID,
    background_tasks: BackgroundTasks,
    user_id=Depends(current_user_id),
) -> StartAnalysisResponse:
    analysis, claimed = repository.claim_analysis_for_user(analysis_id, user_id)
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "analysis_not_found", "message": "Analysis not found"},
        )
    if not claimed and analysis.status == "processing":
        raise HTTPException(
            status_code=409,
            detail={"code": "analysis_already_processing", "message": "Analysis is already processing"},
        )
    if analysis.status == "completed":
        return StartAnalysisResponse(analysis_id=analysis.id, status="completed")

    background_tasks.add_task(analysis_service.run, analysis.id)
    return StartAnalysisResponse(analysis_id=analysis.id, status="processing")


@router.get("/{analysis_id}/status", response_model=AnalysisStatusResponse)
def get_analysis_status(
    analysis_id: UUID,
    user_id=Depends(current_user_id),
) -> AnalysisStatusResponse:
    analysis = repository.get_analysis_for_user(analysis_id, user_id)
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "analysis_not_found", "message": "Analysis not found"},
        )
    return AnalysisStatusResponse(
        analysis_id=analysis.id,
        status=analysis.status,
        progress=analysis.progress,
        stage=analysis.stage,
        error_message=analysis.error_message,
    )


@router.get("/{analysis_id}/dashboard", response_model=DashboardResponse)
def get_dashboard(
    analysis_id: UUID,
    user_id=Depends(current_user_id),
) -> DashboardResponse:
    analysis = repository.get_analysis_for_user(analysis_id, user_id)
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "analysis_not_found", "message": "Analysis not found"},
        )
    if analysis.dashboard is None:
        raise HTTPException(
            status_code=409,
            detail={"code": "dashboard_not_ready", "message": "Analysis has not completed"},
        )
    return analysis.dashboard


@router.get("", response_model=AnalysisHistoryResponse)
def list_analyses(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    user_id=Depends(current_user_id),
) -> AnalysisHistoryResponse:
    items, total = repository.list_analyses(user_id, page, limit)
    return AnalysisHistoryResponse(
        items=[
            AnalysisHistoryItem(
                analysis_id=item.id,
                name=item.name,
                dataset_id=item.dataset_id,
                status=item.status,
                created_at=item.created_at,
            )
            for item in items
        ],
        page=page,
        limit=limit,
        total=total,
    )
