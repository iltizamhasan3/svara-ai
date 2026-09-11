from fastapi import APIRouter

from app.api.v1 import analyses, datasets, health


api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(datasets.router)
api_router.include_router(analyses.router)
