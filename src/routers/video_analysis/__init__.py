from fastapi import APIRouter
from .crud import crud_router
from .analysis import analysis_router
from .search import search_router

video_analysis_router = APIRouter(prefix="/video-analysis", tags=["video-analysis"])
video_analysis_router.include_router(crud_router)
video_analysis_router.include_router(analysis_router)
video_analysis_router.include_router(search_router)
