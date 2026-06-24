# routers/chat.py — Aggregated chat router
from fastapi import APIRouter
from routers.chat_core import router as chat_core_router
from routers.chat_role import router as chat_role_router
from routers.chat_tts import router as chat_tts_router

chat_router = APIRouter()

chat_router.include_router(chat_core_router, prefix="/chat", tags=["chat"])
chat_router.include_router(chat_role_router, prefix="/chat", tags=["chat"])
chat_router.include_router(chat_tts_router, prefix="/chat", tags=["chat"])
