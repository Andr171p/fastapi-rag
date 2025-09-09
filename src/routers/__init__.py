__all__ = ["router"]

from fastapi import APIRouter

from .chat import chat_router

router = APIRouter(prefix="/api/v1")

router.include_router(chat_router)
