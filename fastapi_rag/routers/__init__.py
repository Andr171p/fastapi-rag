__all__ = ("router",)

from fastapi import APIRouter

from .chat import router as chat_router
from .documents import router as documents_router
from .ws import router as ws_router

router = APIRouter()

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(chat_router)
api_router.include_router(documents_router)

router.include_router(api_router)
router.include_router(ws_router)
