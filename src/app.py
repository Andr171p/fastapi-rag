from typing import Final

from fastapi import FastAPI

from .routers import router

app: Final[FastAPI] = FastAPI()

app.include_router(router)
