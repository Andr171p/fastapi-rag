from typing import Final

from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .broker import app as faststream_app
from .database.base import create_db, create_tables
from .depends import create_index
from .routers import router


async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    create_index("rag-index")
    create_db()
    await create_tables()
    await faststream_app.broker.start()
    yield
    await faststream_app.broker.stop()


app: Final[FastAPI] = FastAPI(lifespan=lifespan)

app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
