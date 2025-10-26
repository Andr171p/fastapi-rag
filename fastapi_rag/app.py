from typing import Final

from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .broker import app as faststream_app
from .database.base import create_db, create_tables
from .depends import create_index
from .exceptions import AppError
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


@app.exception_handler(AppError)
def handle_app_error(request: Request, exc: AppError) -> JSONResponse:  # noqa: ARG001
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": str(exc), "code": exc.code},
    )


@app.exception_handler(ValueError)
def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:  # noqa: ARG001
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"message": str(exc), "code": "VALIDATION_FAILED"},
    )
