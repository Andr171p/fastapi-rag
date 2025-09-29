from typing import Final

from fastapi import FastAPI, File, Request, UploadFile, status
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from .indexing import indexing_chain, process_file
from .rag import agent
from .schemas import Message, Role, UserMessage

DEFAULT_K = 10
DEFAULT_MAX_LENGTH = 10
DEFAULT_TTL = 3600

app: Final[FastAPI] = FastAPI()


@app.post(
    path="api/v1/documents",
    status_code=status.HTTP_201_CREATED,
    response_model=list[Document],
    summary="",
)
async def add_document(document: Document) -> list[Document]:
    return await indexing_chain.ainvoke([document])


@app.post(
    path="api/v1/documents/upload",
    status_code=status.HTTP_201_CREATED,
    response_model=list[Document],
    summary="",
)
async def upload_document(file: UploadFile = File(...)) -> list[Document]:
    document = await process_file(file.filename, await file.read())
    return await indexing_chain.ainvoke([document])


@app.put(
    path="/api/v1/prompts",
    status_code=status.HTTP_200_OK,
    response_model=...,
    summary="",
)
async def write_prompt() -> ...: ...


@app.post(
    path="api/v1/rag",
    status_code=status.HTTP_200_OK,
    response_model=Message,
    summary="",
)
async def generate_response(
        user_message: UserMessage
) -> Message:
    config = RunnableConfig(
        configurable={
            "thread_id": user_message.chat_id,
            "k": DEFAULT_K,
            "ttl": DEFAULT_TTL,
            "max_length": DEFAULT_MAX_LENGTH,
        }
    )
    state = await agent.ainvoke({"query": user_message.text}, config=config)
    return Message(role=Role.AI, text=state["response"])


@app.exception_handler(ValueError)
def handle_value_error(exc: ValueError, request: Request) -> JSONResponse:  # noqa: ARG001
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)},
    )
