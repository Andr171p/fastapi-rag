from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .agent import run_agent
from .depends import get_vectorstore
from .documents import save_temp_file, store_document
from .schemas import DocumentsDelete, Message

router = APIRouter(prefix="/api/v1", tags=["API"])


@router.post(
    path="/chat/completion",
    status_code=status.HTTP_200_OK,
    response_model=Message,
    summary="Чат с AI ассистентом",
)
async def chat(message: Message) -> Message:
    response = await run_agent(message.thread_id, message.prompt)
    return Message(role="ai", thread_id=message.thread_id, prompt=response)


@router.post(
    path="/admin/documents/upload",
    status_code=status.HTTP_201_CREATED,
    response_model=list[str],
    summary="Загружает документ в базу знаний"
)
async def upload_document(
        file: Annotated[UploadFile, File(..., description="Документ для загрузки")],
) -> list[str]:
    file_path = await save_temp_file(file.filename, await file.read())
    return await store_document(file_path)


@router.get(
    path="/admin/documents/search",
    status_code=status.HTTP_200_OK,
    response_model=list[Document],
    summary="Выполняет поиск документов по базе знаний"
)
async def search_documents(
        query: Annotated[str, Query(..., description="Запрос для поиска")],
        top_k: Annotated[int, Query(..., description="Количество документов")],
        vectorstore: Annotated[VectorStore, Depends(get_vectorstore)]
) -> list[Document]:
    return await vectorstore.asimilarity_search(query, k=top_k)


@router.delete(
    path="/admin/documents",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удаляет документы по их ID"
)
async def delete_documents(
        documents: DocumentsDelete,
        vectorstore: Annotated[VectorStore, Depends(get_vectorstore)]
) -> None:
    await vectorstore.adelete(documents.ids)
