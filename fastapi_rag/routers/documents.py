from fastapi import APIRouter, File, UploadFile, status
from langchain_core.documents import Document

from ..indexing import indexing_chain, open_temp_file

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    path="/upload",
    status_code=status.HTTP_201_CREATED,
    response_model=list[Document],
    summary="Загружает документы в базу знаний"
)
async def upload_documents(file: UploadFile = File(...)) -> list[Document]:
    data = await file.read()
    async with open_temp_file(data) as temp_file:
        return await indexing_chain.ainvoke(temp_file)
