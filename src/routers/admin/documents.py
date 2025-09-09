from fastapi import APIRouter, UploadFile, status, File

documents_router = APIRouter(prefix="/documents", tags=["Documents"])


@documents_router.post(
    path="/upload",
    status_code=status.HTTP_201_CREATED,
    response_model=...,
    summary="",
)
async def upload_document(file: UploadFile = File(...)) -> ...:
    ...
