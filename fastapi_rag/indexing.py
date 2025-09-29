import logging
import os

import aiofiles
import pymupdf4llm
from docx2md import Converter, DocxFile, DocxMedia
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

from .depends import md_splitter, text_splitter, vectorstore
from .settings import TEMP_DIR

AVAILABLE_EXTENSIONS: tuple[str, ...] = ("doc", "docx", "pdf", "txt", "md")

logger = logging.getLogger(__name__)


async def save_temp_file(filename: str, content: bytes) -> str:
    file_path = TEMP_DIR / filename
    async with aiofiles.open(file_path, "wb") as file:
        await file.write(content)
    logger.info("File %s successfully saved", file_path)
    return str(file_path)


async def process_file(filename: str, content: bytes) -> list[Document]:
    file_path = await save_temp_file(filename, content)
    extension = str(file_path).split(".")[-1]
    if extension not in AVAILABLE_EXTENSIONS:
        raise ValueError(
            f"""Unsupported file format: {extension},
            supported extensions: {AVAILABLE_EXTENSIONS}
            """
        )
    match extension:
        case "docx":
            docx = DocxFile(file_path)
            media = DocxMedia(docx)
            converter = Converter(docx.document(), media, use_md_table=True)
            md_text = converter.convert()
            docx.close()
        case "pdf":
            md_text = pymupdf4llm.to_markdown(file_path)
            import gc  # noqa: PLC0415
            gc.collect()
        case _:
            async with aiofiles.open(file_path, encoding="utf-8") as file:
                md_text = await file.read()
    if ".tmp" in file_path:
        os.remove(file_path)
    logger.info("File %s successfully handled", file_path)
    return md_splitter.split_text(md_text)


def split_documents(documents: list[Document]) -> list[Document]:
    return text_splitter.split_documents(documents)


async def save_documents(documents: list[Document]) -> list[Document]:
    await vectorstore.aadd_documents(documents)
    return documents


indexing_chain = (
    RunnablePassthrough()
    | split_documents
    | save_documents
)
