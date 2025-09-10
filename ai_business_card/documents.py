import os
import logging
from uuid import uuid4

import aiofiles
import pymupdf4llm
from docx2md import Converter, DocxFile, DocxMedia
from langchain_core.documents import Document

from .constants import EXTENSIONS
from .settings import TEMP_DIR
from .depends import splitter, get_vectorstore

logger = logging.getLogger(__name__)


async def save_temp_file(filename: str, content: bytes) -> str:
    file_path = TEMP_DIR / filename
    async with aiofiles.open(file_path, "wb") as file:
        await file.write(content)
    logger.info("File %s successfully saved", file_path)
    return str(file_path)


async def store_document(file_path: str) -> list[str]:
    vectorstore = get_vectorstore()
    extension = str(file_path).split(".")[-1]
    if extension not in EXTENSIONS:
        raise ValueError(
            f"""Unsupported file format: {extension},
            supported extensions: {EXTENSIONS}
            """
        )
    match extension:
        case "docx":
            docx = DocxFile(file_path)
            media = DocxMedia(docx)
            converter = Converter(docx.document(), media, use_md_table=True)
            text = converter.convert()
            docx.close()
        case "pdf":
            text = pymupdf4llm.to_markdown(file_path)

            import gc

            gc.collect()
        case _:
            async with aiofiles.open(file_path, encoding="utf-8") as file:
                text = await file.read()
    os.remove(file_path)
    logger.info("File %s successfully handled", file_path)
    chunks = splitter.split_documents([
        Document(page_content=text, metadata={"file_path": file_path})
    ])
    logger.info(
        "File %s split to % chunks",
        file_path, len(chunks)
    )
    return await vectorstore.aadd_documents(
        chunks, ids=[str(uuid4()) for _ in range(len(chunks))]
    )
