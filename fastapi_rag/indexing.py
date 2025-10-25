import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import aiofiles.tempfile
import pymupdf4llm
from aiofiles.threadpool.binary import AsyncFileIO
from docx2md import Converter, DocxFile, DocxMedia
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

from .depends import md_splitter, text_splitter, vectorstore

AVAILABLE_EXTENSIONS: tuple[str, ...] = ("doc", "docx", "pdf", "txt", "md")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def open_temp_file(data: bytes) -> AsyncGenerator[AsyncFileIO]:
    """Контекстный менеджер для открытия и работы с временным файлом.

    :param data: Данные (поток байтов), которые нужно записать в файл.
    """
    async with aiofiles.tempfile.NamedTemporaryFile(mode="wb", encoding="utf-8") as file:
        await file.write(data)
        await file.flush()
        yield file


async def process_file(file: AsyncFileIO) -> list[Document]:
    """Обрабатывает и преобразует открытый файл в формат Markdown,
    после чего выполняется разделение по h1 заголовкам.

    :param file: Открытый временный файл.
    :return Список документов после разбиения по h1.
    """
    filename = file.name
    extension = str(filename).split(".")[-1]
    if extension not in AVAILABLE_EXTENSIONS:
        raise ValueError(
            f"""Unsupported file format: {extension},
            supported extensions: {AVAILABLE_EXTENSIONS}"""
        )
    match extension:
        case "docx" | "doc":
            docx = DocxFile(filename)
            media = DocxMedia(docx)
            converter = Converter(docx.document(), media, use_md_table=True)
            md_text = converter.convert()
            docx.close()
        case "pdf":
            md_text = pymupdf4llm.to_markdown(filename)
            import gc  # noqa: PLC0415
            gc.collect()
        case _:
            md_text = await file.read()
    logger.info("File %s successfully processed", filename)
    return md_splitter.split_text(md_text)


indexing_chain = (
    RunnablePassthrough()
    | process_file
    | text_splitter.split_documents
    | vectorstore.aadd_documents
)
