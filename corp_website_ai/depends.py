from typing import Final

from embeddings_service.langchain import RemoteHTTPEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_gigachat import GigaChat
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from .constants import CHUNK_OVERLAP, CHUNK_SIZE, TIMEOUT
from .settings import settings

splitter: Final[TextSplitter] = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len
)

embeddings: Final[Embeddings] = RemoteHTTPEmbeddings(
    base_url=settings.embeddings.base_url, normalize_embeddings=False, timeout=60
)


def get_vectorstore() -> VectorStore:
    return PineconeVectorStore(
        embedding=embeddings, pinecone_api_key=settings.pinecone.api_key, index_name="website"
    )


model: Final[BaseChatModel] = GigaChat(
    credentials=settings.gigachat.api_key,
    scope=settings.gigachat.scope,
    model=settings.gigachat.model_name,
    profanity_check=False,
    verify_ssl_certs=False,
    timeout=TIMEOUT
)
