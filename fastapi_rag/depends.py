from typing import Final

from elasticsearch import Elasticsearch
from embeddings_service.langchain import RemoteHTTPEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import ElasticSearchBM25Retriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_elasticsearch import ElasticsearchStore
from langchain_gigachat import GigaChat
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from redis.asyncio import Redis

from .settings import settings

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 20
TIMEOUT = 120

redis: Final[Redis] = Redis.from_url(settings.redis.url)

md_splitter: Final[MarkdownHeaderTextSplitter] = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Заголовок")]
)

text_splitter: Final[TextSplitter] = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n#"],
)

embeddings: Final[Embeddings] = RemoteHTTPEmbeddings(
    base_url=settings.embeddings.base_url, normalize_embeddings=False, timeout=TIMEOUT
)

vectorstore: Final[VectorStore] = ElasticsearchStore(
    es_url=settings.elasticsearch.url,
    index_name="rag-index",
    embedding=embeddings,
)

elasticsearch_client: Final[Elasticsearch] = Elasticsearch(settings.elasticsearch.url)

similarity_retriever: Final[BaseRetriever] = vectorstore.as_retriever(search_type="similarity")

bm25_retriever: Final[BaseRetriever] = ElasticSearchBM25Retriever(
    client=elasticsearch_client, index_name="rag-index",
)

ensemble_retriever: Final[BaseRetriever] = EnsembleRetriever(
    retrievers=[similarity_retriever, bm25_retriever], weights=[0.6, 0.4]
)

llm: Final[BaseChatModel] = GigaChat(
    credentials=settings.gigachat.api_key,
    scope=settings.gigachat.scope,
    model=settings.gigachat.model_name,
    profanity_check=False,
    verify_ssl_certs=False,
    timeout=TIMEOUT,
)
