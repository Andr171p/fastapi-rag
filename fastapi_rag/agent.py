from typing import Final, TypedDict

import logging
from collections.abc import Sequence
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .depends import llm, redis, retriever
from .settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = settings.rag.system_prompt
USER_PROMPT = """История диалога:
{conversation_history}

Запрос пользователя:
{query}
"""
TTL = settings.redis.ttl
MAX_CONVERSATION_HISTORY_LENGTH = settings.rag.max_conversation_history_length


class State(TypedDict):
    """Состояние langgraph агента (FSM)

    Attributes:
        query: Запрос пользователя.
        conversation_history: История сообщений пользователя в рамках диалога.
        documents: Найденные документы по запросу пользователя.
        response: Финальный ответ агента.
    """
    query: str
    conversation_history: list[str]
    documents: list[Document]
    response: str


def format_documents(documents: Sequence[Document]) -> str:
    """Форматирует документы к LLM-friendly формату"""
    return "\n\n".join([document.page_content for document in documents])


def build_conversation_history_key(chat_id: UUID) -> str:
    return f"conversation_history:{chat_id}"


async def get_conversation_history(
        state: State, config: RunnableConfig | None = None  # noqa: ARG001
) -> dict[str, list[str]]:
    """Получение истории диалога пользователя"""
    logger.info("---GET CONVERSATION HISTORY---")
    key = build_conversation_history_key(config["configurable"]["chat_id"])
    messages = await redis.lrange(key, 0, -1)
    return {"conversation_history": [message.decode("utf-8") for message in reversed(messages)]}


async def retrieve(
        state: State, config: RunnableConfig | None = None  # noqa: ARG001
) -> dict[str, list[Document]]:
    """Извлечение документов из базы знаний"""
    logger.info("---RETRIEVE ---")
    documents = await retriever.ainvoke(state["query"])
    return {"documents": documents}


async def generate(
        state: State, config: RunnableConfig | None = None  # noqa: ARG001
) -> dict[str, str]:
    """Генерирует ответ на запрос пользователя"""
    logger.info("---GENERATE ---")
    user_prompt = USER_PROMPT.format(
        conversation_history="\n".join(state["conversation_history"]), query=state["query"]
    )
    chain = ChatPromptTemplate.from_template() | llm | StrOutputParser()
    response = await chain.ainvoke({
        "user_prompt": user_prompt, "context": format_documents(state["documents"]),
    })
    return {"response": response}


async def cache_conversation_history(
        state: State, config: RunnableConfig | None = None
) -> State:
    """Сохраняет истории диалога"""
    logger.info("---CACHE CONVERSATION HISTORY---")
    key = build_conversation_history_key(config["configurable"]["chat_id"])
    messages = [f"User: {state["query"]}", f"AI: {state["response"]}"]
    await redis.lpush(key, *messages)
    await redis.expire(key, TTL)
    await redis.ltrim(key, 0, MAX_CONVERSATION_HISTORY_LENGTH)
    return state


# Инициализация графа
workflow = StateGraph(State)
# Добавление вершин графа
workflow.add_node("get_conversation_history", get_conversation_history)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("cache_conversation_history", cache_conversation_history)
# Добавление ребёр графа
workflow.add_edge(START, "get_conversation_history")
workflow.add_edge("get_conversation_history", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "cache_conversation_history")
workflow.add_edge("cache_conversation_history", END)
# Компиляция графа
agent: Final[CompiledStateGraph[State]] = workflow.compile()


async def execute_agent(chat_id: UUID, query: str) -> str:
    """Функция для вызова RAG агента"""
    config = RunnableConfig(configurable={"chat_id": chat_id})
    accumulated_state = await agent.ainvoke({"query": query}, config=config)
    return accumulated_state.get("response", "")
