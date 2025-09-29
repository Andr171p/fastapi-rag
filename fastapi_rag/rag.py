from typing import TypedDict

import logging
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .depends import llm, redis, vectorstore
from .prompts import SYSTEM_PROMPT, USER_PROMPT

logger = logging.getLogger(__name__)


class State(TypedDict):
    query: str
    chat_history: list[str]
    documents: list[Document]
    response: str


def format_documents(documents: Sequence[Document]) -> str:
    return "\n\n".join([document.page_content for document in documents])


async def get_chat_history(
        state: State, config: RunnableConfig | None = None  # noqa: ARG001
) -> dict[str, list[str]]:
    logger.info("---RECEIVING CHAT HISTORY---")
    key = f"chat_history:{config["configurable"]["thread_id"]}"
    messages = await redis.lrange(key, 0, -1)
    return {"chat_history": [message.decode("utf-8") for message in reversed(messages)]}


async def retrieve(
        state: State, config: RunnableConfig | None = None
) -> dict[str, list[Document]]:
    logger.info("---RETRIEVE---")
    k = config["configurable"]["k"]
    documents = await vectorstore.as_retriever(k=k).ainvoke(state["query"])
    return {"documents": documents}


async def generate(
        state: State, config: RunnableConfig | None = None  # noqa: ARG001
) -> dict[str, str]:
    logger.info("---GENERATE---")
    user_prompt = USER_PROMPT.format(
        chat_history="\n".join(state["chat_history"]), query=state["query"]
    )
    llm_chain = ChatPromptTemplate.from_template(SYSTEM_PROMPT) | llm | StrOutputParser()
    response = await llm_chain.ainvoke({
        "user_prompt": user_prompt, "context": format_documents(state["documents"]),
    })
    return {"response": response}


async def persist_chat_history(state: State, config: RunnableConfig | None = None) -> State:
    logger.info("---PERSIST CHAT HISTORY---")
    key = f"chat_history:{config["configurable"]["thread_id"]}"
    max_length = config["configurable"]["max_length"]
    ttl = config["configurable"]["ttl"]
    user_message, ai_message = f"User: {state["query"]}", f"AI: {state["response"]}"
    await redis.lpush(key, user_message, ai_message)
    await redis.expire(key, ttl)
    await redis.ltrim(key, 0, max_length - 1)
    return state


builder = StateGraph(State)

builder.add_node("get_chat_history", get_chat_history)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("persist_chat_history", persist_chat_history)

builder.add_edge(START, "get_chat_history")
builder.add_edge("get_chat_history", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "persist_chat_history")
builder.add_edge("persist_chat_history", END)

agent = builder.compile()
