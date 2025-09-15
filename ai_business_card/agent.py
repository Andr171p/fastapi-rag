import operator
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import END, START
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph, StateGraph

from .constants import TOP_K, TTL
from .depends import get_vectorstore, model
from .prompts import SYSTEM_PROMPT
from .schemas import Message
from .settings import settings


def format_messages(messages: Sequence[BaseMessage]) -> str:
    return "\n\n".join(
        f"{'User' if isinstance(message, HumanMessage) else 'AI'}: {message.content}"
        for message in messages
    )


def format_documents(documents: Sequence[Document]) -> str:
    return "\n\n".join([document.page_content for document in documents])


async def agent_node(state: MessagesState) -> MessagesState:
    vectorstore = get_vectorstore()
    chain = (
        {
            "context": vectorstore.as_retriever(k=TOP_K) | format_documents,
            "chat_history": RunnablePassthrough() | RunnableLambda(operator.itemgetter("chat_history")),
            "user_prompt": RunnablePassthrough() | RunnableLambda(operator.itemgetter("user_prompt")),
        }
        | ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        | model
    )
    user_prompt = state["messages"][-1].content
    chat_history = format_messages(state["messages"])
    ai_message = await chain.ainvoke({
        "user_prompt": user_prompt, "chat_history": chat_history
    })
    return {"messages": [ai_message]}


def compile_graph(
        checkpointer: BaseCheckpointSaver[MessagesState]
) -> CompiledStateGraph[MessagesState]:
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=checkpointer)


async def run_agent(thread_id: str, messages: list[Message]) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    input = {"messages": [message.model_dump() for message in messages]}
    async with AsyncRedisSaver(redis_url=settings.redis.url, ttl=TTL) as checkpointer:
        graph = compile_graph(checkpointer)
        response = await graph.ainvoke(input, config=config)
    return response["messages"][-1].content
