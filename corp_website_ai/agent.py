from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import END, START
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph, StateGraph

from .constants import TOP_K, TTL
from .depends import get_vectorstore, model
from .prompts import SYSTEM_PROMPT, USER_PROMPT
from .schemas import Message
from .settings import settings
from .utils import format_documents, format_messages


async def agent_node(state: MessagesState) -> MessagesState:
    vectorstore = get_vectorstore()
    chain = (
        {
            "context": vectorstore.as_retriever(k=TOP_K) | format_documents,
            "user_prompt": RunnablePassthrough(),
        }
        | ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        | model
    )
    query = state["messages"][-1].content
    chat_history = format_messages(state["messages"])
    ai_message = await chain.ainvoke(USER_PROMPT.format(query=query, chat_history=chat_history))
    return {"messages": [ai_message]}


def compile_graph(
        checkpointer: BaseCheckpointSaver[MessagesState]
) -> CompiledStateGraph[MessagesState]:
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=checkpointer)


async def run_agent(chat_id: str, messages: list[Message]) -> str:
    config = {"configurable": {"thread_id": chat_id}}
    state = {"messages": [message.model_dump() for message in messages]}
    async with AsyncRedisSaver(redis_url=settings.redis.url, ttl=TTL) as checkpointer:
        graph = compile_graph(checkpointer)
        response = await graph.ainvoke(state, config=config)
    return response["messages"][-1].content
