from collections.abc import Sequence

from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat import GigaChat
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, END
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.checkpoint.redis import AsyncRedisSaver
from langchain_core.messages import BaseMessage, HumanMessage

from .constants import TIMEOUT, TTL, TOP_K
from .prompts import SYSTEM_PROMPT
from .settings import settings

embeddings = HuggingFaceEmbeddings(
    model_name=settings.embeddings.model_name,
    model_kwargs=settings.embeddings.model_kwargs,
    encode_kwargs=settings.embeddings.encode_kwargs,
)

vectorstore = PineconeVectorStore(
    index="main", embedding=embeddings, pinecone_api_key=settings.pinecone.api_key
)

retriever = vectorstore.as_retriever(k=TOP_K)

model = GigaChat(
    credentials=settings.gigachat.api_key,
    scope=settings.gigachat.scope,
    model=settings.gigachat.model_name,
    profanity_check=False,
    verify_ssl_certs=False,
    timeout=TIMEOUT
)


def format_messages(messages: Sequence[BaseMessage]) -> str:
    return "\n\n".join(
        f"{'User' if isinstance(message, HumanMessage) else 'AI'}: {message.content}"
        for message in messages
    )


async def rag_node(state: MessagesState) -> MessagesState:
    chain = retriever | ChatPromptTemplate.from_template(SYSTEM_PROMPT) | model
    user_prompt = state["messages"][-1]
    chat_history = format_messages(state["messages"])
    ai_message = await chain.ainvoke({
        "user_prompt": user_prompt, "chat_history": chat_history
    })
    return {"messages": [ai_message]}


def compile_graph(
        checkpointer: BaseCheckpointSaver[MessagesState]
) -> CompiledStateGraph[MessagesState]:
    workflow = StateGraph(MessagesState)
    workflow.add_node("rag", rag_node)
    workflow.add_edge(START, "rag")
    workflow.add_edge("rag", END)
    return workflow.compile(checkpointer=checkpointer)


async def run_agent(thread_id: str, user_prompt: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [{"role": "human", "content": user_prompt}]}
    async with AsyncRedisSaver(redis_url=settings.redis.url, ttl=TTL) as checkpointer:
        graph = compile_graph(checkpointer)
        response = await graph.ainvoke(inputs, config=config)
    return response["messages"][-1]
