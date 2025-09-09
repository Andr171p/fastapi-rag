from fastapi import APIRouter, status

from ..agent import run_agent
from ..schemas import Message

chat_router = APIRouter(prefix="/chat", tags=["Chat"])


@chat_router.post(
    path="/completion",
    status_code=status.HTTP_200_OK,
    response_model=Message,
    summary="Чат с AI ассистентом"
)
async def chat(message: Message) -> Message:
    response = await run_agent(message.thread_id, message.prompt)
    return Message(role="ai", thread_id=message.thread_id, prompt=response)
