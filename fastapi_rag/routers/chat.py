from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Query, status
from pydantic import PositiveInt

from ..agent import execute_agent
from ..broker import broker
from ..database.queries import persist_task, read_chat_history, read_task
from ..schemas import ChatHistory, Message, Role, Task, TaskProcess, TaskStatus

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    path="/completions",
    status_code=status.HTTP_200_OK,
    response_model=Message,
    summary="Чат с RAG агентом",
)
async def create_chat_completion(
        user_message: Message, background_tasks: BackgroundTasks
) -> Message:
    response = await execute_agent(user_message.chat_id, user_message.text)
    ai_message = Message(chat_id=user_message.chat_id, role=Role.AI, text=response)
    background_tasks.add_task(
        broker.publish, [user_message, ai_message], queue="messages_persisting"
    )
    return ai_message


@router.post(
    path="/completions/async",
    status_code=status.HTTP_201_CREATED,
    response_model=Task,
    summary=""
)
async def create_chat_completion_async(
        user_message: Message, background_tasks: BackgroundTasks
) -> Task:
    task = Task(status=TaskStatus.PENDING)
    background_tasks.add_task(
        broker.publish,
        TaskProcess(id=task.id, user_message=user_message),
        queue="pending_tasks"
    )
    await persist_task(task)
    return task


@router.get(
    path="/tasks/{task_id}",
    status_code=status.HTTP_200_OK,
    response_model=Task,
    summary=""
)
async def get_chat_task(task_id: UUID) -> Task:
    return await read_task(task_id)


@router.get(
    path="/chat/history/{chat_id}",
    status_code=status.HTTP_200_OK,
    response_model=ChatHistory,
    summary="Получение истории сообщений чата"
)
async def get_chat_history(
        chat_id: UUID, page: PositiveInt = Query(...), limit: PositiveInt = Query(...)
) -> ChatHistory:
    return await read_chat_history(chat_id, page, limit)
