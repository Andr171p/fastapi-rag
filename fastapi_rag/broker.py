from typing import Final

from faststream import FastStream, Logger
from faststream.redis import RedisBroker

from .agent import execute_agent
from .database.queries import persist_messages, update_task
from .exceptions import AppError
from .schemas import Message, Role, TaskProcess, TaskStatus
from .settings import settings

broker = RedisBroker(url=settings.redis.url)

app: Final[FastStream] = FastStream(broker)


@broker.subscriber("pending_tasks")
@broker.publisher("messages_persisting")
async def handle_task(task: TaskProcess, logger: Logger) -> list[Message]:
    try:
        user_message = task.user_message
        response = await execute_agent(user_message.chat_id, user_message.text)
        ai_message = Message(chat_id=user_message.chat_id, role=Role.AI, text=response)
        await update_task(task.id, status=TaskStatus.DONE, message_id=ai_message.id)
    except AppError:
        logger.exception("{e}")
        await update_task(task.id, status=TaskStatus.ERROR)
    else:
        return [user_message, ai_message]


@broker.subscriber("messages_persisting")
async def handle_messages(messages: list[Message], logger: Logger) -> None:
    await persist_messages(messages)
    logger.info("Messages persisting successfully")
