from uuid import UUID

from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from ..schemas import ChatHistory, Message, Task
from .base import sessionmaker
from .models import MessageModel, TaskModel


async def persist_messages(messages: list[Message]) -> None:
    """Сохраняет сообщения в базу данных"""
    try:
        async with sessionmaker() as session:
            stmt = insert(MessageModel)
            values = [message.model_dump() for message in messages]
            await session.execute(stmt, values)
            await session.commit()
    except SQLAlchemyError as e:
        await session.rollback()
        ...
    except IntegrityError as e:
        await session.rollback()
        ...


async def read_message(id: UUID) -> Message | None:  # noqa: A002
    """Получает сообщение по его уникальному идентификатору"""
    try:
        async with sessionmaker() as session:
            stmt = select(MessageModel).where(MessageModel.id == id)
            result = await session.execute(stmt)
            model = result.scalar_one_or_none()
        return Message.model_validate(model) if model else None
    except SQLAlchemyError as e:
        ...


async def read_chat_history(chat_id: UUID, page: int, limit: int) -> ChatHistory:
    try:
        async with sessionmaker() as session:
            stmt = (
                select(MessageModel, func.count(MessageModel.id).over().label("total_count"))
                .where(MessageModel.chat_id == chat_id)
                .order_by(MessageModel.created_at.asc())
                .offset((page - 1) * limit)
                .limit(limit)
            )
            results = await session.execute(stmt)
            rows = results.all()
            if not rows:
                return ChatHistory(
                    total_count=0, page=page, limit=limit, chat_id=chat_id, messages=[]
                )
            total_count = rows[0].total_count
            messages = [Message.model_validate(row[0]) for row in rows]
            return ChatHistory(
                total_count=total_count, page=page, limit=limit, chat_id=chat_id, messages=messages
            )
    except SQLAlchemyError as e:
        ...


async def persist_task(task: Task) -> None:
    try:
        async with sessionmaker() as session:
            stmt = insert(TaskModel).values(**task.model_dump())
            await session.execute(stmt)
            await session.commit()
    except SQLAlchemyError as e:
        ...


async def read_task(task_id: UUID) -> Task | None:
    try:
        async with sessionmaker() as session:
            stmt = (
                select(TaskModel, MessageModel)
                .join(MessageModel, TaskModel.message_id == MessageModel.id, isouter=True)
                .where(TaskModel.id == task_id)
            )
            result = await session.execute(stmt)
            row = result.first()
            if row is None:
                return None
            task_model, message_model = row
            return Task(
                id=task_model.id,
                status=task_model.status,
                message=Message.model_validate(message_model) if message_model else None,
            )
    except SQLAlchemyError as e:
        await session.rollback()
        ...


async def update_task(task_id: UUID, **kwargs) -> None:
    try:
        async with sessionmaker() as session:
            stmt = (
                update(TaskModel)
                .where(TaskModel.id == task_id)
                .values(**kwargs)
                .returning(TaskModel)
            )
            await session.execute(stmt)
            await session.commit()
    except SQLAlchemyError as e:
        await session.rollback()
