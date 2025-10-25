from uuid import UUID

from sqlalchemy import CheckConstraint, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class MessageModel(Base):
    __tablename__ = "messages"

    chat_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), unique=False)
    role: Mapped[str]
    text: Mapped[str] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("role IN ('user', 'ai')", name="check_role_values"),
    )


class TaskModel(Base):
    __tablename__ = "tasks"

    status: Mapped[str]
    message_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), unique=True, nullable=True
    )
