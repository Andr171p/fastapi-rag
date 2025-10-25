from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class Role(StrEnum):
    USER = "user"
    AI = "ai"


class Message(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    chat_id: UUID
    role: Role
    text: str

    model_config = ConfigDict(from_attributes=True)


class ChatHistory(BaseModel):
    total_count: int
    page: PositiveInt
    limit: PositiveInt
    chat_id: UUID
    messages: list[Message]


class TaskStatus(StrEnum):
    PENDING = "pending"
    DONE = "done"
    ERROR = "error"


class TaskProcess(BaseModel):
    id: UUID
    user_message: Message


class Task(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    status: TaskStatus
    message: Message | None = None
