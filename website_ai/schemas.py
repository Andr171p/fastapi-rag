from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel


class Role(StrEnum):
    AI = "ai"
    HUMAN = "human"


class Message(BaseModel):
    role: Role
    content: str


class HumanMessage(Message):
    role: Role = Role.HUMAN


class AIMessage(Message):
    role: Role = Role.AI


class DocumentsDelete(BaseModel):
    ids: list[str | UUID]


class DocumentAdd(BaseModel):
    text: str
