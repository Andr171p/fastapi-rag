from typing import Literal

from uuid import UUID

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "ai"]
    thread_id: str | UUID
    prompt: str


class DocumentsDelete(BaseModel):
    ids: list[str | UUID]


class DocumentAdd(BaseModel):
    text: str
