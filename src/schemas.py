from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "ai"]
    thread_id: str
    prompt: str
