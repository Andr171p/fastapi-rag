from enum import StrEnum

from pydantic import BaseModel

DEFAULT_K = 10
DEFAULT_MAX_LENGTH = 10
DEFAULT_TTL = 3600


class Role(StrEnum):
    USER = "user"
    AI = "ai"


class Message(BaseModel):
    chat_id: str
    role: Role
    text: str
