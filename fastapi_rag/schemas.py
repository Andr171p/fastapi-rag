from enum import StrEnum

from pydantic import BaseModel, field_validator

DEFAULT_K = 10
DEFAULT_MAX_LENGTH = 10
DEFAULT_TTL = 3600


class Role(StrEnum):
    AI = "ai"
    USER = "user"


class Message(BaseModel):
    role: Role
    text: str


class UserMessage(Message):
    chat_id: str
    role: Role = Role.USER


class Configurable(BaseModel):
    thread_id: str
    k: int = DEFAULT_K
    ttl: int = DEFAULT_TTL
    max_length: int = DEFAULT_MAX_LENGTH

    @field_validator("k", mode="before")
    def require_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("")
        return value
