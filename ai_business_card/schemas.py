from uuid import UUID

from pydantic import BaseModel


class DocumentsDelete(BaseModel):
    ids: list[str | UUID]


class DocumentAdd(BaseModel):
    text: str
