from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage

from .constants import MAX_MESSAGES


def format_messages(messages: Sequence[BaseMessage], max_length: int = MAX_MESSAGES) -> str:
    return "\n\n".join(
        f"{'Пользователь' if isinstance(message, HumanMessage) else 'AI'}: {message.content}"
        for message in messages[:max_length]
    )


def format_documents(documents: Sequence[Document]) -> str:
    return "\n\n".join([document.page_content for document in documents])
