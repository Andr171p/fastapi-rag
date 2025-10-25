from typing import Final, TypeVar

import logging
from abc import ABC, abstractmethod
from uuid import UUID

from fastapi import WebSocket
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ConnectionId = str | UUID
PayloadType = TypeVar("PayloadType", bound=BaseModel)


class ConnectionManager(ABC):
    @abstractmethod
    async def connect(self, websocket: WebSocket, connection_id: ConnectionId) -> None:
        """Создаёт websocket соединение"""

    @abstractmethod
    async def disconnect(self, connection_id: ConnectionId) -> None:
        """Разрывает websocket соединение"""

    @abstractmethod
    async def get_connection(self, connection_id: ConnectionId) -> WebSocket | None:
        """Получает websocket соединение по его идентификатору"""

    async def send(self, connection_id: ConnectionId, payload: PayloadType) -> None:
        """Выполнение отправки данных через websocket соединение.

        :param connection_id: Уникальный идентификатор соединения.
        :param payload: Данные, который нужно передать.
        """
        connection = await self.get_connection(connection_id)
        if connection is None:
            return
        await connection.send_json(payload)


class InMemoryConnectionManager(ConnectionManager):
    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, connection_id: ConnectionId) -> None:
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info("Created new connection with id %s", connection_id)

    async def disconnect(self, connection_id: ConnectionId) -> None:
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            logger.info("Deleted connection by id %s", connection_id)

    async def get_connection(self, connection_id: ConnectionId) -> WebSocket:
        return self.active_connections.get(connection_id)


connection_manager: Final[ConnectionManager] = InMemoryConnectionManager()
