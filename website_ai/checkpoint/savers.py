from typing import Any

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
)
from redis.asyncio import Redis as AsyncRedis

from .base import TTL, RedisCheckpointError
from .utils import (
    _filter_keys,
    _load_writes,
    _make_redis_checkpoint_key,
    _make_redis_checkpoint_writes_key,
    _parse_redis_checkpoint_data,
    _parse_redis_checkpoint_key,
    _parse_redis_checkpoint_writes_key,
)


class AsyncRedisSaver(BaseCheckpointSaver):
    connection: AsyncRedis

    def __init__(self, connection: AsyncRedis) -> None:
        super().__init__()
        self.connection = connection

    @classmethod
    @asynccontextmanager
    async def from_connection_params(
            cls,
            *,
            host: str,
            port: int,
            db: int
    ) -> AsyncIterator["AsyncRedisSaver"]:
        connection: AsyncRedis | None = None
        try:
            connection = AsyncRedis(host=host, port=port, db=db)
            yield AsyncRedisSaver(connection)
        except Exception as ex:  # noqa: BLE001
            raise RedisCheckpointError(ex)  # noqa: B904
        finally:
            if connection:
                await connection.aclose()

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,  # noqa: ARG002
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        key = _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.serde.dumps(metadata)
        data = {
            "checkpoint": serialized_checkpoint,
            "type": type_,
            "checkpoint_id": checkpoint_id,
            "metadata": serialized_metadata,
            "parent_checkpoint_id": parent_checkpoint_id or "",
        }
        # await self.connection.hset(key, mapping=data)  # Solution for Redis >=4.0  # noqa: ERA001
        for field, value in data.items():  # Solution for Redis <4.0
            if value is not None:
                await self.connection.hset(key, field, value)
        await self.connection.expire(key, TTL)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",  # noqa: ARG002
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            key = _make_redis_checkpoint_writes_key(
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
            )
            type_, serialized_value = self.serde.dumps_typed(value)
            data = {"channel": channel, "type": type_, "value": serialized_value}
            if all(w[0] in WRITES_IDX_MAP for w in writes):
                await self.connection.hset(key, mapping=data)
                await self.connection.expire(key, TTL)
            else:
                for field, value in data.items():  # noqa: PLW2901
                    await self.connection.hsetnx(key, field, value)
                    await self.connection.expire(key, TTL)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        checkpoint_key = await self._aget_checkpoint_key(
            self.connection, thread_id, checkpoint_ns, checkpoint_id
        )
        if not checkpoint_key:
            return None
        checkpoint_data = await self.connection.hgetall(checkpoint_key)

        # load pending writes
        checkpoint_id = (
                checkpoint_id
                or _parse_redis_checkpoint_key(checkpoint_key)["checkpoint_id"]
        )
        pending_writes = await self._aload_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )
        return _parse_redis_checkpoint_data(
            self.serde, checkpoint_key, checkpoint_data, pending_writes=pending_writes
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002, ARG002
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncGenerator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        pattern = _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")
        keys = _filter_keys(await self.connection.keys(pattern), before, limit)
        for key in keys:
            data = await self.connection.hgetall(key)
            if data and b"checkpoint" in data and b"metadata" in data:
                checkpoint_id = _parse_redis_checkpoint_key(key.decode())[
                    "checkpoint_id"
                ]
                pending_writes = await self._aload_pending_writes(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                yield _parse_redis_checkpoint_data(
                    self.serde, key.decode(), data, pending_writes=pending_writes
                )

    async def _aload_pending_writes(
            self,
            thread_id: str,
            checkpoint_ns: str,
            checkpoint_id: str
    ) -> list[PendingWrite]:
        writes_key = _make_redis_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, "*", None
        )
        matching_keys = await self.connection.keys(pattern=writes_key)
        parsed_keys = [
            _parse_redis_checkpoint_writes_key(key.decode()) for key in matching_keys
        ]
        return _load_writes(
            self.serde,
            {
                (parsed_key["task_id"], parsed_key["idx"]): await self.connection.hgetall(key)
                for key, parsed_key in sorted(
                    zip(matching_keys, parsed_keys, strict=False), key=lambda x: x[1]["idx"]
                )
            },
        )

    @staticmethod
    async def _aget_checkpoint_key(
            connection: AsyncRedis,
            thread_id: str,
            checkpoint_ns: str,
            checkpoint_id: str | None
    ) -> str | None:
        if checkpoint_id:
            return _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        all_keys = await connection.keys(
            _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")
        )
        if not all_keys:
            return None

        latest_key = max(
            all_keys,
            key=lambda k: _parse_redis_checkpoint_key(k.decode())["checkpoint_id"],
        )
        return latest_key.decode()
