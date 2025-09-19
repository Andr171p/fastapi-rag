from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Optional,
    Sequence,
    Tuple,
    Dict,
    List
)

from redis.asyncio import Redis as AsyncRedis

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointTuple,
    CheckpointMetadata,
    ChannelVersions,
    WRITES_IDX_MAP,
    PendingWrite,
    get_checkpoint_id
)

from .exceptions import RedisCheckpointException
from .constants import TTL
from .utils import (
    _make_redis_checkpoint_key,
    _make_redis_checkpoint_writes_key,
    _parse_redis_checkpoint_key,
    _parse_redis_checkpoint_data,
    _parse_redis_checkpoint_writes_key,
    _filter_keys,
    _load_writes
)


class AsyncRedisCheckpointSaver(BaseCheckpointSaver):
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
    ) -> AsyncIterator["AsyncRedisCheckpointSaver"]:
        connection: Optional[AsyncRedis] = None
        try:
            connection = AsyncRedis(host=host, port=port, db=db)
            yield AsyncRedisCheckpointSaver(connection)
        except Exception as ex:
            raise RedisCheckpointException(ex)
        finally:
            if connection:
                await connection.aclose()

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
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
            "parent_checkpoint_id": parent_checkpoint_id
            if parent_checkpoint_id
            else "",
        }
        # await self.connection.hset(key, mapping=data)  # Solution for Redis >=4.0
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
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
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
                for field, value in data.items():
                    await self.connection.hsetnx(key, field, value)
                    await self.connection.expire(key, TTL)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
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
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncGenerator[CheckpointTuple, None]:
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
    ) -> List[PendingWrite]:
        writes_key = _make_redis_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, "*", None
        )
        matching_keys = await self.connection.keys(pattern=writes_key)
        parsed_keys = [
            _parse_redis_checkpoint_writes_key(key.decode()) for key in matching_keys
        ]
        pending_writes = _load_writes(
            self.serde,
            {
                (parsed_key["task_id"], parsed_key["idx"]): await self.connection.hgetall(key)
                for key, parsed_key in sorted(zip(matching_keys, parsed_keys), key=lambda x: x[1]["idx"])
            },
        )
        return pending_writes

    @staticmethod
    async def _aget_checkpoint_key(
            connection: AsyncRedis,
            thread_id: str,
            checkpoint_ns: str,
            checkpoint_id: Optional[str]
    ) -> Optional[str]:
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