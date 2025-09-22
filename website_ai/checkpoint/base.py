from typing import TypedDict

REDIS_KEY_SEPARATOR = "$"
TTL = 3600


class RedisCheckpointError(Exception):
    pass


class RedisCheckpointKey(TypedDict):
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str


class RedisCheckpointWritesKey(TypedDict):
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str
    task_id: str
    idx: int | str | None
