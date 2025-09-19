from typing import Optional, Union, TypedDict


class RedisCheckpointKey(TypedDict):
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str


class RedisCheckpointWritesKey(TypedDict):
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str
    task_id: str
    idx: Optional[Union[int, str]]
