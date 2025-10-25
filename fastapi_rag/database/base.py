from typing import Final

import sqlite3
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, func
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ..settings import SQLITE_URL

engine: Final[AsyncEngine] = create_async_engine(url=SQLITE_URL, echo=True)

sessionmaker: Final[async_sessionmaker[AsyncSession]] = async_sessionmaker(
        engine, class_=AsyncSession, autoflush=False, expire_on_commit=False
)


def create_db() -> None:
    sqlite3.connect(SQLITE_URL)


class Base(AsyncAttrs, DeclarativeBase):
    __abstract__ = True

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid()
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


async def create_tables() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
