from typing import Final

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

from ..settings import settings

engine: Final[AsyncEngine] = create_async_engine(url=settings.postgres.sqlalchemy_url, echo=True)

sessionmaker: Final[async_sessionmaker[AsyncSession]] = async_sessionmaker(
        engine, class_=AsyncSession, autoflush=False, expire_on_commit=False
)


class Base(AsyncAttrs, DeclarativeBase):
    __abstract__ = True

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid()
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
