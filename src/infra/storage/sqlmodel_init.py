from __future__ import annotations

from sqlmodel import SQLModel

from infra.logging.logger import logger as log
from infra.storage.mysql_connector import mysql_connector

# Ensure ORM tables are registered into metadata
import models.sqlmodel  # noqa: F401


async def create_tables_if_not_exists() -> None:
    """
    Create SQLModel tables if they do not exist.
    Uses the existing async MySQL engine.
    """
    engine = await mysql_connector.get_engine()
    async with engine.begin() as conn:
        log.info("Ensuring SQLModel tables exist...")
        await conn.run_sync(SQLModel.metadata.create_all)
        log.info("SQLModel table check complete.")

