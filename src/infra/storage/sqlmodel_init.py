from __future__ import annotations

from sqlmodel import SQLModel
from sqlalchemy import text

from infra.logging.logger import logger as log
from infra.storage.mysql_connector import mysql_connector

# Ensure ORM tables are registered into metadata
import models.sqlmodel  # noqa: F401


async def _ensure_http_request_trace_columns() -> None:
    """create_all 不会给已有表加列；旧库补 http_request_traces 扩展字段。"""
    engine = await mysql_connector.get_engine()
    stmts = [
        "ALTER TABLE http_request_traces ADD COLUMN process_id INT NULL",
        "ALTER TABLE http_request_traces ADD COLUMN upstream_task_id VARCHAR(128) NULL",
        "ALTER TABLE http_request_traces ADD COLUMN business_id VARCHAR(64) NULL",
        "ALTER TABLE http_request_traces ADD COLUMN business_success TINYINT(1) NULL",
    ]
    async with engine.begin() as conn:
        for sql in stmts:
            try:
                await conn.execute(text(sql))
                log.info("Applied HTTP trace column migration: %s", sql[:80])
            except Exception as e:
                msg = str(e).lower()
                if "duplicate" in msg or "1060" in msg:
                    log.debug("HTTP trace column exists, skip: %s", sql[:72])
                    continue
                log.warning("HTTP trace column migration failed: %s", e)


async def _ensure_history_request_id_columns() -> None:
    """
    create_all 不会给已有表加列；旧库需要补 request_id。
    """
    engine = await mysql_connector.get_engine()
    stmts = [
        "ALTER TABLE image_history_cards ADD COLUMN request_id VARCHAR(36) NULL",
        "ALTER TABLE video_analysis_history ADD COLUMN request_id VARCHAR(36) NULL",
    ]
    async with engine.begin() as conn:
        for sql in stmts:
            try:
                await conn.execute(text(sql))
                log.info("Applied column migration: %s", sql[:72])
            except Exception as e:
                msg = str(e).lower()
                if "duplicate" in msg or "1060" in msg:
                    log.debug("Column already exists, skip: %s", sql[:60])
                    continue
                log.warning("Column migration failed (check DB user permissions): %s", e)


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
    await _ensure_http_request_trace_columns()
    await _ensure_history_request_id_columns()

