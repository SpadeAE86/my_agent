from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlmodel import select

from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.image_history import ImageHistoryCard


class ImageHistoryDBService:
    async def list_all(self) -> List[Dict[str, Any]]:
        async with mysql_connector.session_scope() as session:
            res = await session.execute(select(ImageHistoryCard).order_by(ImageHistoryCard.created_at.desc()))
            out: List[Dict[str, Any]] = []
            for row in res.scalars().all():
                d = row.model_dump(exclude_none=True)
                # front-end expects `url`, prefer obs_url then fallback to doubao_url
                d["url"] = d.get("obs_url") or d.get("doubao_url")
                out.append(d)
            return out

    async def upsert_many(self, items: List[Dict[str, Any]]) -> None:
        """
        Replace-by-id behavior for each row. This mirrors the existing JSON overwrite behavior.
        """
        async with mysql_connector.session_scope() as session:
            for item in items:
                item_id = item.get("id")
                if not item_id:
                    continue
                # accept front-end `url` as doubao_url by default
                if "url" in item and "doubao_url" not in item and "obs_url" not in item:
                    item = dict(item)
                    item["doubao_url"] = item.pop("url")
                existing = await session.get(ImageHistoryCard, item_id)
                if existing is None:
                    session.add(ImageHistoryCard(**item))
                else:
                    for k, v in item.items():
                        setattr(existing, k, v)
            await session.commit()

    async def update_obs_url(self, item_id: str, new_url: str) -> None:
        async with mysql_connector.session_scope() as session:
            existing = await session.get(ImageHistoryCard, item_id)
            if existing is None:
                return
            existing.obs_url = new_url
            await session.commit()


image_history_db_service = ImageHistoryDBService()

