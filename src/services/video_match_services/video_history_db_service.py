"""
video_history_db_service.py
复用 image_history_cards 表存储视频生成历史。
type 字段为 t2v / i2v，与图像 (t2i / i2i) 区分。
numeric_id 在启动时通过 ensure_video_auto_increment_offset() 对齐到
offset=10_000，确保视频 ID 与图像 ID 不冲突。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import or_, update, func, text
from sqlmodel import select

from infra.storage.mysql_connector import mysql_connector
from infra.logging.logger import logger as log
from models.sqlmodel.image_history import ImageHistoryCard
from utils.api_datetime import attach_image_row_duration_ms, normalize_row_utc_iso

_VIDEO_TYPES = ("t2v", "i2v")

_ALLOWED_UPSERT_KEYS = {
    "prompt",
    "model",
    "size",
    "resolution",
    "ratio",
    "duration",
    "doubao_url",
    "obs_url",
    "time",
    "type",
    "referenceMedia",
    "error",
    "taskId",
    "status",
    "request_id",
    "current_run_started_at",
}

def _row_to_api(row: ImageHistoryCard) -> Dict[str, Any]:
    d = row.model_dump(exclude_none=True)
    d.pop("legacy_id", None)
    nid = d.pop("numeric_id", None)
    d["id"] = str(nid) if nid is not None else ""
    d["url"] = d.get("obs_url") or d.get("doubao_url")
    return attach_image_row_duration_ms(normalize_row_utc_iso(d))


class VideoHistoryDBService:
    async def mark_interrupted_running_as_failed(self, reason: str) -> int:
        """启动恢复：将 t2v/i2v 类型的 running/pending 行标为 failed。"""
        async with mysql_connector.session_scope() as session:
            stmt = (
                update(ImageHistoryCard)
                .where(
                    ImageHistoryCard.type.in_(list(_VIDEO_TYPES)),
                    func.lower(func.coalesce(ImageHistoryCard.status, "")).in_(
                        ["running", "pending", "processing"]
                    ),
                )
                .values(status="failed", error=reason)
            )
            res = await session.execute(stmt)
            await session.commit()
            return int(res.rowcount or 0)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def _fetch_one(
        self, session: Any, key: str
    ) -> Optional[ImageHistoryCard]:
        k = (key or "").strip()
        if not k:
            return None
        if k.isdigit():
            row = await session.get(ImageHistoryCard, int(k))
            if row and row.type in _VIDEO_TYPES:
                return row
            return None
        stmt = select(ImageHistoryCard).where(
            or_(ImageHistoryCard.legacy_id == k, ImageHistoryCard.taskId == k),
            ImageHistoryCard.type.in_(list(_VIDEO_TYPES)),
        )
        res = await session.execute(stmt)
        return res.scalars().first()

    async def list_all(self, ids: Optional[str] = None) -> List[Dict[str, Any]]:
        async with mysql_connector.session_scope() as session:
            stmt = (
                select(ImageHistoryCard)
                .where(ImageHistoryCard.type.in_(list(_VIDEO_TYPES)))
                .order_by(ImageHistoryCard.created_at.desc())
            )
            ids_str = (ids or "").strip()
            if ids_str:
                id_list = [i.strip() for i in ids_str.split(",") if i.strip()]
                if id_list:
                    stmt = stmt.where(ImageHistoryCard.numeric_id.in_([int(i) for i in id_list if i.isdigit()]))
            res = await session.execute(stmt)
            return [_row_to_api(row) for row in res.scalars().all()]

    async def get_by_id_or_task_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        key = (item_id or "").strip()
        if not key:
            return None
        async with mysql_connector.session_scope() as session:
            hit = await self._fetch_one(session, key)
            if hit is None:
                return None
            return _row_to_api(hit)

    async def upsert_many(self, items: List[Dict[str, Any]]) -> None:
        async with mysql_connector.session_scope() as session:
            for raw in items:
                ext_id = raw.get("id")
                if not ext_id:
                    continue
                item_id = str(ext_id).strip()
                item = dict(raw)
                if "url" in item and "doubao_url" not in item and "obs_url" not in item:
                    item["doubao_url"] = item.pop("url")

                # Strictly filter: only allow video types
                t = item.get("type")
                if t and t not in _VIDEO_TYPES:
                    continue

                existing = await self._fetch_one(session, item_id)
                if existing is None:
                    # When inserting a new record, the type must be a video type
                    t = item.get("type")
                    if not t or t not in _VIDEO_TYPES:
                        continue
                    kwargs: Dict[str, Any] = {"legacy_id": item_id}
                    for k, v in item.items():
                        if k == "id" or k not in _ALLOWED_UPSERT_KEYS:
                            continue
                        kwargs[k] = v
                    session.add(ImageHistoryCard(**kwargs))
                else:
                    for k, v in item.items():
                        if k == "id":
                            continue
                        if k not in _ALLOWED_UPSERT_KEYS:
                            continue
                        setattr(existing, k, v)
            await session.commit()

    async def update_obs_url(self, item_id: str, new_url: str) -> None:
        async with mysql_connector.session_scope() as session:
            existing = await self._fetch_one(session, item_id)
            if existing is None:
                return
            existing.obs_url = new_url
            await session.commit()

    async def delete_by_id(self, item_id: str) -> bool:
        async with mysql_connector.session_scope() as session:
            existing = await self._fetch_one(session, item_id)
            if existing is None:
                return False
            await session.delete(existing)
            await session.commit()
            return True


video_history_db_service = VideoHistoryDBService()
