from __future__ import annotations

import os
from typing import Any, Dict, Optional

from sqlmodel import select

from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.video_upload_cache import VideoSourceUploadCache


def _normalize_file_name(file_name: str) -> str:
    fn = (file_name or "").strip()
    if not fn:
        return ""
    return os.path.basename(fn.replace("\\", "/"))


class VideoUploadCacheService:
    async def get_by_file_name(self, file_name: str) -> Optional[Dict[str, Any]]:
        fn = _normalize_file_name(file_name)
        if not fn:
            return None
        async with mysql_connector.session_scope() as session:
            res = await session.execute(select(VideoSourceUploadCache).where(VideoSourceUploadCache.file_name == fn))
            row = res.scalars().first()
            return row.model_dump(exclude_none=True) if row else None

    async def upsert(
        self,
        *,
        file_name: str,
        abs_path: Optional[str] = None,
        file_size: Optional[int] = None,
        file_mtime: Optional[int] = None,
        obs_key: Optional[str],
        obs_url: Optional[str],
    ) -> None:
        fn = _normalize_file_name(file_name)
        if not fn:
            return
        async with mysql_connector.session_scope() as session:
            res = await session.execute(select(VideoSourceUploadCache).where(VideoSourceUploadCache.file_name == fn))
            existing = res.scalars().first()
            if existing is None:
                session.add(
                    VideoSourceUploadCache(
                        file_name=fn,
                        abs_path=abs_path,
                        file_size=int(file_size) if file_size is not None else None,
                        file_mtime=int(file_mtime) if file_mtime is not None else None,
                        obs_key=obs_key,
                        obs_url=obs_url,
                    )
                )
            else:
                if abs_path:
                    existing.abs_path = abs_path
                if file_size is not None:
                    existing.file_size = int(file_size)
                if file_mtime is not None:
                    existing.file_mtime = int(file_mtime)
                if obs_key:
                    existing.obs_key = obs_key
                if obs_url:
                    existing.obs_url = obs_url
            await session.commit()


video_upload_cache_service = VideoUploadCacheService()
