from __future__ import annotations

from typing import Optional, Dict, Any

from sqlmodel import select

from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.video_upload_cache import VideoSourceUploadCache


class VideoUploadCacheService:
    async def get_by_sig(self, sig: str) -> Optional[Dict[str, Any]]:
        sig = (sig or "").strip()
        if not sig:
            return None
        async with mysql_connector.session_scope() as session:
            row = await session.get(VideoSourceUploadCache, sig)
            return row.model_dump(exclude_none=True) if row else None

    async def upsert(
        self,
        *,
        sig: str,
        file_name: str,
        abs_path: str,
        file_size: int,
        file_mtime: int,
        obs_key: Optional[str],
        obs_url: Optional[str],
    ) -> None:
        sig = (sig or "").strip()
        if not sig:
            return
        async with mysql_connector.session_scope() as session:
            existing = await session.get(VideoSourceUploadCache, sig)
            if existing is None:
                session.add(
                    VideoSourceUploadCache(
                        sig=sig,
                        file_name=file_name,
                        abs_path=abs_path,
                        file_size=int(file_size or 0),
                        file_mtime=int(file_mtime or 0),
                        obs_key=obs_key,
                        obs_url=obs_url,
                    )
                )
            else:
                existing.file_name = file_name or existing.file_name
                existing.abs_path = abs_path or existing.abs_path
                existing.file_size = int(file_size or existing.file_size or 0)
                existing.file_mtime = int(file_mtime or existing.file_mtime or 0)
                existing.obs_key = obs_key or existing.obs_key
                existing.obs_url = obs_url or existing.obs_url
            await session.commit()


video_upload_cache_service = VideoUploadCacheService()

