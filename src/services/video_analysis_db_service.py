from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlmodel import select, delete
from sqlalchemy import tuple_

from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.video_analysis import VideoAnalysisHistory, VideoAnalysisShotCard


class VideoAnalysisDBService:
    async def list_history(self) -> List[Dict[str, Any]]:
        async with mysql_connector.session_scope() as session:
            res = await session.execute(
                select(VideoAnalysisHistory).order_by(VideoAnalysisHistory.created_at.desc())
            )
            return [row.model_dump(exclude_none=True) for row in res.scalars().all()]

    async def get_history_item(self, history_id: str) -> Optional[Dict[str, Any]]:
        async with mysql_connector.session_scope() as session:
            hist = await session.get(VideoAnalysisHistory, history_id)
            if hist is None:
                return None

            res = await session.execute(
                select(VideoAnalysisShotCard)
                .where(VideoAnalysisShotCard.history_id == history_id)
                .order_by(VideoAnalysisShotCard.scene_id.asc())
            )
            cards = [c.model_dump(exclude_none=True) for c in res.scalars().all()]

            item = hist.model_dump(exclude_none=True)
            item["cards"] = cards
            return item

    async def upsert_history_item(self, item: Dict[str, Any]) -> None:
        """
        Replace-by-id behavior for a single history row and its cards.
        """
        history_id = item.get("id")
        if not history_id:
            return

        cards = item.get("cards") or []

        async with mysql_connector.session_scope() as session:
            existing = await session.get(VideoAnalysisHistory, history_id)
            if existing is None:
                session.add(
                    VideoAnalysisHistory(
                        id=history_id,
                        name=item.get("name") or "",
                        time=item.get("time") or "",
                        video_url=item.get("video_url"),
                    )
                )
            else:
                existing.name = item.get("name") or existing.name
                existing.time = item.get("time") or existing.time
                existing.video_url = item.get("video_url", existing.video_url)

            # delete old cards then insert new ones
            await session.execute(
                delete(VideoAnalysisShotCard).where(VideoAnalysisShotCard.history_id == history_id)
            )

            for c in cards:
                if not isinstance(c, dict):
                    continue
                session.add(
                    VideoAnalysisShotCard(
                        history_id=history_id,
                        scene_id=int(c.get("scene_id") or 0),
                        start_time=float(c.get("start_time") or 0.0),
                        end_time=float(c.get("end_time") or 0.0),
                        duration_seconds=float(c.get("duration_seconds") or 0.0),
                        thumbnail=c.get("thumbnail"),
                        frame_urls=c.get("frame_urls"),
                        description=c.get("description"),
                        subject=c.get("subject"),
                        object=c.get("object"),
                        movement=c.get("movement"),
                        adjective=c.get("adjective"),
                        search_tags=c.get("search_tags"),
                        marketing_tags=c.get("marketing_tags"),
                        appealing_audience=c.get("appealing_audience"),
                        visual_quality=c.get("visual_quality"),
                        error=c.get("error"),
                        # index status defaults to PENDING; allow override on upsert
                        os_index_status=str(c.get("os_index_status") or "PENDING"),
                        os_index_error=c.get("os_index_error"),
                    )
                )

            await session.commit()

    async def overwrite_history(self, history: List[Dict[str, Any]]) -> None:
        """
        Mirrors the existing JSON overwrite endpoint:
        it overwrites/updates each history item by id.
        """
        for item in history:
            await self.upsert_history_item(item)

    async def list_all_cards(self) -> List[Dict[str, Any]]:
        """
        Return all shot cards across all histories.
        """
        async with mysql_connector.session_scope() as session:
            res = await session.execute(
                select(VideoAnalysisShotCard).order_by(
                    VideoAnalysisShotCard.history_id.desc(),
                    VideoAnalysisShotCard.scene_id.asc(),
                )
            )
            return [c.model_dump(exclude_none=True) for c in res.scalars().all()]

    async def get_cards_by_keys(self, keys: List[tuple[str, int]]) -> List[Dict[str, Any]]:
        """
        Fetch shot cards by (history_id, scene_id) pairs.
        Returns cards in arbitrary DB order; caller can reorder.
        """
        keys = [(hid, int(sid)) for (hid, sid) in (keys or []) if hid]
        if not keys:
            return []

        async with mysql_connector.session_scope() as session:
            res = await session.execute(
                select(VideoAnalysisShotCard).where(
                    tuple_(VideoAnalysisShotCard.history_id, VideoAnalysisShotCard.scene_id).in_(keys)
                )
            )
            return [c.model_dump(exclude_none=True) for c in res.scalars().all()]

    async def update_cards_index_status(
        self,
        keys: List[tuple[str, int]],
        *,
        status: str,
        error: Optional[str] = None,
    ) -> int:
        """
        Update os_index_status / os_index_error for given (history_id, scene_id) pairs.
        Returns affected rows count (best effort).
        """
        keys = [(hid, int(sid)) for (hid, sid) in (keys or []) if hid]
        if not keys:
            return 0

        async with mysql_connector.session_scope() as session:
            res = await session.execute(
                select(VideoAnalysisShotCard).where(
                    tuple_(VideoAnalysisShotCard.history_id, VideoAnalysisShotCard.scene_id).in_(keys)
                )
            )
            rows = res.scalars().all()
            for r in rows:
                r.os_index_status = status
                r.os_index_error = error
            await session.commit()
            return len(rows)


video_analysis_db_service = VideoAnalysisDBService()

