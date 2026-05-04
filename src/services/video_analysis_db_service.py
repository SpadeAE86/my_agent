from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

from sqlmodel import select, delete
from sqlalchemy import tuple_

from infra.logging.logger import logger as log
from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.video_analysis import (
    VideoAnalysisHistory,
    VideoAnalysisSceneFrames,
    VideoAnalysisSceneSplitFramesCache,
    VideoAnalysisShotCard,
    VideoAnalysisShotCardV2,
    VideoAnalysisVideoV2,
)

ShotCardsVersion = Literal["v1", "v2"]


def shot_card_v2_item_dict_from_scene(
    *,
    scene: Dict[str, Any],
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build one `cards[]` element for `upsert_history_item(..., shot_cards_version=\"v2\")`.
    Merges per-scene envelope (times only) with vision `analysis` (SceneAnalysisResultV2 fields).

    参考帧 URL 写入 ``item[\"scene_frames\"]``（按 scene_id），不放在分镜卡片列里。
    """
    a = analysis or {}
    return {
        "scene_id": int(scene.get("scene_id") or 0),
        "start_time": float(scene.get("start_time") or 0.0),
        "end_time": float(scene.get("end_time") or 0.0),
        "duration_seconds": float(scene.get("duration_seconds") or 0.0),
        "description": a.get("description"),
        "subject": a.get("subject"),
        "object": a.get("object"),
        "movement": a.get("movement"),
        "analysis_doc_id": a.get("id"),
        "id": a.get("id"),
        "car_model": a.get("car_model"),
        "frame_size": a.get("frame_size"),
        "resolution": a.get("resolution"),
        "video_duration": a.get("video_duration"),
        "footage_type": a.get("footage_type"),
        "shot_style": a.get("shot_style"),
        "shot_type": a.get("shot_type"),
        "camera_movement": a.get("camera_movement"),
        "scene_location": a.get("scene_location"),
        "car_color": a.get("car_color"),
        "product_status_scene": a.get("product_status_scene"),
        "has_presenter": a.get("has_presenter"),
        "generic_hq_road_run": a.get("generic_hq_road_run", False),
        "person_detail": a.get("person_detail"),
        "key_words": a.get("key_words"),
        "text": a.get("text"),
        "video_usage": a.get("video_usage"),
        "design_adjectives": a.get("design_adjectives"),
        "function_adjectives": a.get("function_adjectives"),
        "design_selling_points": a.get("design_selling_points"),
        "function_selling_points": a.get("function_selling_points"),
        "scenario_a": a.get("scenario_a"),
        "scenario_b": a.get("scenario_b"),
        "marketing_phrases": a.get("marketing_phrases"),
        "marketing_tags": a.get("marketing_tags"),
        "appealing_audience": a.get("appealing_audience"),
        "topic": a.get("topic"),
        "weather": a.get("weather"),
        "time": a.get("time"),
        "error": None,
    }


def _float_or_none(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _shot_card_v2_from_dict(video_db_id: int, c: Dict[str, Any]) -> VideoAnalysisShotCardV2:
    """Build a v2 ORM row from a plain dict (e.g. run_video merge of scene envelope + analysis)."""
    ghq = c.get("generic_hq_road_run", False)
    if isinstance(ghq, str):
        ghq = ghq.strip().lower() in ("true", "1", "yes", "是")
    else:
        ghq = bool(ghq)

    return VideoAnalysisShotCardV2(
        video_id=video_db_id,
        scene_id=int(c.get("scene_id") or 0),
        start_time=float(c.get("start_time") or 0.0),
        end_time=float(c.get("end_time") or 0.0),
        duration_seconds=float(c.get("duration_seconds") or 0.0),
        description=c.get("description"),
        subject=c.get("subject"),
        object=c.get("object"),
        movement=c.get("movement"),
        analysis_doc_id=c.get("analysis_doc_id") or c.get("id"),
        car_model=c.get("car_model"),
        frame_size=c.get("frame_size"),
        resolution=c.get("resolution"),
        video_duration=_float_or_none(c.get("video_duration")),
        footage_type=c.get("footage_type"),
        shot_style=c.get("shot_style"),
        shot_type=c.get("shot_type"),
        camera_movement=c.get("camera_movement"),
        scene_location=c.get("scene_location"),
        car_color=c.get("car_color"),
        product_status_scene=c.get("product_status_scene"),
        has_presenter=c.get("has_presenter"),
        generic_hq_road_run=ghq,
        person_detail=c.get("person_detail"),
        key_words=c.get("key_words"),
        text=c.get("text"),
        video_usage=c.get("video_usage"),
        design_adjectives=c.get("design_adjectives"),
        function_adjectives=c.get("function_adjectives"),
        design_selling_points=c.get("design_selling_points"),
        function_selling_points=c.get("function_selling_points"),
        scenario_a=c.get("scenario_a"),
        scenario_b=c.get("scenario_b"),
        marketing_phrases=c.get("marketing_phrases"),
        marketing_tags=c.get("marketing_tags"),
        appealing_audience=c.get("appealing_audience"),
        topic=c.get("topic"),
        weather=c.get("weather"),
        time=c.get("time"),
        error=c.get("error"),
        os_index_status=str(c.get("os_index_status") or "PENDING"),
        os_index_error=c.get("os_index_error"),
    )


def _normalize_scene_frames_map(raw: Any) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        try:
            sid = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, list):
            out[sid] = [str(x) for x in v if x]
        elif v:
            out[sid] = [str(v)]
    return out


def _collect_scene_frames_from_cards(cards: List[Any]) -> Dict[int, List[str]]:
    """Backward helper: if callers still put frame_urls on each card, fold into scene_frames."""
    m: Dict[int, List[str]] = {}
    for c in cards or []:
        if not isinstance(c, dict):
            continue
        sid = int(c.get("scene_id") or 0)
        if sid <= 0:
            continue
        fu = c.get("frame_urls")
        if isinstance(fu, list) and fu:
            m[sid] = [str(x) for x in fu if x]
    return m


def _v2_card_to_api_dict(
    row: VideoAnalysisShotCardV2,
    *,
    video_key: str,
    obs_video_url: Optional[str],
    frame_urls: List[str],
) -> Dict[str, Any]:
    d = row.model_dump(exclude_none=True)
    d["video_key"] = video_key
    d["history_id"] = video_key
    d["obs_video_url"] = obs_video_url
    d["frame_urls"] = list(frame_urls or [])
    return d


class VideoAnalysisDBService:
    async def resolve_video_v2_for_source_file(
        self,
        *,
        file_name: str,
    ) -> Tuple[int, str]:
        """
        为「源文件 basename」解析或创建 ``video_analysis_video_v2`` 行。

        同一 ``source_file_name``（basename）复用同一行；``video_key`` 新库为 ``str(id)``。
        与 ``video_source_upload_cache`` 无字段耦合；overwrite 时帧序列仍按 ``video_id`` 在业务层整表替换即可。

        返回 ``(0, \"\")`` 表示失败（调用方应中止分析）。
        """
        fn = os.path.basename((file_name or "").strip().replace("\\", "/"))
        if not fn:
            return 0, ""

        async with mysql_connector.session_scope() as session:
            try:
                res = await session.execute(
                    select(VideoAnalysisVideoV2).where(VideoAnalysisVideoV2.source_file_name == fn)
                )
                hit = res.scalar_one_or_none()
                if hit is not None and hit.id is not None:
                    return int(hit.id), str(hit.video_key)
            except Exception as e:
                log.warning("resolve_video_v2: lookup by source_file_name skipped (column missing?): %s", e)
                await session.rollback()

            ph = f"_tmp_{uuid.uuid4().hex[:24]}"
            row = VideoAnalysisVideoV2(video_key=ph, source_file_name=fn)
            session.add(row)
            await session.flush()
            await session.refresh(row)
            vid = int(row.id or 0)
            if vid <= 0:
                await session.rollback()
                return 0, ""
            row.video_key = str(vid)
            await session.commit()
            await session.refresh(row)
            return vid, str(row.video_key)

    async def get_or_create_video_v2(self, video_key: str) -> int:
        """
        兼容旧逻辑：按 ``video_key`` 字符串查找或创建一行。

        新流水线请优先用 ``resolve_video_v2_for_source_file``（``video_key == str(id)``）。
        """
        async with mysql_connector.session_scope() as session:
            res = await session.execute(select(VideoAnalysisVideoV2).where(VideoAnalysisVideoV2.video_key == video_key))
            existing = res.scalar_one_or_none()
            if existing is not None and existing.id is not None:
                return int(existing.id)
            row = VideoAnalysisVideoV2(video_key=video_key)
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return int(row.id or 0)

    async def get_split_frame_obs_cache(self, video_id: int) -> Dict[int, List[str]]:
        """读取 ``video_analysis_scene_split_frames_cache``：``scene_id`` -> OBS 帧 URL 列表。"""
        out: Dict[int, List[str]] = {}
        if video_id <= 0:
            return out
        try:
            async with mysql_connector.session_scope() as session:
                res = await session.execute(
                    select(VideoAnalysisSceneSplitFramesCache)
                    .where(VideoAnalysisSceneSplitFramesCache.video_id == int(video_id))
                    .order_by(VideoAnalysisSceneSplitFramesCache.scene_id.asc())
                )
                for r in res.scalars().all():
                    out[int(r.scene_id)] = list(r.obs_frame_url_list or [])
        except Exception as e:
            log.warning("get_split_frame_obs_cache skipped (table missing or DB error): %s", e)
        return out

    async def replace_split_frame_obs_cache_session(
        self,
        session: Any,
        video_id: int,
        scene_id_to_urls: Dict[int, List[str]],
    ) -> None:
        """
        与当前切分结果对齐：先删该 ``video_id`` 下全部行，再按 ``scene_id_to_urls`` 重建（不保留历史帧序列版本）。
        空 URL 列表的镜不会插入（等价于删除）。
        """
        if video_id <= 0:
            return
        await session.execute(
            delete(VideoAnalysisSceneSplitFramesCache).where(
                VideoAnalysisSceneSplitFramesCache.video_id == int(video_id)
            )
        )
        for sid, urls in sorted((scene_id_to_urls or {}).items(), key=lambda x: int(x[0])):
            sid_i = int(sid)
            if sid_i <= 0:
                continue
            clean = [str(u).strip() for u in (urls or []) if str(u).strip()]
            if not clean:
                continue
            session.add(
                VideoAnalysisSceneSplitFramesCache(
                    video_id=int(video_id),
                    scene_id=sid_i,
                    obs_frame_url_list=clean,
                )
            )

    async def replace_split_frame_obs_cache(self, video_id: int, scene_id_to_urls: Dict[int, List[str]]) -> None:
        """独立事务写入切分帧 OBS URL 缓存。"""
        if video_id <= 0:
            return
        try:
            async with mysql_connector.session_scope() as session:
                await self.replace_split_frame_obs_cache_session(session, video_id, scene_id_to_urls)
                await session.commit()
        except Exception as e:
            log.warning("replace_split_frame_obs_cache failed (table missing?): %s", e)

    async def list_history(self) -> List[Dict[str, Any]]:
        async with mysql_connector.session_scope() as session:
            res = await session.execute(
                select(VideoAnalysisHistory).order_by(VideoAnalysisHistory.created_at.desc())
            )
            return [row.model_dump(exclude_none=True) for row in res.scalars().all()]

    async def get_history_item(
        self,
        history_id: str,
        *,
        shot_cards_version: ShotCardsVersion = "v1",
    ) -> Optional[Dict[str, Any]]:
        async with mysql_connector.session_scope() as session:
            hist = await session.get(VideoAnalysisHistory, history_id)
            if hist is None:
                return None

            if shot_cards_version == "v2":
                vres = await session.execute(
                    select(VideoAnalysisVideoV2).where(VideoAnalysisVideoV2.video_key == history_id)
                )
                vrow = vres.scalar_one_or_none()
                if vrow is None or vrow.id is None:
                    item = hist.model_dump(exclude_none=True)
                    item["cards"] = []
                    item["shot_cards_version"] = shot_cards_version
                    return item

                vid = int(vrow.id)
                res = await session.execute(
                    select(VideoAnalysisShotCardV2)
                    .where(VideoAnalysisShotCardV2.video_id == vid)
                    .order_by(VideoAnalysisShotCardV2.scene_id.asc())
                )
                shot_rows = list(res.scalars().all())

                fres = await session.execute(
                    select(VideoAnalysisSceneFrames).where(VideoAnalysisSceneFrames.video_id == vid)
                )
                frames_by_sid = {int(r.scene_id): list(r.frame_paths or []) for r in fres.scalars().all()}

                obs_url = hist.video_url
                cards = [
                    _v2_card_to_api_dict(
                        c,
                        video_key=history_id,
                        obs_video_url=obs_url,
                        frame_urls=frames_by_sid.get(int(c.scene_id), []),
                    )
                    for c in shot_rows
                ]
            else:
                res = await session.execute(
                    select(VideoAnalysisShotCard)
                    .where(VideoAnalysisShotCard.history_id == history_id)
                    .order_by(VideoAnalysisShotCard.scene_id.asc())
                )
                cards = [c.model_dump(exclude_none=True) for c in res.scalars().all()]

            item = hist.model_dump(exclude_none=True)
            item["cards"] = cards
            item["shot_cards_version"] = shot_cards_version
            return item

    async def upsert_history_item(
        self,
        item: Dict[str, Any],
        *,
        shot_cards_version: ShotCardsVersion = "v1",
    ) -> None:
        """
        Replace-by-id behavior for a single history row and its cards.

        shot_cards_version:
        - v1: 写入 `video_analysis_shot_cards`（旧版字段）
        - v2: 写入 `video_analysis_shot_cards_v2` + `video_analysis_scene_frames_v2`（按 video_key 绑定自增 video_id）
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

            if shot_cards_version == "v2":
                vres = await session.execute(
                    select(VideoAnalysisVideoV2).where(VideoAnalysisVideoV2.video_key == history_id)
                )
                vrow = vres.scalar_one_or_none()
                if vrow is None:
                    vrow = VideoAnalysisVideoV2(video_key=str(history_id))
                    session.add(vrow)
                    await session.flush()
                elif vrow.id is None:
                    await session.flush()

                video_db_id = int(vrow.id or 0)
                if video_db_id <= 0:
                    await session.commit()
                    return

                scene_frames_in_request = "scene_frames" in item
                scene_frames = _normalize_scene_frames_map(item.get("scene_frames") or {})
                if not scene_frames:
                    scene_frames = _collect_scene_frames_from_cards(cards)

                # 仅更新 history（例如分析流水线开始时）时不要清空已有分镜与参考帧表。
                if not cards and not scene_frames:
                    await session.commit()
                    return

                # 仅有分镜理解字段、且请求体未显式带 scene_frames 键时：只替换 shot 行，保留已有参考帧表。
                # run_video 流水线会显式传 scene_frames（可为空 dict），不得走此分支，否则会永远不插入参考帧行。
                if cards and not scene_frames and not scene_frames_in_request:
                    await session.execute(
                        delete(VideoAnalysisShotCardV2).where(VideoAnalysisShotCardV2.video_id == video_db_id)
                    )
                    for c in cards:
                        if not isinstance(c, dict):
                            continue
                        session.add(_shot_card_v2_from_dict(video_db_id, c))
                    await session.commit()
                    return

                await session.execute(delete(VideoAnalysisShotCardV2).where(VideoAnalysisShotCardV2.video_id == video_db_id))
                await session.execute(
                    delete(VideoAnalysisSceneFrames).where(VideoAnalysisSceneFrames.video_id == video_db_id)
                )

                for sid, paths in sorted(scene_frames.items(), key=lambda x: x[0]):
                    if sid <= 0:
                        continue
                    session.add(
                        VideoAnalysisSceneFrames(
                            video_id=video_db_id,
                            scene_id=int(sid),
                            frame_paths=list(paths or []),
                        )
                    )

                try:
                    await self.replace_split_frame_obs_cache_session(session, video_db_id, scene_frames)
                except Exception as e:
                    log.warning("split frame obs cache sync in upsert skipped: %s", e)

                for c in cards:
                    if not isinstance(c, dict):
                        continue
                    session.add(_shot_card_v2_from_dict(video_db_id, c))
            else:
                await session.execute(delete(VideoAnalysisShotCard).where(VideoAnalysisShotCard.history_id == history_id))
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

    async def _v2_video_ids_for_keys(
        self, session, keys: List[Tuple[str, int]]
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        hids = list({h for h, _ in keys if h})
        if not hids:
            return {}, {}
        res = await session.execute(select(VideoAnalysisVideoV2).where(VideoAnalysisVideoV2.video_key.in_(hids)))
        key_to_id: Dict[str, int] = {}
        id_to_key: Dict[int, str] = {}
        for r in res.scalars().all():
            if r.id is not None and r.video_key:
                key_to_id[r.video_key] = int(r.id)
                id_to_key[int(r.id)] = r.video_key
        return key_to_id, id_to_key

    async def list_all_cards(self, *, shot_cards_version: ShotCardsVersion = "v1") -> List[Dict[str, Any]]:
        """
        Return all shot cards across all histories.
        """
        async with mysql_connector.session_scope() as session:
            if shot_cards_version == "v2":
                res = await session.execute(
                    select(VideoAnalysisShotCardV2).order_by(
                        VideoAnalysisShotCardV2.video_id.desc(),
                        VideoAnalysisShotCardV2.scene_id.asc(),
                    )
                )
                shot_rows = list(res.scalars().all())
                if not shot_rows:
                    return []

                vids = list({r.video_id for r in shot_rows})
                vres = await session.execute(select(VideoAnalysisVideoV2).where(VideoAnalysisVideoV2.id.in_(vids)))
                id_to_key = {int(v.id): v.video_key for v in vres.scalars().all() if v.id is not None}

                keys_hist = list({id_to_key.get(r.video_id, "") for r in shot_rows if id_to_key.get(r.video_id)})
                hist_map: Dict[str, Optional[str]] = {}
                for hk in keys_hist:
                    if not hk:
                        continue
                    h = await session.get(VideoAnalysisHistory, hk)
                    hist_map[hk] = h.video_url if h else None

                fres = await session.execute(
                    select(VideoAnalysisSceneFrames).where(VideoAnalysisSceneFrames.video_id.in_(vids))
                )
                frames_lookup: Dict[Tuple[int, int], List[str]] = {}
                for fr in fres.scalars().all():
                    frames_lookup[(int(fr.video_id), int(fr.scene_id))] = list(fr.frame_paths or [])

                out: List[Dict[str, Any]] = []
                for c in shot_rows:
                    vk = id_to_key.get(c.video_id, "")
                    obs_url = hist_map.get(vk)
                    fus = frames_lookup.get((int(c.video_id), int(c.scene_id)), [])
                    out.append(_v2_card_to_api_dict(c, video_key=vk, obs_video_url=obs_url, frame_urls=fus))
                return out

            res = await session.execute(
                select(VideoAnalysisShotCard).order_by(
                    VideoAnalysisShotCard.history_id.desc(),
                    VideoAnalysisShotCard.scene_id.asc(),
                )
            )
            return [c.model_dump(exclude_none=True) for c in res.scalars().all()]

    async def get_cards_by_keys(
        self,
        keys: List[tuple[str, int]],
        *,
        shot_cards_version: ShotCardsVersion = "v1",
    ) -> List[Dict[str, Any]]:
        """
        Fetch shot cards by (video_key, scene_id) pairs — ``video_key`` 与历史表 ``id`` / OpenSearch 前缀一致。
        Returns cards in arbitrary DB order; caller can reorder.
        """
        keys = [(hid, int(sid)) for (hid, sid) in (keys or []) if hid]
        if not keys:
            return []

        async with mysql_connector.session_scope() as session:
            if shot_cards_version == "v2":
                key_to_id, _ = await self._v2_video_ids_for_keys(session, keys)
                triples: List[Tuple[int, int]] = []
                for hk, sid in keys:
                    vid = key_to_id.get(hk)
                    if vid is None:
                        continue
                    triples.append((vid, sid))
                if not triples:
                    return []

                res = await session.execute(
                    select(VideoAnalysisShotCardV2).where(
                        tuple_(VideoAnalysisShotCardV2.video_id, VideoAnalysisShotCardV2.scene_id).in_(triples)
                    )
                )
                rows = list(res.scalars().all())
                if not rows:
                    return []

                vids = list({r.video_id for r in rows})
                vk_res = await session.execute(select(VideoAnalysisVideoV2).where(VideoAnalysisVideoV2.id.in_(vids)))
                id_to_key = {int(v.id): v.video_key for v in vk_res.scalars().all() if v.id is not None}

                fres = await session.execute(
                    select(VideoAnalysisSceneFrames).where(VideoAnalysisSceneFrames.video_id.in_(vids))
                )
                frames_lookup: Dict[Tuple[int, int], List[str]] = {}
                for fr in fres.scalars().all():
                    frames_lookup[(int(fr.video_id), int(fr.scene_id))] = list(fr.frame_paths or [])

                out: List[Dict[str, Any]] = []
                for r in rows:
                    vk = id_to_key.get(int(r.video_id), "")
                    h = await session.get(VideoAnalysisHistory, vk) if vk else None
                    obs_url = h.video_url if h else None
                    fus = frames_lookup.get((int(r.video_id), int(r.scene_id)), [])
                    out.append(_v2_card_to_api_dict(r, video_key=vk, obs_video_url=obs_url, frame_urls=fus))
                return out

            model = VideoAnalysisShotCard
            res = await session.execute(select(model).where(tuple_(model.history_id, model.scene_id).in_(keys)))
            return [c.model_dump(exclude_none=True) for c in res.scalars().all()]

    async def update_cards_index_status(
        self,
        keys: List[tuple[str, int]],
        *,
        status: str,
        error: Optional[str] = None,
        shot_cards_version: ShotCardsVersion = "v1",
    ) -> int:
        """
        Update os_index_status / os_index_error for given (video_key, scene_id) pairs.
        Returns affected rows count (best effort).
        """
        keys = [(hid, int(sid)) for (hid, sid) in (keys or []) if hid]
        if not keys:
            return 0

        async with mysql_connector.session_scope() as session:
            if shot_cards_version == "v2":
                key_to_id, _ = await self._v2_video_ids_for_keys(session, keys)
                triples = [(key_to_id[h], sid) for h, sid in keys if h in key_to_id]
                if not triples:
                    return 0
                res = await session.execute(
                    select(VideoAnalysisShotCardV2).where(
                        tuple_(VideoAnalysisShotCardV2.video_id, VideoAnalysisShotCardV2.scene_id).in_(triples)
                    )
                )
                rows = res.scalars().all()
            else:
                model = VideoAnalysisShotCard
                res = await session.execute(select(model).where(tuple_(model.history_id, model.scene_id).in_(keys)))
                rows = res.scalars().all()
            for r in rows:
                r.os_index_status = status
                r.os_index_error = error
            await session.commit()
            return len(rows)


video_analysis_db_service = VideoAnalysisDBService()
