"""
AND token join 模板：控制 segment 哪些字段生成 SearchToken 时为 AND，并在 video-analysis/search 中映射为 term filter。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import update
from sqlmodel import select

from infra.logging.logger import logger as log
from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.video_analysis import VideoAnalysisTokenJoinTemplate

# 与 car_interior_analysis_v2 中 Keyword 标量字段对齐（subject/description 等为 Text，不参与 term filter）
TOKEN_JOIN_TERM_FIELDS_V2: frozenset[str] = frozenset(
    {
        "car_model",
        "frame_size",
        "frame_orientation",
        "resolution",
        "footage_type",
        "shot_style",
        "shot_type",
        "car_color",
        "product_status_scene",
        "movement",
        "camera_movement",
        "topic",
        "weather",
        "time",
    }
)

DEFAULT_AND_SEGMENT_FIELDS: List[str] = ["car_model", "movement", "product_status_scene"]


def normalize_v2_term_filter_value(field: str, value: str) -> str:
    """
    将 UI / 脚本侧常用取值映射为 OpenSearch keyword（与入库或派生字段一致）。
    """
    f = (field or "").strip()
    v = (value or "").strip()
    if not v:
        return v
    if f == "frame_size":
        m = {
            "竖版9:16": "9:16",
            "竖屏9:16": "9:16",
            "横版16:9": "16:9",
            "横屏16:9": "16:9",
        }
        return m.get(v, v)
    if f == "frame_orientation":
        alias = {
            "portrait": "竖屏",
            "landscape": "横屏",
            "vertical": "竖屏",
            "horizontal": "横屏",
            "竖版": "竖屏",
            "横版": "横屏",
        }
        vl = v.lower()
        if vl in alias:
            return alias[vl]
        if v in alias:
            return alias[v]
        return v
    return v


def normalize_and_segment_fields(fields: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for f in fields or []:
        k = str(f or "").strip()
        if not k or k in seen:
            continue
        if k not in TOKEN_JOIN_TERM_FIELDS_V2:
            continue
        seen.add(k)
        out.append(k)
    return out


async def list_templates(*, workspace: Optional[str] = None) -> List[Dict[str, Any]]:
    async with mysql_connector.session_scope() as session:
        stmt = select(VideoAnalysisTokenJoinTemplate).order_by(
            VideoAnalysisTokenJoinTemplate.workspace,
            VideoAnalysisTokenJoinTemplate.id,
        )
        if workspace and workspace.strip():
            stmt = stmt.where(VideoAnalysisTokenJoinTemplate.workspace == workspace.strip())
        res = await session.execute(stmt)
        rows = list(res.scalars().all())
    return [r.model_dump() for r in rows]


async def get_default_and_fields(workspace: str) -> List[str]:
    """返回指定 workspace 默认模板的 and_segment_fields；无则返回内置默认。"""
    ws = (workspace or "v2").strip() or "v2"
    async with mysql_connector.session_scope() as session:
        stmt = (
            select(VideoAnalysisTokenJoinTemplate)
            .where(
                VideoAnalysisTokenJoinTemplate.workspace == ws,
                VideoAnalysisTokenJoinTemplate.is_default == True,  # noqa: E712
            )
            .limit(1)
        )
        res = await session.execute(stmt)
        row = res.scalars().first()
        if row and row.and_segment_fields:
            normalized = normalize_and_segment_fields(row.and_segment_fields)
            if normalized:
                return normalized
    log.debug("token_join_template: no default for workspace=%s, using built-in", ws)
    return list(DEFAULT_AND_SEGMENT_FIELDS)


async def get_template_by_id(template_id: int) -> Optional[VideoAnalysisTokenJoinTemplate]:
    async with mysql_connector.session_scope() as session:
        return await session.get(VideoAnalysisTokenJoinTemplate, template_id)


async def create_template(
    *,
    name: str,
    workspace: str,
    and_segment_fields: List[str],
    is_default: bool,
) -> VideoAnalysisTokenJoinTemplate:
    ws = (workspace or "v2").strip() or "v2"
    fields = normalize_and_segment_fields(and_segment_fields)
    row = VideoAnalysisTokenJoinTemplate(
        name=(name or "").strip() or "未命名",
        workspace=ws,
        and_segment_fields=fields,
        is_default=bool(is_default),
    )
    async with mysql_connector.session_scope() as session:
        if row.is_default:
            await session.execute(
                update(VideoAnalysisTokenJoinTemplate)
                .where(VideoAnalysisTokenJoinTemplate.workspace == ws)
                .values(is_default=False)
            )
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def update_template(
    template_id: int,
    *,
    name: Optional[str] = None,
    and_segment_fields: Optional[List[str]] = None,
    is_default: Optional[bool] = None,
) -> Optional[VideoAnalysisTokenJoinTemplate]:
    async with mysql_connector.session_scope() as session:
        row = await session.get(VideoAnalysisTokenJoinTemplate, template_id)
        if row is None:
            return None
        ws = row.workspace
        if name is not None:
            row.name = (name or "").strip() or row.name
        if and_segment_fields is not None:
            row.and_segment_fields = normalize_and_segment_fields(and_segment_fields)
        if is_default is not None:
            row.is_default = bool(is_default)
            if row.is_default:
                await session.execute(
                    update(VideoAnalysisTokenJoinTemplate)
                    .where(
                        VideoAnalysisTokenJoinTemplate.workspace == ws,
                        VideoAnalysisTokenJoinTemplate.id != template_id,
                    )
                    .values(is_default=False)
                )
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def delete_template(template_id: int) -> bool:
    async with mysql_connector.session_scope() as session:
        row = await session.get(VideoAnalysisTokenJoinTemplate, template_id)
        if row is None:
            return False
        await session.delete(row)
        await session.commit()
    return True


async def set_default_template(template_id: int) -> Optional[VideoAnalysisTokenJoinTemplate]:
    async with mysql_connector.session_scope() as session:
        row = await session.get(VideoAnalysisTokenJoinTemplate, template_id)
        if row is None:
            return None
        ws = row.workspace
        await session.execute(
            update(VideoAnalysisTokenJoinTemplate)
            .where(VideoAnalysisTokenJoinTemplate.workspace == ws)
            .values(is_default=False)
        )
        row.is_default = True
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def seed_token_join_templates_if_empty() -> None:
    """无模板时写入 v1/v2 默认行（幂等）。"""
    async with mysql_connector.session_scope() as session:
        res = await session.execute(select(VideoAnalysisTokenJoinTemplate.id).limit(1))
        if res.first():
            return
    for ws in ("v1", "v2"):
        await create_template(
            name="默认",
            workspace=ws,
            and_segment_fields=list(DEFAULT_AND_SEGMENT_FIELDS),
            is_default=True,
        )
    log.info("Seeded default video_analysis_token_join_template for v1/v2")
