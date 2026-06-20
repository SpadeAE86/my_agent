from typing import Any, Dict, Literal
from fastapi import APIRouter, Query, HTTPException
from sqlmodel import select

from models.sqlmodel.volco_timbre import VolcoTimbre
from infra.storage.mysql_connector import mysql_connector
from infra.logging.logger import logger as log

audio_router = APIRouter(prefix="/audio", tags=["audio"])

@audio_router.get("/models")
async def list_voice_models(
    model_type: Literal["all", "big", "small"] = Query(default="all", description="Filter model family"),
) -> Dict[str, Any]:
    """
    获取所有的音色列表，并按优先级与模型类型排序、去重
    """
    try:
        async with mysql_connector.session_scope() as session:
            # 查询所有启用的音色，并按 priority DESC, voice_model_type ASC, voice_character ASC 排序
            stmt = select(VolcoTimbre).where(VolcoTimbre.is_enabled == 1).order_by(
                VolcoTimbre.priority.desc(),
                VolcoTimbre.voice_model_type.asc(),
                VolcoTimbre.voice_character.asc()
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        seen = set()
        unique_rows = []
        for row in rows:
            # 兼容空类型处理
            actual_model_type = row.voice_model_type
            if not actual_model_type:
                actual_model_type = "small" if row.voice_character == "天才少女" else "big"

            if model_type != "all" and actual_model_type != model_type:
                continue

            pair = (row.voice_character, actual_model_type)
            if pair not in seen:
                seen.add(pair)
                unique_rows.append((row, actual_model_type))

        def _to_item(row: VolcoTimbre, model_type_val: str) -> dict[str, Any]:
            return {
                "id": row.id,
                "voice_character": row.voice_character,
                "voice_code": row.voice_code,
                "voice_model_type": model_type_val,
                "note": row.note,
                "is_enabled": bool(row.is_enabled),
                "priority": row.priority,
                "age_type": row.age_type,
                "sex": row.sex,
                "full_voice": row.full_voice,
            }

        if model_type == "all":
            grouped: dict[str, list[dict[str, Any]]] = {"big": [], "small": []}
            for row, model_type_val in unique_rows:
                grouped.setdefault(model_type_val, []).append(_to_item(row, model_type_val))
            data: Any = grouped
        else:
            data = [_to_item(row, model_type_val) for row, model_type_val in unique_rows]

        return {"code": 200, "message": "ok", "data": data}
    except Exception as e:
        log.error(f"查询音色列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询音色列表失败: {str(e)}")
