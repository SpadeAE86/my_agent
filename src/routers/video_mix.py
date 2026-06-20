from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.video_compose_services import get_mix_compose_job, start_mix_compose_for_job


class MixComposeBody(BaseModel):
    job_id: str = Field(..., min_length=1, description="video_match_job.id")
    mock: Optional[bool] = Field(
        default=None,
        description="覆盖 mix_compose.mock；null 则读配置文件",
    )
    prefer_srt: bool = Field(
        default=False,
        description="true 时混剪请求体不含 cap_config，完成后返回 result_srt_text",
    )


video_mix_router = APIRouter(prefix="/video-mix", tags=["video-mix"])


@video_mix_router.post("/compose")
async def create_mix_compose(body: MixComposeBody):
    try:
        return await start_mix_compose_for_job(body.job_id, mix_mock=body.mock, prefer_srt=body.prefer_srt)
    except ValueError as e:
        msg = str(e)
        if msg == "job not found":
            raise HTTPException(status_code=404, detail=msg) from e
        raise HTTPException(status_code=400, detail=msg) from e


@video_mix_router.get("/compose/{compose_id}")
async def read_mix_compose(compose_id: str):
    data = await get_mix_compose_job(compose_id)
    if data is None:
        raise HTTPException(status_code=404, detail="compose job not found")
    return data
