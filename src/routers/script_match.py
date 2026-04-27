from __future__ import annotations

from fastapi import APIRouter

from models.pydantic.script_match_request import ScriptMatchRequest
from services.script_rewrite_service import rewrite_script_to_storyboard_and_tags
from services.script_match_service import match_script_tags_segments


script_match_router = APIRouter(prefix="/script", tags=["script"])


@script_match_router.post("/match")
async def match_script(req: ScriptMatchRequest):
    """
    End-to-end:
    script -> (stage1 storyboard + stage2 tags) -> OpenSearch topK -> DB video_path
    """
    storyboard, tags = await rewrite_script_to_storyboard_and_tags(req.script, index=0)
    matches = await match_script_tags_segments(
        [s.model_dump(exclude_none=True) for s in tags.segment_result],
        top_k=req.top_k,
        search_pipeline=req.search_pipeline,
        mode=req.mode,
    )
    return {
        "success": True,
        "storyboard": storyboard.model_dump(exclude_none=True),
        "tags": tags.model_dump(exclude_none=True),
        "matches": matches,
    }

