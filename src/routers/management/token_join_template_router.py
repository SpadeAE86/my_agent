from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

from services.video_match_services.token_join_template_service import (
    TOKEN_JOIN_TERM_FIELDS_V2,
    create_template,
    delete_template,
    get_default_and_fields,
    list_templates,
    set_default_template,
    update_template,
)

router = APIRouter()

class TokenJoinTemplateCreate(BaseModel):
    name: str
    workspace: str = "v2"
    and_segment_fields: List[str] = Field(default_factory=list)
    is_default: bool = False

class TokenJoinTemplateUpdate(BaseModel):
    name: Optional[str] = None
    and_segment_fields: Optional[List[str]] = None
    is_default: Optional[bool] = None

@router.get("/token-join-templates/allowed-fields")
async def token_join_allowed_fields():
    """可作 AND term filter 的 v2 索引 keyword 字段（与 segment 字段名一致）。"""
    return {"success": True, "fields": sorted(TOKEN_JOIN_TERM_FIELDS_V2)}

@router.get("/token-join-templates/default-fields")
async def token_join_default_fields(workspace: str = Query("v2")):
    fields = await get_default_and_fields(workspace)
    return {"success": True, "workspace": (workspace or "v2").strip() or "v2", "and_segment_fields": fields}

@router.get("/token-join-templates")
async def token_join_templates_list(workspace: Optional[str] = Query(None)):
    rows = await list_templates(workspace=workspace)
    return {"success": True, "templates": rows}

@router.post("/token-join-templates")
async def token_join_templates_create(req: TokenJoinTemplateCreate):
    row = await create_template(
        name=req.name,
        workspace=req.workspace,
        and_segment_fields=req.and_segment_fields,
        is_default=req.is_default,
    )
    return {"success": True, "template": row.model_dump()}

@router.put("/token-join-templates/{template_id}")
async def token_join_templates_put(template_id: int, req: TokenJoinTemplateUpdate):
    row = await update_template(
        template_id,
        name=req.name,
        and_segment_fields=req.and_segment_fields,
        is_default=req.is_default,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="template not found")
    return {"success": True, "template": row.model_dump()}

@router.delete("/token-join-templates/{template_id}")
async def token_join_templates_delete(template_id: int):
    ok = await delete_template(template_id)
    if not ok:
        raise HTTPException(status_code=404, detail="template not found")
    return {"success": True}

@router.post("/token-join-templates/{template_id}/set-default")
async def token_join_templates_set_default_route(template_id: int):
    row = await set_default_template(template_id)
    if row is None:
        raise HTTPException(status_code=404, detail="template not found")
    return {"success": True, "template": row.model_dump()}
