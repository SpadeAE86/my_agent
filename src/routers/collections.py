import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import select

from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.collections import ThemeSpace, CollectionItem

collections_router = APIRouter(prefix="/collections", tags=["collections"])


# ─── Pydantic Request Models ──────────────────────────────────────
class ToggleFavoriteRequest(BaseModel):
    item_type: str  # media / template / prompt
    title: str
    cover_url: Optional[str] = None
    data: Dict[str, Any]
    space_id: Optional[str] = None
    tags: Optional[List[str]] = None


class ThemeSpaceCreateRequest(BaseModel):
    name: str
    category: str  # media / template / prompt
    description: Optional[str] = None
    space_tags: Optional[List[str]] = None


class ThemeSpaceUpdateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    space_tags: Optional[List[str]] = None


class MoveToSpaceRequest(BaseModel):
    space_id: Optional[str] = None


class BatchMoveToSpaceRequest(BaseModel):
    item_ids: List[str]
    space_id: Optional[str] = None


# ─── Endpoints ───────────────────────────────────────────────────

@collections_router.get("")
async def list_collections():
    """
    列出所有已收藏的项
    """
    async with mysql_connector.session_scope() as session:
        res = await session.execute(select(CollectionItem))
        items = res.scalars().all()
        return {"success": True, "items": [item.model_dump() for item in items]}


@collections_router.post("/toggle")
async def toggle_favorite(req: ToggleFavoriteRequest):
    """
    点亮或取消收藏项（多态匹配）
    """
    async with mysql_connector.session_scope() as session:
        res = await session.execute(
            select(CollectionItem).where(CollectionItem.item_type == req.item_type)
        )
        items = res.scalars().all()
        
        existing = None
        for item in items:
            if not item.data:
                continue
            if req.item_type == "media":
                curr_url = item.data.get("url") or item.data.get("image_url")
                new_url = req.data.get("url") or req.data.get("image_url")
                if curr_url and curr_url == new_url:
                    existing = item
                    break
            elif req.item_type == "template":
                curr_text = item.data.get("template_text") or item.data.get("content")
                new_text = req.data.get("template_text") or req.data.get("content")
                if curr_text and curr_text == new_text:
                    existing = item
                    break
            elif req.item_type == "prompt":
                curr_prompt = item.data.get("prompt")
                new_prompt = req.data.get("prompt")
                if curr_prompt and curr_prompt == new_prompt:
                    existing = item
                    break
        
        if existing:
            # 已存在，执行取消收藏
            await session.delete(existing)
            await session.commit()
            return {"success": True, "favorited": False, "message": "unfavorited"}
        else:
            # 不存在，执行收藏
            new_id = str(uuid.uuid4())
            new_item = CollectionItem(
                id=new_id,
                space_id=req.space_id,
                item_type=req.item_type,
                title=req.title,
                cover_url=req.cover_url,
                data=req.data,
                tags=req.tags or []
            )
            session.add(new_item)
            await session.commit()
            return {
                "success": True, 
                "favorited": True, 
                "message": "favorited", 
                "item": new_item.model_dump()
            }


@collections_router.get("/theme-spaces")
async def list_theme_spaces():
    """
    列出所有自定义的主题空间
    """
    async with mysql_connector.session_scope() as session:
        res = await session.execute(select(ThemeSpace))
        spaces = res.scalars().all()
        return {"success": True, "theme_spaces": [space.model_dump() for space in spaces]}


@collections_router.post("/theme-spaces")
async def create_theme_space(req: ThemeSpaceCreateRequest):
    """
    新建一个主题分类空间
    """
    async with mysql_connector.session_scope() as session:
        new_id = str(uuid.uuid4())
        new_space = ThemeSpace(
            id=new_id,
            name=req.name,
            category=req.category,
            description=req.description,
            space_tags=req.space_tags or []
        )
        session.add(new_space)
        await session.commit()
        return {"success": True, "theme_space": new_space.model_dump()}


@collections_router.put("/theme-spaces/{space_id}")
async def update_theme_space(space_id: str, req: ThemeSpaceUpdateRequest):
    """
    更新主题分类空间的名称、描述、候选标签
    """
    async with mysql_connector.session_scope() as session:
        space = await session.get(ThemeSpace, space_id)
        if not space:
            raise HTTPException(status_code=404, detail="Theme space not found")

        space.name = req.name
        if req.description is not None:
            space.description = req.description
        if req.space_tags is not None:
            space.space_tags = req.space_tags

        session.add(space)
        await session.commit()
        await session.refresh(space)
        return {"success": True, "theme_space": space.model_dump()}


@collections_router.delete("/theme-spaces/{space_id}")
async def delete_theme_space(space_id: str):
    """
    删除主题空间（自动解除其下收藏项的归属关系）
    """
    async with mysql_connector.session_scope() as session:
        space = await session.get(ThemeSpace, space_id)
        if not space:
            raise HTTPException(status_code=404, detail="Theme space not found")
        
        # 解绑归属
        res = await session.execute(
            select(CollectionItem).where(CollectionItem.space_id == space_id)
        )
        items = res.scalars().all()
        for item in items:
            item.space_id = None
            session.add(item)
            
        await session.delete(space)
        await session.commit()
        return {"success": True}


@collections_router.post("/{item_id}/space")
async def move_to_space(item_id: str, req: MoveToSpaceRequest):
    """
    将收藏项分类移入或移出指定主题空间
    """
    async with mysql_connector.session_scope() as session:
        item = await session.get(CollectionItem, item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Collection item not found")
        
        item.space_id = req.space_id
        session.add(item)
        await session.commit()
        return {"success": True, "item": item.model_dump()}


@collections_router.post("/batch/space")
async def batch_move_to_space(req: BatchMoveToSpaceRequest):
    """
    批量将收藏项分类移入或移出指定主题空间
    """
    async with mysql_connector.session_scope() as session:
        for item_id in req.item_ids:
            item = await session.get(CollectionItem, item_id)
            if item:
                item.space_id = req.space_id
                session.add(item)
        await session.commit()
        return {"success": True}
