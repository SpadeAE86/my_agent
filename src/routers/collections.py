import uuid
import json
import os
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlmodel import select

from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.collections import ThemeSpace, CollectionItem
from infra.logging.logger import logger as log

collections_router = APIRouter(prefix="/collections", tags=["collections"])

# 内存中后台自动打标任务状态字典
auto_tag_tasks: Dict[str, Dict[str, Any]] = {}


# ─── Qwen3-VL-Embedding-2B & Tag Library Integration Helpers ──────
_QWEN_VL_MODEL = None
_QWEN_VL_MODEL_LOCK = asyncio.Lock()

def get_qwen_vl_model_path() -> str:
    base_dir = r"C:\Users\admin\.cache\huggingface\hub\models--Qwen--Qwen3-VL-Embedding-2B\snapshots"
    if os.path.exists(base_dir):
        subdirs = os.listdir(base_dir)
        if subdirs:
            for subdir in subdirs:
                full_path = os.path.join(base_dir, subdir)
                if os.path.isdir(full_path):
                    return full_path
    return r"C:\Users\admin\.cache\huggingface\hub\models--Qwen--Qwen3-VL-Embedding-2B\snapshots\9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"

async def get_qwen_vl_model():
    global _QWEN_VL_MODEL
    if _QWEN_VL_MODEL is not None:
        return _QWEN_VL_MODEL
    async with _QWEN_VL_MODEL_LOCK:
        if _QWEN_VL_MODEL is not None:
            return _QWEN_VL_MODEL
        from sentence_transformers import SentenceTransformer
        model_path = get_qwen_vl_model_path()
        log.info(f"Loading Qwen3-VL-Embedding-2B from {model_path}...")
        _QWEN_VL_MODEL = await asyncio.to_thread(SentenceTransformer, model_path)
        log.info("Qwen3-VL-Embedding-2B loaded successfully.")
        return _QWEN_VL_MODEL

async def _add_tags_to_library(tags: List[str]):
    if not tags:
        return
    try:
        from core.auto_cluster import AutoClusterOrchestrator
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        state_file = os.path.join(base_dir, "test", "auto_cluster_state.pkl")
        
        orchestrator = AutoClusterOrchestrator()
        loaded = orchestrator.load_state(state_file)
        if not loaded:
            state_file_alt = os.path.join(base_dir, "core", "auto_cluster", "auto_cluster_state.pkl")
            loaded = orchestrator.load_state(state_file_alt)
            if loaded:
                state_file = state_file_alt
                
        updated_orchestrator = False
        for t in tags:
            t = t.strip()
            if not t:
                continue
            exists = False
            if loaded and orchestrator.history:
                for path, items in orchestrator.history.items():
                    if any(item["tag"] == t for item in items):
                        exists = True
                        break
            if not exists:
                await orchestrator.process_tag(t)
                updated_orchestrator = True
                
        if updated_orchestrator:
            await orchestrator.discover_soft_links(similarity_threshold=0.45)
            orchestrator.save_state(state_file)
            log.info(f"Successfully processed and saved tags {tags} to AutoClusterOrchestrator")
    except Exception as e:
        log.error(f"Failed to feed tags into orchestrator: {e}")


async def sync_existing_tags_to_library():
    """
    将 MySQL 数据库中所有 CollectionItem 的 tags 同步回自组织标签库 (AutoClusterOrchestrator)
    """
    try:
        from sqlmodel import select
        from infra.storage.mysql_connector import mysql_connector
        from models.sqlmodel.collections import CollectionItem
        from core.auto_cluster import AutoClusterOrchestrator
        
        async with mysql_connector.session_scope() as session:
            res = await session.execute(select(CollectionItem))
            items = res.scalars().all()
            db_tags = set()
            for item in items:
                if item.tags:
                    for t in item.tags:
                        db_tags.add(t.strip())
                        
        if not db_tags:
            return
            
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        state_file = os.path.join(base_dir, "test", "auto_cluster_state.pkl")
        
        orchestrator = AutoClusterOrchestrator()
        loaded = orchestrator.load_state(state_file)
        if not loaded:
            state_file_alt = os.path.join(base_dir, "core", "auto_cluster", "auto_cluster_state.pkl")
            loaded = orchestrator.load_state(state_file_alt)
            if loaded:
                state_file = state_file_alt
                
        # 找出所有已经在标签库历史里的 tags
        lib_tags = set()
        if loaded and orchestrator.history:
            for path, items in orchestrator.history.items():
                for item in items:
                    lib_tags.add(item["tag"])
                    
        missing_tags = db_tags - lib_tags
        if missing_tags:
            log.info(f"Detecting {len(missing_tags)} tags in DB but missing in library, backfilling: {missing_tags}")
            for t in missing_tags:
                await orchestrator.process_tag(t)
            await orchestrator.discover_soft_links(similarity_threshold=0.45)
            orchestrator.save_state(state_file)
            log.info("Tags backfill synchronization completed.")
    except Exception as e:
        log.error(f"Failed to synchronize DB tags to library: {e}")


async def sync_library_to_db():
    """
    将本地自组织标签库 (AutoClusterOrchestrator pkl) 状态同步镜像到 MySQL `tag_library` 表
    """
    try:
        from sqlmodel import select
        from infra.storage.mysql_connector import mysql_connector
        from models.sqlmodel.collections import TagLibraryItem
        from core.auto_cluster import AutoClusterOrchestrator
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        state_file = os.path.join(base_dir, "test", "auto_cluster_state.pkl")
        
        orchestrator = AutoClusterOrchestrator()
        loaded = orchestrator.load_state(state_file)
        if not loaded:
            state_file_alt = os.path.join(base_dir, "core", "auto_cluster", "auto_cluster_state.pkl")
            loaded = orchestrator.load_state(state_file_alt)
            if not loaded:
                log.warning("Tag library pkl not found, skip sync to DB")
                return
                
        # 收集当前最新的所有 (tag, category_path) 项
        current_items = {}  # (tag, category_path) -> {cluster_id, concept_name, is_soft_link}
        
        # 1. 物理标签
        for path, items in orchestrator.history.items():
            for item in items:
                tag = item["tag"].strip()
                cluster_id = item.get("cluster_id", -1)
                concept_name = orchestrator.stable_concept_names.get(path, {}).get(cluster_id, "临时/萌芽概念")
                current_items[(tag, path)] = {
                    "cluster_id": cluster_id,
                    "concept_name": concept_name,
                    "is_soft_link": False
                }
                
        # 2. 逻辑软链接
        for path, soft_tags in orchestrator.logical_soft_links.items():
            for tag in soft_tags:
                tag = tag.strip()
                # 如果这个物理标签已经在当前类目下，则不作为软链接重复添加
                if (tag, path) in current_items:
                    continue
                current_items[(tag, path)] = {
                    "cluster_id": -1,
                    "concept_name": "逻辑软链接",
                    "is_soft_link": True
                }
                
        # 3. 同步到数据库
        async with mysql_connector.session_scope() as session:
            # 获取数据库已有的所有数据
            res = await session.execute(select(TagLibraryItem))
            db_items = res.scalars().all()
            db_map = {(item.tag, item.category_path): item for item in db_items}
            
            # 更新/增加
            for (tag, path), info in current_items.items():
                if (tag, path) in db_map:
                    db_item = db_map[(tag, path)]
                    db_item.cluster_id = info["cluster_id"]
                    db_item.concept_name = info["concept_name"]
                    db_item.is_soft_link = info["is_soft_link"]
                    session.add(db_item)
                else:
                    new_item = TagLibraryItem(
                        tag=tag,
                        category_path=path,
                        cluster_id=info["cluster_id"],
                        concept_name=info["concept_name"],
                        is_soft_link=info["is_soft_link"]
                    )
                    session.add(new_item)
            
            # 删除多余的
            current_keys = set(current_items.keys())
            for key, db_item in db_map.items():
                if key not in current_keys:
                    await session.delete(db_item)
                    
            await session.commit()
            
            # 更新同步时间标记，供 compare_tags 等脚本读取
            sync_time_file = os.path.join(base_dir, "test", "tag_library_sync_time.json")
            import time
            with open(sync_time_file, "w") as f:
                json.dump({"last_sync_timestamp": time.time()}, f)
                
            log.info(f"Successfully synchronized {len(current_items)} tags from pkl to tag_library table.")
    except Exception as e:
        log.error(f"Failed to synchronize tag library pkl to DB: {e}")


async def start_tag_library_db_sync_task():
    log.info("Starting background task: sync_library_to_db loop (every 15 minutes)")
    # 第一次启动时立即运行同步
    try:
        await sync_library_to_db()
    except Exception as e:
        log.error(f"Error in initial sync_library_to_db: {e}")
        
    while True:
        await asyncio.sleep(15 * 60)
        try:
            await sync_library_to_db()
        except Exception as e:
            log.error(f"Error in sync_library_to_db background task: {e}")



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


class UpdateItemTagsRequest(BaseModel):
    tags: List[str]


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
            if req.tags:
                await _add_tags_to_library(req.tags)
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


@collections_router.put("/{item_id}/tags")
async def update_item_tags(item_id: str, req: UpdateItemTagsRequest):
    """
    更新收藏项的自定义标签
    """
    async with mysql_connector.session_scope() as session:
        item = await session.get(CollectionItem, item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Collection item not found")
        item.tags = req.tags
        session.add(item)
        await session.commit()
        await session.refresh(item)
        if req.tags:
            await _add_tags_to_library(req.tags)
        return {"success": True, "item": item.model_dump()}


async def _bg_auto_tag_item(item_id: str, use_recall: bool = True):
    try:
        auto_tag_tasks[item_id] = {"status": "RUNNING"}
        async with mysql_connector.session_scope() as session:
            item = await session.get(CollectionItem, item_id)
            if not item:
                auto_tag_tasks[item_id] = {"status": "FAILED", "error": "Collection item not found"}
                return
            
            image_url = item.cover_url or item.data.get("url") or item.data.get("image_url")
            if not image_url:
                auto_tag_tasks[item_id] = {"status": "FAILED", "error": "No preview image URL found for this item"}
                return
            
            guideline_info = None
            if use_recall:
                try:
                    from core.auto_cluster import AutoClusterOrchestrator
                    from core.auto_cluster.router import CategoryRouter
                    from sentence_transformers import util
                    from PIL import Image
                    import requests
                    import io
                    
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    state_file = os.path.join(base_dir, "test", "auto_cluster_state.pkl")
                    
                    orchestrator = AutoClusterOrchestrator()
                    loaded = orchestrator.load_state(state_file)
                    if not loaded:
                        state_file_alt = os.path.join(base_dir, "core", "auto_cluster", "auto_cluster_state.pkl")
                        loaded = orchestrator.load_state(state_file_alt)
                    
                    candidate_tags = set()
                    if loaded and orchestrator.history:
                        for path, items in orchestrator.history.items():
                            for it in items:
                                candidate_tags.add(it["tag"])
                                
                    mock_tags_file = os.path.join(base_dir, "test", "mock_tags.json")
                    if os.path.exists(mock_tags_file):
                        try:
                            with open(mock_tags_file, "r", encoding="utf-8") as f:
                                mock_list = json.load(f)
                                for t in mock_list:
                                    candidate_tags.add(t)
                        except Exception as em:
                            log.warning(f"Failed to load mock_tags.json: {em}")
                            
                    candidate_tags_list = sorted(list(candidate_tags))
                    
                    if candidate_tags_list:
                        img_pil = None
                        if image_url.startswith(("http://", "https://")):
                            resp = await asyncio.to_thread(requests.get, image_url, timeout=10)
                            resp.raise_for_status()
                            img_pil = Image.open(io.BytesIO(resp.content))
                        else:
                            local_path = os.path.abspath(image_url)
                            if os.path.exists(local_path):
                                img_pil = Image.open(local_path)
                            else:
                                alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), image_url)
                                if os.path.exists(alt_path):
                                    img_pil = Image.open(alt_path)
                                    
                        if img_pil:
                            model = await get_qwen_vl_model()
                            img_emb = await asyncio.to_thread(model.encode, img_pil, convert_to_tensor=True)
                            tag_embs = await asyncio.to_thread(model.encode, candidate_tags_list, convert_to_tensor=True)
                            
                            sim_scores = util.cos_sim(img_emb.float(), tag_embs.float())[0].cpu().numpy()
                            
                            router = CategoryRouter()
                            recalled_by_cat = {}
                            for idx, tag in enumerate(candidate_tags_list):
                                score = float(sim_scores[idx])
                                category = router.route(tag)
                                if category not in recalled_by_cat:
                                    recalled_by_cat[category] = []
                                recalled_by_cat[category].append((tag, score))
                                
                            for cat in recalled_by_cat:
                                recalled_by_cat[cat].sort(key=lambda x: x[1], reverse=True)
                                
                            sorted_categories = sorted(
                                recalled_by_cat.items(),
                                key=lambda x: x[1][0][1] if x[1] else -1.0,
                                reverse=True
                            )
                            
                            guideline_lines = []
                            for cat_path, tag_scores in sorted_categories:
                                top_tags = [f"'{t}' (相似度: {s:.3f})" for t, s in tag_scores[:3]]
                                line = f"- 类目 {cat_path}: 推荐样例词包括: {', '.join(top_tags)}"
                                guideline_lines.append(line)
                            
                            guideline_info = "\n".join(guideline_lines)
                            log.info(f"Visual recall generated guideline_info:\n{guideline_info}")
                except Exception as er:
                    log.error(f"Failed during visual recall phase: {er}. Proceeding without recall guidelines.")
            
            from utils.call_model_utils import call_doubao_vision
            
            prompt = (
                "你是一个图像分析专家。请仔细观察这张图片，提取 5 个最能精准概括该图片特征和画面内容的主题标签。"
                "这些标签可以包括主体、材质、色彩、情感、场景或风格，字数控制在 2-6 字之间。"
                "请严格以 JSON 数组格式返回这 5 个标签，例如: [\"标签1\", \"标签2\", \"标签3\", \"标签4\", \"标签5\"]。"
                "不要包含任何其他的解释、Markdown 代码块或前导后导文字，仅输出 JSON 数组。"
            )
            
            res_text = await call_doubao_vision(prompt, [image_url], guideline_info=guideline_info)
            if not res_text:
                raise ValueError("AI Vision model returned empty response")
            
            res_text = res_text.strip()
            if res_text.startswith("```"):
                lines = res_text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                res_text = "\n".join(lines).strip()
            
            new_tags = json.loads(res_text)
            if not isinstance(new_tags, list):
                raise ValueError("Parsed result is not a list")
                
            new_tags = [str(t).strip() for t in new_tags if t][:5]
            
            current_tags = list(item.tags or [])
            updated = False
            new_added_tags = []
            for t in new_tags:
                if t not in current_tags:
                    current_tags.append(t)
                    new_added_tags.append(t)
                    updated = True
            
            if updated:
                item.tags = current_tags
                session.add(item)
                await session.commit()
                await session.refresh(item)
                await _add_tags_to_library(new_added_tags)
                
            auto_tag_tasks[item_id] = {"status": "SUCCESS", "tags": item.tags}
    except Exception as e:
        log.error(f"Auto-tagging item {item_id} failed: {e}")
        auto_tag_tasks[item_id] = {"status": "FAILED", "error": str(e)}


@collections_router.post("/{item_id}/auto-tag")
async def auto_tag_item(
    item_id: str,
    background_tasks: BackgroundTasks,
    use_recall: bool = True
):
    """
    使用豆包 Vision 多模态模型自动为收藏项打标 (异步后台任务模式)
    """
    async with mysql_connector.session_scope() as session:
        item = await session.get(CollectionItem, item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Collection item not found")
        
        image_url = item.cover_url or item.data.get("url") or item.data.get("image_url")
        if not image_url:
            raise HTTPException(status_code=400, detail="No preview image URL found for this item")
            
    auto_tag_tasks[item_id] = {"status": "PENDING"}
    background_tasks.add_task(_bg_auto_tag_item, item_id, use_recall)
    return {"success": True, "task_id": item_id, "status": "PENDING"}


@collections_router.get("/{item_id}/auto-tag/status")
async def get_auto_tag_status(item_id: str):
    """
    获取自动打标后台任务的最新状态
    """
    task = auto_tag_tasks.get(item_id)
    if not task:
        async with mysql_connector.session_scope() as session:
            item = await session.get(CollectionItem, item_id)
            if not item:
                raise HTTPException(status_code=404, detail="Collection item not found")
            return {"success": True, "status": "IDLE", "tags": item.tags or []}
    return {"success": True, **task}


@collections_router.get("/tag-tree")
async def get_tag_tree():
    """
    返回分类层级本体树，并包含各个类目下的所有标签（含逻辑软链接标签）。
    """
    # 每次请求 tag-tree 时在后台进行增量同步校验，确保库和图片打标完全同步
    asyncio.create_task(sync_existing_tags_to_library())
    
    import os
    from core.auto_cluster import AutoClusterOrchestrator
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    state_file = os.path.join(base_dir, "test", "auto_cluster_state.pkl")
    
    orchestrator = AutoClusterOrchestrator()
    loaded = orchestrator.load_state(state_file)
    if not loaded:
        state_file_alt = os.path.join(base_dir, "core", "auto_cluster", "auto_cluster_state.pkl")
        loaded = orchestrator.load_state(state_file_alt)
        
    if not loaded:
        # 兜底默认分类及标签，防止系统冷启动无数据
        result_tree = {
            "Character": {
                "path": "Character",
                "tags": ["亚斯娜", "闪光亚斯娜", "桐人", "结城明日奈"],
                "is_leaf": True,
                "children": None
            },
            "Hair": {
                "path": "Hair",
                "tags": ["双马尾", "单马尾", "金发", "黑发", "银发"],
                "is_leaf": True,
                "children": None
            },
            "Clothing": {
                "path": "Clothing",
                "tags": ["百褶裙", "过膝袜", "裤袜", "水手服"],
                "is_leaf": True,
                "children": None
            },
            "Scenery": {
                "path": "Scenery",
                "tags": ["蓝天白云", "落日余晖", "樱花树下"],
                "is_leaf": True,
                "children": None
            },
            "General": {
                "path": "General",
                "tags": ["太刀", "阐释者", "逐暗者", "誓约胜利之剑"],
                "is_leaf": True,
                "children": None
            }
        }
        return {"success": True, "tag_tree": result_tree}
        
    def build_tree_node(tree_dict: dict, current_prefix: str = "") -> dict:
        node_tree = {}
        for key, subtree in tree_dict.items():
            path = f"{current_prefix}/{key}" if current_prefix else key
            
            physical_tags = [item["tag"] for item in orchestrator.history.get(path, [])]
            soft_tags = orchestrator.logical_soft_links.get(path, [])
            all_tags = sorted(list(set(physical_tags + soft_tags)))
            
            if not subtree:
                node_tree[key] = {
                    "path": path,
                    "tags": all_tags,
                    "is_leaf": True,
                    "children": None
                }
            else:
                node_tree[key] = {
                    "path": path,
                    "tags": all_tags,
                    "is_leaf": False,
                    "children": build_tree_node(subtree, path)
                }
        return node_tree

    full_taxonomy = dict(orchestrator.taxonomy)
    if "General" not in full_taxonomy:
        full_taxonomy["General"] = {}
        
    result_tree = build_tree_node(full_taxonomy)
    return {"success": True, "tag_tree": result_tree}

