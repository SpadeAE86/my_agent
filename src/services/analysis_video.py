# services/analysis_video.py — 视频分析业务服务
# 职责:
#   1. 接收本地视频路径
#   2. 调用 video_process_utils 提取分镜帧
#   3. 并行上传每个分镜的帧到 OBS
#   4. 并行调用豆包视觉模型分析每个分镜
#   5. 聚合成前端需要的分镜卡片列表

import os
import json
import asyncio
import shutil
from typing import Any, List, Optional

from utils.video_process_utils import get_video_scenes, get_video_single_scene_frames
from utils.obs_utils import batch_upload_to_obs
from utils.call_model_utils import call_doubao_vision
from models.pydantic.dataclass.scene_split_result import SceneSplitResult
from models.pydantic.model_output_schema.video_analysis_schema import SceneAnalysisResult, SceneAnalysisResultV2
from models.pydantic.video_analysis_request import ShotCard
from models.pydantic.opensearch_index.car_interior_analysis import CarInteriorAnalysis
from infra.storage.opensearch.document_writer import bulk_index
from infra.logging.logger import logger as log


from models.sqlmodel.video_upload_cache import VideoSourceUploadCache
from infra.storage.mysql_connector import mysql_connector
from sqlmodel import select
from utils.cache_utils import get_from_cache, set_to_cache
from services.video_analysis_db_service import video_analysis_db_service

async def _get_or_upload_source_video(local_video_path: str, project_id: str) -> Optional[str]:
    """
    Check video_source_upload_cache for the source video.
    If not found, upload to OBS and cache it.
    """
    basename = os.path.basename(local_video_path)
    async with mysql_connector.session_scope() as session:
        res = await session.execute(
            select(VideoSourceUploadCache).where(VideoSourceUploadCache.file_name == basename)
        )
        row = res.scalar_one_or_none()
        if row and row.obs_url:
            log.info(f"[{project_id}] 命中源视频缓存: {row.obs_url}")
            return row.obs_url

    # Not found, upload
    obs_key_prefix = f"ai_picture/car_video_analysis/source_video/{project_id}"
    log.info(f"[{project_id}] 源视频未命中缓存，开始上传: {local_video_path}")
    try:
        from utils.obs_utils import upload_to_obs
        obs_url = await upload_to_obs(local_video_path, obs_prefix=obs_key_prefix)
        
        # Save to cache
        async with mysql_connector.session_scope() as session:
            try:
                new_row = VideoSourceUploadCache(
                    file_name=basename,
                    abs_path=local_video_path,
                    obs_url=obs_url,
                    obs_key=obs_url.split(".com/")[-1] if ".com/" in obs_url else obs_url
                )
                session.add(new_row)
                await session.commit()
            except Exception as e:
                log.warning(f"[{project_id}] 写入 video_source_upload_cache 失败: {e}")
                
        return obs_url
    except Exception as e:
        log.error(f"[{project_id}] 上传源视频失败: {e}")
        return None

import dataclasses

async def _get_cached_scenes(
    basename: str,
    frame_interval: float,
    threshold: float,
    split_scenes: bool
) -> Optional[List[SceneSplitResult]]:
    """
    Try to get cached scenes (with OBS URLs) from local cache_utils or DB.
    """
    cache_key = f"video_analysis_scenes_{basename}_{frame_interval}_{threshold}_{split_scenes}"
    
    # 1. Check local TTL cache
    try:
        cached_data = await asyncio.to_thread(get_from_cache, cache_key)
        if cached_data:
            data = json.loads(cached_data)
            log.info(f"命中本地 TTL 缓存 scenes: {basename}")
            return [SceneSplitResult(**item) for item in data]
    except Exception as e:
        log.warning(f"读取本地 TTL 缓存失败: {e}")

    # 2. Check DB (video_analysis_scene_frames & video_analysis_shot_cards_v2)
    # We only check DB if we can find the video_id and it has scenes
    try:
        video_id, _ = await video_analysis_db_service.resolve_video_v2_for_source_file(file_name=basename)
        if video_id > 0:
            async with mysql_connector.session_scope() as session:
                from models.sqlmodel.video_analysis import VideoAnalysisShotCardV2, VideoAnalysisSceneFrames
                
                # Get scene metadata
                res_cards = await session.execute(
                    select(VideoAnalysisShotCardV2)
                    .where(VideoAnalysisShotCardV2.video_id == video_id)
                    .order_by(VideoAnalysisShotCardV2.scene_id.asc())
                )
                cards = res_cards.scalars().all()
                
                # Get scene frames
                res_frames = await session.execute(
                    select(VideoAnalysisSceneFrames)
                    .where(VideoAnalysisSceneFrames.video_id == video_id)
                )
                frames_map = {r.scene_id: list(r.frame_paths or []) for r in res_frames.scalars().all()}
                
                if cards and frames_map:
                    scenes = []
                    for c in cards:
                        urls = frames_map.get(c.scene_id, [])
                        if not urls:
                            continue
                        scenes.append(SceneSplitResult(
                            scene_id=c.scene_id,
                            start_time=c.start_time,
                            end_time=c.end_time,
                            duration_seconds=c.duration_seconds,
                            frame_url_list=urls
                        ))
                    
                    if scenes:
                        log.info(f"命中 DB 缓存 scenes: {basename}, 共 {len(scenes)} 镜")
                        # Save back to local TTL cache
                        try:
                            scenes_dict = [dataclasses.asdict(s) for s in scenes]
                            await asyncio.to_thread(set_to_cache, cache_key, json.dumps(scenes_dict).encode('utf-8'))
                        except Exception as e:
                            log.warning(f"写入本地 TTL 缓存失败: {e}")
                        return scenes
    except Exception as e:
        log.warning(f"读取 DB 缓存 scenes 失败: {e}")

    return None

async def _cache_scenes_locally(
    basename: str,
    frame_interval: float,
    threshold: float,
    split_scenes: bool,
    scenes: List[SceneSplitResult]
):
    cache_key = f"video_analysis_scenes_{basename}_{frame_interval}_{threshold}_{split_scenes}"
    try:
        scenes_dict = [dataclasses.asdict(s) for s in scenes]
        await asyncio.to_thread(set_to_cache, cache_key, json.dumps(scenes_dict).encode('utf-8'))
    except Exception as e:
        log.warning(f"写入本地 TTL 缓存失败: {e}")

DEFAULT_VISION_PROMPT = """你是一个专业的视频分镜分析师，同时你也了解用户在搜索视频时的习惯。
请分析这些视频片段里的画面。
【重要规则】
1. 提取 object 时，请使用最通用的词汇，贴合日常口语表达。
2. search_tags 字段极其重要，请发挥联想，写出用户搜什么词时应该看到这个视频。

### 1. 营销场景标签
- **场景类型**：判断属于哪种营销场景
  可选：产品展示、使用场景、情感共鸣、品牌故事、教程演示、对比评测、生活方式展示

- **目标受众**：这个画面最能打动哪类人群？
  示例：Z世代、精致妈妈、职场精英、银发族、健身达人、美食爱好者

### 2. 商业价值评估 (0-10分)
- 产品展示清晰度：画面是否适合展示产品细节
- 情感共鸣度：是否能引起观众情感共鸣
- 画面美感度：构图、光线、色彩的专业程度
- 通用适配性：是否容易与其他素材混剪
"""

_EMBEDDING_MODEL: Any = None

def get_embedding_model():
    """
    Singleton SentenceTransformer model loader.
    Loads once per process; subsequent calls reuse the same instance to avoid
    repeated "Loading weights" overhead on every reindex/search.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    from sentence_transformers import SentenceTransformer
    _EMBEDDING_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _EMBEDDING_MODEL


async def _upload_scene_frames(scene: SceneSplitResult, obs_key_prefix: str) -> List[str]:
    """将单个分镜的本地帧上传到 OBS, 返回公网 URL 列表"""
    if not scene.frame_url_list:
        return []
    try:
        return await batch_upload_to_obs(
            file_paths=scene.frame_url_list,
            obs_key_prefix=obs_key_prefix,
            max_concurrency=5,
        )
    except Exception as e:
        log.error(f"scene {scene.scene_id} 上传 OBS 失败: {e}")
        return []


async def _analyze_single_scene(
    scene: SceneSplitResult,
    frame_urls: List[str],
    prompt: str,
    workspace: str = "v1",
    car_model: Optional[str] = None,
    resolution: Optional[str] = None,
    frame_size: Optional[str] = None,
    video_duration: Optional[float] = None,
    project_id: str = "",
) -> ShotCard:
    """对单个分镜调用豆包视觉模型, 组装成 ShotCard"""
    card = ShotCard(
        scene_id=scene.scene_id,
        start_time=scene.start_time,
        end_time=scene.end_time,
        duration_seconds=scene.duration_seconds,
        thumbnail=frame_urls[0] if frame_urls else None,
        frame_urls=frame_urls,
    )

    if not frame_urls:
        card.error = "无可用帧, 跳过豆包分析"
        return card

    try:
        schema = SceneAnalysisResultV2.model_json_schema() if workspace == "v2" else SceneAnalysisResult.model_json_schema()
        raw = await call_doubao_vision(prompt, frame_urls, schema)
        if not raw:
            card.error = "豆包返回为空"
            return card

        # 豆包以 JSON 字符串返回
        data = raw if isinstance(raw, dict) else json.loads(raw)
        card.description = data.get("description")
        card.subject = data.get("subject")
        card.object = data.get("object")
        card.movement = data.get("movement")
        card.adjective = data.get("adjective")
        card.search_tags = data.get("search_tags")
        card.marketing_tags = data.get("marketing_tags")
        card.appealing_audience = data.get("appealing_audience")
        card.visual_quality = data.get("visual_quality")
        
        # v2 fields
        card.car_model = car_model if car_model else data.get("car_model")
        card.frame_size = frame_size if frame_size else data.get("frame_size")
        card.resolution = resolution if resolution else data.get("resolution")
        card.video_duration = video_duration if video_duration else data.get("video_duration")
        card.analysis_doc_id = f"{project_id}_{scene.scene_id}"
        
        card.footage_type = data.get("footage_type")
        card.shot_style = data.get("shot_style")
        card.shot_type = data.get("shot_type")
        card.camera_movement = data.get("camera_movement")
        card.scene_location = data.get("scene_location")
        card.car_color = data.get("car_color")
        card.product_status_scene = data.get("product_status_scene")
        card.has_presenter = data.get("has_presenter")
        card.generic_hq_road_run = data.get("generic_hq_road_run", False)
        card.person_detail = data.get("person_detail")
        card.key_words = data.get("key_words")
        card.text = data.get("text")
        card.video_usage = data.get("video_usage")
        card.design_adjectives = data.get("design_adjectives")
        card.function_adjectives = data.get("function_adjectives")
        card.design_selling_points = data.get("design_selling_points")
        card.function_selling_points = data.get("function_selling_points")
        card.scenario_a = data.get("scenario_a")
        card.scenario_b = data.get("scenario_b")
        card.marketing_phrases = data.get("marketing_phrases")
        card.topic = data.get("topic")
        card.weather = data.get("weather")
        card.time = data.get("time")

    except Exception as e:
        log.error(f"scene {scene.scene_id} 豆包分析失败: {e}")
        card.error = f"分析失败: {e}"

    return card


import cv2
import math

async def analyze_video(
    local_video_path: str,
    project_id: str,
    frame_interval: float = 2.0,
    threshold: float = 30.0,
    custom_prompt: Optional[str] = None,
    split_scenes: bool = True,
    cleanup_workspace: bool = True,
    workspace_dir: Optional[str] = None,
    workspace: str = "v1",
    car_model: Optional[str] = None,
) -> List[ShotCard]:
    """完整的视频分析流水线

    Args:
        local_video_path: 本地视频文件路径
        project_id: 本次分析的唯一 ID, 用作 OBS 的二级目录和工作目录
        frame_interval: 抽帧间隔秒数
        threshold: 场景切换灵敏度
        custom_prompt: 自定义分析提示词, 不传则使用默认
        workspace_dir: 本地帧保存目录

    Returns:
        分镜卡片列表
    """
    if not os.path.exists(local_video_path):
        raise FileNotFoundError(f"视频文件不存在: {local_video_path}")

    # 获取视频媒体信息
    try:
        cap = cv2.VideoCapture(local_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        resolution = f"{width}x{height}" if width and height else "未知"
        video_duration = frame_count / fps if fps > 0 else 0.0
        
        def get_aspect_ratio(w, h):
            if w == 0 or h == 0: return "未知"
            gcd = math.gcd(w, h)
            rw, rh = w//gcd, h//gcd
            if rw == 8 and rh == 9: return "16:9" # 近似 1920x1080 等
            return f"{rw}:{rh}"
            
        frame_size = get_aspect_ratio(width, height)
        if width == 1920 and height == 1080: frame_size = "16:9"
        elif width == 1080 and height == 1920: frame_size = "9:16"
    except Exception as e:
        log.warning(f"[{project_id}] 获取视频媒体信息失败: {e}")
        resolution = "未知"
        video_duration = 0.0
        frame_size = "未知"

    workspace_dir = workspace_dir or f"./video_analysis_workspace/{project_id}"
    obs_key_prefix = f"ai_picture/video_analysis/{project_id}"
    prompt = custom_prompt or DEFAULT_VISION_PROMPT

    # Step 1: 分镜检测 + 抽帧 (CPU 密集, 放到线程池)
    basename = os.path.basename(local_video_path)
    
    # 尝试从缓存中获取（跳过 CPU 抽帧和上传 OBS）
    cached_scenes = await _get_cached_scenes(
        basename=basename,
        frame_interval=frame_interval,
        threshold=threshold,
        split_scenes=split_scenes
    )
    
    if cached_scenes:
        log.info(f"[{project_id}] 命中场景缓存，跳过本地抽帧和上传")
        scenes = cached_scenes
    else:
        if split_scenes:
            log.info(f"[{project_id}] 开始场景检测与抽帧: {local_video_path}")
            scenes = await asyncio.to_thread(
                get_video_scenes,
                local_video_path,
                frame_interval,
                threshold,
                workspace_dir,
            )
            log.info(f"[{project_id}] 场景检测完成, 共 {len(scenes)} 个分镜")
        else:
            log.info(f"[{project_id}] 跳过分镜切分，整段抽帧: {local_video_path}")
            scenes = await asyncio.to_thread(
                get_video_single_scene_frames,
                local_video_path,
                frame_interval,
                workspace_dir,
            )
            log.info(f"[{project_id}] 整段抽帧完成, 共 {len(scenes[0].frame_url_list) if scenes else 0} 帧")

    if not scenes:
        return []

    # Step 2 + 3: 每个分镜并行执行 "上传 OBS -> 调用豆包" 的子流水线
    async def _process_scene(scene: SceneSplitResult) -> ShotCard:
        # 如果 frame_url_list 已经是 OBS URL（命中缓存），则跳过上传
        if scene.frame_url_list and scene.frame_url_list[0].startswith("http"):
            frame_urls = scene.frame_url_list
            log.info(f"[{project_id}] scene {scene.scene_id} 命中 OBS 缓存 URL, {len(frame_urls)} 帧")
        else:
            frame_urls = await _upload_scene_frames(scene, obs_key_prefix)
            log.info(f"[{project_id}] scene {scene.scene_id} 上传完成, {len(frame_urls)} 帧")
            # 更新 scene 中的 url 为 OBS url，以便后续存入本地 TTL 缓存
            scene.frame_url_list = frame_urls
            
        card = await _analyze_single_scene(
            scene, frame_urls, prompt, workspace,
            car_model=car_model, resolution=resolution, frame_size=frame_size, video_duration=video_duration, project_id=project_id
        )
        log.info(f"[{project_id}] scene {scene.scene_id} 豆包分析完成")
        return card

    try:
        cards = await asyncio.gather(*[_process_scene(s) for s in scenes])
        # 按 scene_id 排序, 保证前端展示顺序正确
        cards.sort(key=lambda c: c.scene_id)
        log.info(f"[{project_id}] 视频分析全流程完成, 共 {len(cards)} 张卡片")
        
        # 如果是新抽帧的，将其存入本地 TTL 缓存
        if not cached_scenes:
            await _cache_scenes_locally(
                basename=basename,
                frame_interval=frame_interval,
                threshold=threshold,
                split_scenes=split_scenes,
                scenes=scenes
            )
            
    finally:
        # Cleanup extracted frames workspace to avoid repo bloat.
        if cleanup_workspace:
            try:
                if workspace_dir and os.path.exists(workspace_dir):
                    shutil.rmtree(workspace_dir, ignore_errors=True)
                    log.info(f"[{project_id}] 已清理抽帧目录: {workspace_dir}")
            except Exception as e:
                log.warning(f"[{project_id}] 清理抽帧目录失败: {e}")
    
    return list(cards)


async def map_shotcards_to_car_interior_docs(
    cards: List[ShotCard],
    *,
    embedding_model,
    id_prefix: str,
    workspace: str = "v1",
) -> List[Any]:
    """
    Convert ShotCard list into CarInteriorAnalysis documents (with embeddings).
    `id_prefix` is typically project_id / history_id, combined with scene_id.
    """
    docs = []
    for c in cards:
        if c.error:
            continue
        
        if workspace == "v2":
            from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
            analysis_result = {
                "id": f"{id_prefix}_{c.scene_id}",
                "description": c.description or "",
                "subject": c.subject or "",
                "object": c.object or [],
                "movement": c.movement or "",
                "car_model": c.car_model or "未知",
                "frame_size": c.frame_size or "未知",
                "resolution": c.resolution or "未知",
                "video_duration": c.video_duration or 0.0,
                "start_time": c.start_time or 0.0,
                "end_time": c.end_time or 0.0,
                "footage_type": c.footage_type or "未知",
                "shot_style": c.shot_style or "未知",
                "shot_type": c.shot_type or "未知",
                "camera_movement": c.camera_movement or "未知",
                "scene_location": c.scene_location or [],
                "car_color": c.car_color or "未知",
                "product_status_scene": c.product_status_scene or "未知",
                "has_presenter": c.has_presenter,
                "generic_hq_road_run": c.generic_hq_road_run or False,
                "person_detail": c.person_detail or [],
                "key_words": c.key_words or [],
                "topic": c.topic or "未知",
                "text": c.text or [],
                "weather": c.weather or "未知",
                "time": c.time or "未知",
                "video_usage": c.video_usage or [],
                "design_adjectives": c.design_adjectives or [],
                "function_adjectives": c.function_adjectives or [],
                "design_selling_points": c.design_selling_points or [],
                "function_selling_points": c.function_selling_points or [],
                "scenario_a": c.scenario_a or [],
                "scenario_b": c.scenario_b or [],
                "marketing_phrases": c.marketing_phrases or [],
                "appealing_audience": c.appealing_audience or [],
            }
            docs.append(CarInteriorAnalysisV2.from_analysis_result(analysis_result, embedding_model))
        else:
            analysis_result = {
                "id": f"{id_prefix}_{c.scene_id}",
                "description": c.description or "",
                "subject": c.subject or "",
                "object": c.object or [],
                "movement": c.movement or "",
                "adjective": c.adjective or [],
                "search_tags": c.search_tags or [],
                "marketing_tags": c.marketing_tags or [],
                "appealing_audience": c.appealing_audience or [],
                "visual_quality": c.visual_quality or [0, 0, 0, 0],
            }
            docs.append(CarInteriorAnalysis.from_analysis_result(analysis_result, embedding_model))
    return docs


async def index_shotcards_to_opensearch(
    cards: List[ShotCard],
    *,
    id_prefix: str,
    embedding_model=None,
    refresh: bool = False,
    workspace: str = "v1",
) -> dict:
    """
    Convenience method:
    - build embeddings
    - bulk index into `car_interior_analysis`
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    docs = await map_shotcards_to_car_interior_docs(cards, embedding_model=embedding_model, id_prefix=id_prefix, workspace=workspace)
    if not docs:
        return {"success": True, "items": 0}
    
    if workspace == "v2":
        from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
        resp = await bulk_index(CarInteriorAnalysisV2, docs, refresh=refresh)
    else:
        resp = await bulk_index(CarInteriorAnalysis, docs, refresh=refresh)
        
    return {"success": True, "items": len(docs), "opensearch": resp}
