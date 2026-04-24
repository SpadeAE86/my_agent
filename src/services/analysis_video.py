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
from typing import List, Optional

from utils.video_process_utils import get_video_scenes, get_video_single_scene_frames
from utils.obs_utils import batch_upload_to_obs
from utils.call_model_utils import call_doubao_vision
from models.pydantic.dataclass.scene_split_result import SceneSplitResult
from models.pydantic.model_output_schema.video_analysis_schema import SceneAnalysisResult
from models.pydantic.video_analysis_request import ShotCard
from models.pydantic.opensearch_index.car_interior_analysis import CarInteriorAnalysis
from infra.storage.opensearch.document_writer import bulk_index
from infra.logging.logger import logger as log


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
        schema = SceneAnalysisResult.model_json_schema()
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
    except Exception as e:
        log.error(f"scene {scene.scene_id} 豆包分析失败: {e}")
        card.error = f"分析失败: {e}"

    return card


async def analyze_video(
    local_video_path: str,
    project_id: str,
    frame_interval: float = 2.0,
    threshold: float = 30.0,
    custom_prompt: Optional[str] = None,
    split_scenes: bool = True,
    workspace_dir: Optional[str] = None,
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

    workspace_dir = workspace_dir or f"./video_analysis_workspace/{project_id}"
    obs_key_prefix = f"ai_picture/video_analysis/{project_id}"
    prompt = custom_prompt or DEFAULT_VISION_PROMPT

    # Step 1: 分镜检测 + 抽帧 (CPU 密集, 放到线程池)
    if split_scenes:
        log.info(f"[{project_id}] 开始场景检测与抽帧: {local_video_path}")
        scenes: List[SceneSplitResult] = await asyncio.to_thread(
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
        frame_urls = await _upload_scene_frames(scene, obs_key_prefix)
        log.info(f"[{project_id}] scene {scene.scene_id} 上传完成, {len(frame_urls)} 帧")
        card = await _analyze_single_scene(scene, frame_urls, prompt)
        log.info(f"[{project_id}] scene {scene.scene_id} 豆包分析完成")
        return card

    cards = await asyncio.gather(*[_process_scene(s) for s in scenes])
    # 按 scene_id 排序, 保证前端展示顺序正确
    cards.sort(key=lambda c: c.scene_id)
    log.info(f"[{project_id}] 视频分析全流程完成, 共 {len(cards)} 张卡片")
    
    # Step 4: 入库 OpenSearch（异步，不阻塞接口返回）
    try:
        task = asyncio.create_task(index_shotcards_to_opensearch(cards, id_prefix=project_id, refresh=False))
        def _log_task_result(t: asyncio.Task):
            try:
                r = t.result()
                log.info(f"[{project_id}] OpenSearch 入库完成: {r}")
            except Exception as _e:
                log.error(f"[{project_id}] OpenSearch 入库失败: {_e}")
        task.add_done_callback(_log_task_result)
    except Exception as e:
        log.warning(f"[{project_id}] 创建 OpenSearch 入库任务失败: {e}")

    return list(cards)


async def map_shotcards_to_car_interior_docs(
    cards: List[ShotCard],
    *,
    embedding_model,
    id_prefix: str,
) -> List[CarInteriorAnalysis]:
    """
    Convert ShotCard list into CarInteriorAnalysis documents (with embeddings).
    `id_prefix` is typically project_id / history_id, combined with scene_id.
    """
    docs: List[CarInteriorAnalysis] = []
    for c in cards:
        if c.error:
            continue
        analysis_result = {
            "id": f"{id_prefix}_scene_{c.scene_id:03d}",
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
) -> dict:
    """
    Convenience method:
    - build embeddings
    - bulk index into `car_interior_analysis`
    """
    if embedding_model is None:
        # Lazy import to avoid heavy model load at service import time
        from sentence_transformers import SentenceTransformer

        # Keep consistent with QueryBuilder default
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    docs = await map_shotcards_to_car_interior_docs(cards, embedding_model=embedding_model, id_prefix=id_prefix)
    if not docs:
        return {"success": True, "items": 0}
    resp = await bulk_index(CarInteriorAnalysis, docs, refresh=refresh)
    return {"success": True, "items": len(docs), "opensearch": resp}
