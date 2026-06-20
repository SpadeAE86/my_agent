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
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from models.pydantic.opensearch_index import index_v2_enums
from utils.video_process_utils import get_video_scenes, get_video_single_scene_frames
from utils.obs_utils import batch_upload_to_obs
from utils.call_model_utils import call_doubao_vision
from models.pydantic.dataclass.scene_split_result import SceneSplitResult
from models.pydantic.video_analysis_request import ShotCard
from models.pydantic.opensearch_index.car_interior_analysis import CarInteriorAnalysis
from infra.storage.opensearch.document_writer import bulk_index
from infra.logging.logger import logger as log
from core.workspace import get_workspace


from models.sqlmodel.video_upload_cache import VideoSourceUploadCache
from infra.storage.mysql_connector import mysql_connector
from sqlmodel import select
from utils.cache_utils import get_from_cache, set_to_cache
from services.video_match_services.video_analysis_db_service import video_analysis_db_service
from services.video_match_services.zhiji_product_context import (
    build_v2_vision_selling_appendix,
    normalize_zhiji_car_key,
)


def _car_model_slug_for_obs(car_model: Optional[str]) -> str:
    """
    OBS 路径分段用「车型/产品」目录名；空则用 unknown，非法字符压缩为下划线。
    """
    raw = (car_model or "").strip()
    if not raw:
        return "unknown"
    slug = []
    for ch in raw[:80]:
        if ch.isalnum() or ch in ("-", "_", "."):
            slug.append(ch)
        elif ch.isspace():
            slug.append("_")
        else:
            slug.append("_")
    s = "".join(slug).strip("._-")
    while "__" in s:
        s = s.replace("__", "_")
    return s[:64] if s else "unknown"


async def _get_or_upload_source_video(
    local_video_path: str,
    project_id: str,
    car_model: Optional[str] = None,
) -> Optional[str]:
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

    # 仅按车型目录 + 文件名，路径可预判，且与 video_source_upload_cache（按 basename）一致；分析任务仍用 project_id 区分
    slug = _car_model_slug_for_obs(car_model)
    obs_key_prefix = f"ai_picture/car_video_analysis/source_video/{slug}"
    log.info(f"[{project_id}] 源视频未命中缓存，开始上传 (car_model_slug={slug}): {local_video_path}")
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

def _join_choices(xs: List[str]) -> str:
    return ", ".join([x for x in (xs or []) if x])

DEFAULT_VISION_PROMPT_V2 = f"""
你是一个专业的智己汽车视频素材分析师，擅长把“可检索的结构化标签”从画面中抽取出来，支持后续营销脚本混剪检索。

请仅根据这些首帧+2秒间隔的关键帧画面+尾帧的可见信息输出 JSON（必须符合给定 schema），不要输出解释。

关键要求：
- movement：只写“核心动作”（单值），必须标准化，不带环境词、不带评价。例：掉头/转弯/泊车/充电/静态展示
- camera_movement：运镜（单值，固定枚举）：{_join_choices(index_v2_enums.CAMERA_MOVEMENT_CHOICES)}。与 shot_style（车内POV/跟拍等拍摄方式）区分；无明确推拉摇移跟随环绕则填 未知
- generic_hq_road_run：boolean。仅当画面为高质量展示路跑外观的镜头（稳定、清晰、可作无主题兜底）时为 true；否则 false
- footage_type：画面类型（固定枚举）：{_join_choices(index_v2_enums.FOOTAGE_TYPE_CHOICES)}
- shot_style：镜头风格/拍摄方式（固定枚举）：{_join_choices(index_v2_enums.SHOT_STYLE_CHOICES)}
- shot_type：镜头景幅/景别（固定枚举）：{_join_choices(index_v2_enums.SHOT_TYPE_CHOICES)}
- scene_location：画面场景/路况/空间类型（1-6 个），短词名词化，如：地库/公路/冰雪/现代城区/赛道/展厅 等
- car_color：车色（枚举）
- product_status_scene：产品状态场景（标准化），如：静态内饰/路跑外观/发布会现场 等
- has_presenter：是否包含出镜讲解员/达人/主持人（boolean）
- person_detail：人物细分标签（枚举，可多值）：无人物/老人/小孩/男性/女性/多人。无人物时只填 无人物；多人时可同时填多个（如 男性+女性、小孩+女性）。
- weather/time：天气/时间（均为固定枚举）
- video_usage：素材用途（枚举列表）。尽量只写 1 个；如确实同时满足多个方向且都有用，才写多个（最多 3 个）。
- object：只允许车与乘客相关（车身/轮毂/轮胎/天窗/座椅/中控屏/方向盘/驾驶员/乘客等），不要写环境（树木/建筑/天空/湖水/道路等）。控制 1-4 个。
- design_adjectives / function_adjectives：两组形容词列表，各 2-4 个；前者偏外观/质感，后者偏性能/体验；每组内部语义尽量靠拢，避免“舒适/大屏”等跨组重复。
- design_selling_points / function_selling_points：两组卖点列表，各 2-4 个；前者偏实体部件/可见结构，后者偏能力模块/特殊功能；每组内部语义尽量靠拢，不要混入环境/动作。
- scenario_a / scenario_b：两组生活/用车场景列表，各 1-4 个；A 内部语义尽量靠拢，B 与 A 尽量不同。
- marketing_phrases：营销短句/口播式检索短语（1-6 个），贴近用户语言，不要用“演示/展示”。例：雨夜看得清、堵车跟车不累、地库一把掉头、停车一把进
- topic：视频所属的大致主题（枚举，单值）。只能从:{_join_choices(index_v2_enums.TOPIC_CHOICES)} 范围里选，比如：节能快充属于电池，麋鹿测试属于恶劣路况天气，转向属于操作性，路跑属于外观
- text：画面关键文字与数值（列表）。尽量收集屏幕/UI/字幕里出现的关键词与数值：NOA/Auto Park/800V/15分钟/310公里/1500km/4.79米/27.1英寸/5K 等。
- key_words：重要关键字（枚举列表，可多值），没有看到对应的要素就不要填，只能从给定的枚举范围里选：{_join_choices(index_v2_enums.KEY_WORDS_CHOICES)}

禁止：
- 不要编造画面看不到的具体数值参数（如续航km、电池kWh等）

一致性提示（用于避免语义涣散）：
- design_* 只写“看得见/摸得着”的实体与外观：如 轮毂/车漆/门把手/座椅/中控台/屏幕/灯组/线条/材质
- function_* 只写“能力/功能/算法/性能”：如 一键AI泊车/雨夜模式/NOA/爆胎稳定控制/四轮转向/快充/主动降噪
- 同一个词不要同时出现在 design_* 与 function_*（必要时放到更匹配的一侧）
- 对于画面观感上的描述，比如画风，氛围，情绪等，不要混入设计/功能卖点和形容词里，直接放在description里 里（例：沉稳大气、科技感满满、未来感十足、年轻活力）

规范化与纠错（必须遵守）：
A) shot_type vs shot_style 不可混用：
   - 如果你要输出的值属于景别（{_join_choices(index_v2_enums.SHOT_TYPE_CHOICES)}），只能写入 shot_type。
   - shot_style 必须输出拍摄方式（{_join_choices(index_v2_enums.SHOT_STYLE_CHOICES)}），禁止输出“特写/中景/远景”等景别词。
B) weather vs time 不可混用：
   - time 只能从：{_join_choices(index_v2_enums.TIME_CHOICES)}
   - weather 只能从：{_join_choices(index_v2_enums.WEATHER_CHOICES)}
   - 禁止把“白天/夜晚/黄昏/室内”写进 weather；禁止把“雨天/雪天/阴天/晴天/极寒”等写进 time。
C) car_color 归一化（禁止输出同义变体）：
   - car_color 必须严格从：{_join_choices(index_v2_enums.CAR_COLOR_CHOICES)}
   - 禁止输出“黑色/白色/蓝色/银色/绿色”等带“色”或不在枚举里的值；颜色细节若很关键请写进 description 或 text。
D) video_usage 归一化（只允许标准枚举）：
   - video_usage(list) 必须从：{_join_choices(index_v2_enums.VIDEO_USAGE_CHOICES)}
   - 同义归并：品牌传达/品牌形象传达 -> 品牌/形象传达；权益说明 -> 权益/价格说明；路跑场景展示 -> 使用场景展示。
E) product_status_scene 不允许带括号备注：
   - product_status_scene 必须从：{_join_choices(index_v2_enums.PRODUCT_STATUS_SCENE_CHOICES)}
   - 像“含动态灯语/充电状态/节日装饰”等细节，请尽量写进 description 或 text（如果有明确屏幕文案/数字）。
   
""".strip()

DEFAULT_VISION_PROMPT_V1 = """你是一个专业的视频分镜分析师。
请分析视频片段，提取 object、search_tags 并进行商业价值评估。

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

_DEFAULT_SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_EMBEDDING_MODEL: Any = None
_EMBEDDING_MODEL_LOCK = threading.Lock()
# 单线程池：预热与 ensure 回落路径共用，避免并行重复加载/huggingface 连接风暴
_EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="st_embed")

_embedding_warmup_task: Optional[asyncio.Task[None]] = None


def get_embedding_model():
    """
    Singleton SentenceTransformer model loader.
    Loads once per process; subsequent calls reuse the same instance to avoid
    repeated "Loading weights" overhead on every reindex/search.

    Env:
        SENTENCE_TRANSFORMER_MODEL — 可选。设为**本地模型目录的绝对路径**（推荐离线），或 Hugging Face 模型 ID（需联网）。
        本地模式建议同时：HF_HUB_OFFLINE=1（或 FastAPI 启动前设 SKIP_HF_MIRROR=1），避免 lifespan 强设镜像后仍走代理；
        若日志出现 ``127.0.0.1:10808`` 代理拒绝，请清空或修正 HTTP_PROXY/HTTPS_PROXY。
        默认 ID ``paraphrase-multilingual-MiniLM-L12-v2`` 首次会从 Hub 解析/下载；不通网时请拷贝已缓存目录并设上述环境变量。
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    with _EMBEDDING_MODEL_LOCK:
        if _EMBEDDING_MODEL is not None:
            return _EMBEDDING_MODEL
        from sentence_transformers import SentenceTransformer

        model_id = (os.environ.get("SENTENCE_TRANSFORMER_MODEL") or "").strip() or _DEFAULT_SENTENCE_TRANSFORMER_MODEL
        try:
            _EMBEDDING_MODEL = SentenceTransformer(model_id)
        except Exception as e:
            log.error(
                "SentenceTransformer 加载失败，OpenSearch 向量入库会一并失败。"
                "常见原因：首次运行需访问 huggingface.co 下载权重，当前环境 SSL/网络不通。"
                "处理：1) 先把模型下载到本地，设置环境变量 SENTENCE_TRANSFORMER_MODEL=本地目录；"
                "2) 或配置 HF_ENDPOINT 镜像；3) 在无网机器上从已缓存环境复制 ~/.cache/huggingface 。"
                "model_id={} 原始错误: {}",
                model_id,
                e,
            )
            raise
        return _EMBEDDING_MODEL


def start_embedding_warmup_background() -> asyncio.Task[None]:
    """
    在应用 lifespan 中尽早 create_task：不阻塞 yield，HTTP 可先就绪；
    向量模型在线程池加载。入库前须 await ensure_embedding_model_ready()，以等待本任务完成而非另起加载。
    """
    global _embedding_warmup_task
    if _embedding_warmup_task is not None:
        return _embedding_warmup_task

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        log.info("开始后台预热向量模型 (SentenceTransformer)...")
        try:
            await loop.run_in_executor(_EMBED_EXECUTOR, get_embedding_model)
            log.info("向量模型预热完成。")
        except Exception as e:
            log.error("向量模型预热失败: {}", e)
            raise

    _embedding_warmup_task = asyncio.create_task(_run())
    return _embedding_warmup_task


async def ensure_embedding_model_ready() -> None:
    """OpenSearch 向量化前调用：若启动时已暖机则等待暖机结束，避免与主线程重复抢载。"""
    if _EMBEDDING_MODEL is not None:
        return
    if _embedding_warmup_task is not None:
        await _embedding_warmup_task
        if _EMBEDDING_MODEL is None:
            raise RuntimeError("向量模型未就绪：预热已结束但未成功加载，请查看日志中的 SentenceTransformer 错误")
        return
    loop = asyncio.get_running_loop()
    log.info("未检测到后台预热任务，在线程池加载向量模型...")
    await loop.run_in_executor(_EMBED_EXECUTOR, get_embedding_model)


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
        schema = get_workspace(workspace).schema_class.model_json_schema()
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


def _probe_video_metadata(local_video_path: str, project_id: str) -> tuple[float, str, str]:
    """
    OpenCV 同步读媒体信息。须在 asyncio.to_thread 中调用，
    避免批量视频分析时阻塞事件循环（否则 /health 与其它 API 一起卡死）。
    """
    try:
        cap = cv2.VideoCapture(local_video_path)
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

        resolution = f"{width}x{height}" if width and height else "未知"
        video_duration = frame_count / fps if fps and fps > 0 else 0.0

        def get_aspect_ratio(w: int, h: int) -> str:
            if w == 0 or h == 0:
                return "未知"
            g = math.gcd(w, h)
            rw, rh = w // g, h // g
            if rw == 8 and rh == 9:
                return "16:9"
            return f"{rw}:{rh}"

        frame_size = get_aspect_ratio(width, height)
        if width == 1920 and height == 1080:
            frame_size = "16:9"
        elif width == 1080 and height == 1920:
            frame_size = "9:16"
        return video_duration, resolution, frame_size
    except Exception as e:
        log.warning(f"[{project_id}] 获取视频媒体信息失败: {e}")
        return 0.0, "未知", "未知"


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

    basename = os.path.basename(local_video_path)
    user_car = (car_model or "").strip()
    # 卖点词表仅按用户在表单中选择的车型注入（与 _get_or_upload_source_video 使用同一 car_model）
    car_key_for_glossary = normalize_zhiji_car_key(car_model)

    video_duration, resolution, frame_size = await asyncio.to_thread(
        _probe_video_metadata, local_video_path, project_id
    )

    workspace_dir = workspace_dir or f"./video_analysis_workspace/{project_id}"
    obs_key_prefix = f"ai_picture/video_analysis/{project_id}"

    appendix = ""
    ws_cfg = get_workspace(workspace)
    prompt = custom_prompt if custom_prompt else ws_cfg.default_prompt
    if workspace == "v2":
        appendix = build_v2_vision_selling_appendix(car_key_for_glossary)
        if appendix:
            prompt = f"{prompt}\n\n{appendix}"
        prompt = f"{prompt}\n\n可以额外参考视频文件名（辅助卖点/语境，仍以画面为准）：{basename}"
        log.info(
            f"[{project_id}] v2 vision 提示元数据: form_car_model={user_car!r} "
            f"car_key_for_glossary={car_key_for_glossary!r} appendix_chars={len(appendix)} basename={basename}"
        )

    log.info(
        "[{}] vision 提示词总长 {} 字符 workspace={} custom_prompt={}",
        project_id,
        len(prompt),
        workspace,
        bool(custom_prompt),
    )
    log.info("[{}] vision 提示词全文:\n{}", project_id, prompt)

    # Step 1: 分镜检测 + 抽帧 (CPU 密集, 放到线程池)
    
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
            car_model=(user_car or None),
            resolution=resolution,
            frame_size=frame_size,
            video_duration=video_duration,
            project_id=project_id,
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
    opensearch_index_name: Optional[str] = None,
) -> dict:
    """
    Convenience method:
    - build embeddings
    - bulk index into `car_interior_analysis`
    """
    if embedding_model is None:
        await ensure_embedding_model_ready()
        embedding_model = get_embedding_model()

    docs = await map_shotcards_to_car_interior_docs(cards, embedding_model=embedding_model, id_prefix=id_prefix, workspace=workspace)
    if not docs:
        return {"success": True, "items": 0}
    
    ws_cfg = get_workspace(workspace)
    resp = await bulk_index(
        ws_cfg.index_class,
        docs,
        refresh=refresh,
        index_name_override=opensearch_index_name,
    )
    return {"success": True, "items": len(docs), "opensearch": resp}
