"""
Test runner: analyze selected videos with SceneAnalysisResultV2 schema.

- Skips scene splitting (single segment)
- Extracts key frames
- Uploads frames to OBS（``ai_picture/video_analysis_frames/{video_key}/{scene_id}/``，``video_key`` 与 ``video_analysis_video_v2.video_key`` / 自增 id 对齐；本地抽帧目录为 ``workspace/frames/{video_key}/``，纯 ASCII，避免中文路径写盘失败）
- Calls Doubao vision with JSON schema = SceneAnalysisResultV2
- Writes outputs to a local JSONL file for inspection
- 分镜结果写入 MySQL `video_analysis_shot_cards_v2`（常量 `SHOT_CARDS_VERSION`）

Usage:
  python -m src.test.run_video_analysis_v2
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import csv
import json
import os
import sys
import random
import hashlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure we can import from src/ when running as a module or directly.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models.pydantic.model_output_schema.video_analysis_schema import SceneAnalysisResultV2
from models.pydantic.dataclass.scene_split_result import SceneSplitResult
from utils.video_process_utils import get_video_scenes
from utils.obs_utils import batch_upload_to_obs, download_url_to_file, upload_to_obs, obs_key_exists, OBS_BASE_URL
from utils.call_model_utils import call_doubao_vision
from PIL import Image
from services.video_analysis_db_service import video_analysis_db_service, shot_card_v2_item_dict_from_scene
from models.pydantic.opensearch_index import index_v2_enums
# --- Optional ingestion (OpenSearch IndexV2) ---
from sentence_transformers import SentenceTransformer
from infra.storage.opensearch.create_index import index_manager
from infra.storage.opensearch.document_writer import bulk_index
from infra.storage.opensearch_connector import opensearch_connector
from infra.storage.mysql_connector import mysql_connector
from infra.storage.sqlmodel_init import create_tables_if_not_exists
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from services.video_upload_cache_service import video_upload_cache_service


BASE_VIDEO_DIR = Path(r"C:\Users\admin\Downloads\LS6视频")
# For reproducible cache tests: when non-empty, only analyze these videos.
OVERRIDE_VIDEOS: List[str] = [
    r"D:\wsn_data\aigc_data\数字人素材（LS9、全新L6）\LS9\冰雪\20251216-LS9官号-双车漂移-1.mp4"
]
CAR_MODEL = "LS6"
# Sampling strategy:
# - Scan all mp4 under BASE_VIDEO_DIR
# - Bucket by filename keywords (roughly "subject/topic")
# - Pick a few from each bucket to form a diverse test set
BUCKETS: List[Tuple[str, List[str]]] = [
    ("智驾NOA/领航", ["NOA", "城市NOA", "IMAD", "领航", "智驾", "横向避让"]),
    ("泊车/APA", ["泊车", "APA", "一键泊车", "入库", "车位"]),
    ("雨夜/补盲", ["雨夜", "雨天", "补盲", "DZT"]),
    ("冰雪/低温", ["冰雪", "低温", "零下", "冬季"]),
    ("高温/夏季", ["高温", "夏", "40度"]),
    ("四轮转向/掉头", ["四轮转向", "小转弯半径", "掉头", "转弯半径"]),
    ("底盘/空悬/越野", ["底盘", "空悬", "交叉轴", "脱困", "越野", "烂路", "爬坡"]),
    ("安全/爆胎/防侧翻", ["爆胎", "防侧翻", "安全", "车身结构", "撞击"]),
    ("补能/能源", ["快充", "充电", "充电口", "加油口", "电池", "发动机"]),
    ("座舱屏幕/车机", ["智慧屏", "驾驶屏", "车机", "大屏", "屏保"]),
    ("舒适座椅", ["零重力", "贵妃椅", "按摩", "座椅", "通风", "加热"]),
    ("空间/装载", ["后备箱", "前备箱", "装载", "空间", "车内空间"]),
    ("音响/降噪/静谧", ["音响", "喇叭", "降噪", "静音", "隔音"]),
    ("灯光/外观细节", ["前车灯", "车前灯", "尾灯", "isc", "车标", "门把手", "外观", "车轮", "轮辋"]),
    ("展厅/车阵/活动", ["展厅", "车阵", "武汉", "户外"]),
]

# How many files to pick from each bucket (tune as needed)
PER_BUCKET = 6
RANDOM_SEED = 20260425

# Overall concurrency for processing multiple videos in parallel.
MAX_VIDEO_CONCURRENCY = 50
# Cap CPU-heavy frame extraction to avoid maxing out ffmpeg/opencv threads.
MAX_EXTRACT_CONCURRENCY = 10
# Threadpool size used by asyncio.to_thread (ffmpeg/opencv extract + OBS SDK upload).
THREADPOOL_WORKERS = 50

# Persist sampled testset + analysis results to reduce rework across runs.
TESTSET_CACHE_PATH = Path(__file__).resolve().parent / "workspace" / "testset_v2.json"
ANALYSIS_CACHE_PATH = Path(__file__).resolve().parent / "workspace" / "analysis_cache_v2.json"
# OBS source video upload cache: moved to MySQL (see models.sqlmodel.video_upload_cache).
USE_ANALYSIS_CACHE = True  # keep False to observe upload/frame cache logs when debugging

# Ingest into OpenSearch after analysis (can turn off quickly).
ENABLE_INGEST = True
# 分镜入库：写入 MySQL `video_analysis_shot_cards_v2`（与 SceneAnalysisResultV2 对齐）。
SHOT_CARDS_VERSION = "v2"
SPLIT_SCENES = True
MIN_SCENE_SECONDS = 2.0  # merge scenes shorter than this threshold (avoid too-fragmented cuts)
MAX_SCENE_CONCURRENCY = 10
# True：重新抽帧+分镜并覆盖库里的参考帧与分镜行；False：沿用库里已有参考帧与分镜时间轴（仍跑视觉理解）。
FRAME_OVERWRITE = True


def _join_choices(xs: List[str]) -> str:
    return ", ".join([x for x in (xs or []) if x])


PROMPT_V2 = f"""
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

def random_pick(k: int, xs: List[str|Path]) -> List[str]:
    random.seed(RANDOM_SEED)
    return random.sample(xs, min(k, len(xs)))

def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

_extract_sem: asyncio.Semaphore | None = None
_cache_lock: asyncio.Lock | None = None


def _load_json(path: Path, default: Any):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _analysis_cache_key(video_path: str, *, frame_interval: float) -> str:
    """
    Key includes:
    - absolute path + mtime (content changes)
    - frame_interval (extraction strategy)
    - prompt + schema (labeling spec)
    """
    try:
        st = os.stat(video_path)
        mtime = int(st.st_mtime)
        size = int(st.st_size)
    except Exception:
        mtime = 0
        size = 0
    schema_sig = json.dumps(SceneAnalysisResultV2.model_json_schema(), ensure_ascii=False, sort_keys=True)
    sig = f"{video_path}|{mtime}|{size}|{frame_interval}|{PROMPT_V2}|{schema_sig}"
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()


def _video_sig(video_path: str) -> str:
    """
    Stable signature for a local video file. Used for:
    - video_id stability (so frame folders are discoverable)
    - OBS upload de-duplication
    """
    try:
        st = os.stat(video_path)
        mtime = int(st.st_mtime)
        size = int(st.st_size)
    except Exception:
        mtime = 0
        size = 0
    s = f"{os.path.abspath(video_path)}|{mtime}|{size}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _video_id_for(video_path: str) -> str:
    return f"v2_{_video_sig(video_path)}"

def _existing_frames(workspace_dir: Path) -> List[str]:
    """
    Return existing extracted frame file paths if workspace already contains them.
    """
    if not workspace_dir.exists():
        return []
    pats = ["scene_001_frame_*.webp", "scene_001_frame_*.jpg", "scene_001_frame_*.png"]
    frames: List[Path] = []
    for pat in pats:
        frames.extend(workspace_dir.glob(pat))
    # Sort by filename (frame index is zero-padded)
    frames = sorted([p for p in frames if p.is_file()], key=lambda p: p.name)
    return [str(p) for p in frames]


def _is_remote_frame_path(p: str) -> bool:
    s = str(p or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")


async def _ensure_local_split_frames_for_scenes(
    workspace_dir: Path,
    *,
    video_db_id: int,
    scenes: List[SceneSplitResult],
) -> None:
    """
    先检查各镜 ``frame_url_list`` 是否已是存在的本地路径；否则按顺序：
    1) ``video_analysis_scene_split_frames_cache`` 中该 ``video_id``+``scene_id`` 的 OBS URL；
    2) 仍无则使用当前列表里已有的 http(s) URL；
    再按 ``scene_{sid:03d}_frame_{idx:06d}.webp`` 下载缺失文件并回写 ``frame_url_list``。
    """
    if video_db_id <= 0 or not scenes:
        return
    workspace_dir.mkdir(parents=True, exist_ok=True)
    urls_map = await video_analysis_db_service.get_split_frame_obs_cache(video_db_id)
    for s in scenes:
        sid = int(getattr(s, "scene_id", 0) or 0)
        if sid <= 0:
            continue
        paths = list(s.frame_url_list or [])
        locals_ok = bool(paths) and all(
            (not _is_remote_frame_path(str(p))) and os.path.isfile(str(p)) for p in paths
        )
        if locals_ok:
            continue
        cand = urls_map.get(sid)
        if not cand:
            cand = [str(p).strip() for p in paths if _is_remote_frame_path(str(p))]
        if not cand:
            continue
        new_locals: List[str] = []
        for i, url in enumerate(cand):
            dest = workspace_dir / f"scene_{sid:03d}_frame_{i:06d}.webp"
            if dest.is_file():
                new_locals.append(str(dest))
                continue
            await download_url_to_file(str(url), str(dest))
            new_locals.append(str(dest))
        s.frame_url_list = new_locals


def _merge_short_scenes(scenes, *, min_seconds: float):
    """
    Merge very short scenes into previous one to avoid over-fragmentation.
    Scenes are SceneSplitResult dataclasses.
    """
    if not scenes:
        return scenes
    out = []
    for s in scenes:
        try:
            dur = float(getattr(s, "duration_seconds", 0.0) or 0.0)
        except Exception:
            dur = 0.0
        if out and dur > 0 and dur < float(min_seconds or 0.0):
            prev = out[-1]
            # merge frame lists + extend end_time/duration
            prev.frame_url_list = (prev.frame_url_list or []) + (s.frame_url_list or [])
            try:
                prev.end_time = float(getattr(s, "end_time", prev.end_time) or prev.end_time)
            except Exception:
                pass
            try:
                prev.duration_seconds = float(prev.end_time) - float(prev.start_time)
            except Exception:
                # fallback: add durations
                try:
                    prev.duration_seconds = float(getattr(prev, "duration_seconds", 0.0) or 0.0) + dur
                except Exception:
                    pass
            continue
        out.append(s)
    # re-number scene_id to keep contiguous _scene_001..
    for i, s in enumerate(out):
        try:
            s.scene_id = i + 1
        except Exception:
            pass
    return out


def _normalize_model_output(data: Any) -> Any:
    """
    Defensive normalization for model outputs to match our index schema.
    Some models still return list for single-value enum fields.
    """
    if not isinstance(data, dict):
        return data

    def _first_str(v: Any) -> str:
        if isinstance(v, list):
            xs = [str(x).strip() for x in v if str(x).strip()]
            return xs[0] if xs else "未知"
        if isinstance(v, str) and v.strip():
            return v.strip()
        return "未知"

    # Index expects string for these fields.
    if isinstance(data.get("shot_style"), list) or not isinstance(data.get("shot_style"), str):
        data["shot_style"] = _first_str(data.get("shot_style"))
    if isinstance(data.get("shot_type"), list) or not isinstance(data.get("shot_type"), str):
        data["shot_type"] = _first_str(data.get("shot_type"))
    if isinstance(data.get("camera_movement"), list) or not isinstance(data.get("camera_movement"), str):
        data["camera_movement"] = _first_str(data.get("camera_movement"))

    ghq = data.get("generic_hq_road_run")
    if isinstance(ghq, str):
        data["generic_hq_road_run"] = ghq.strip().lower() in ("true", "1", "yes", "是")
    elif ghq is None:
        data["generic_hq_road_run"] = False
    else:
        data["generic_hq_road_run"] = bool(ghq)
    return data


def _ok_scene_result_rows(scene_results: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in scene_results or []:
        if isinstance(r, dict) and r.get("success") and isinstance(r.get("result"), dict):
            out.append(r)
    return out


def _frame_urls_from_scene_result_row(r: Dict[str, Any]) -> List[str]:
    """Resolve OBS/local frame URLs from a scene_result envelope (top-level or nested in result)."""
    fus = r.get("frame_urls")
    if isinstance(fus, list) and fus:
        return [str(x) for x in fus if x]
    res = r.get("result") if isinstance(r.get("result"), dict) else {}
    fus2 = res.get("frame_urls")
    if isinstance(fus2, list) and fus2:
        return [str(x) for x in fus2 if x]
    return []


async def _persist_v2_shot_cards_db(
    *,
    video_key: str,
    video_name: str,
    video_url: str,
    scene_results: Any,
) -> None:
    """
    Upsert v2 分镜表 + 分镜参考帧表（全量跑与缓存命中时调用）。
    """
    rows = _ok_scene_result_rows(scene_results)
    if not rows:
        return
    try:
        scene_frames: Dict[int, List[str]] = {}
        for i, r in enumerate(rows):
            if not isinstance(r, dict):
                continue
            res = r.get("result") if isinstance(r.get("result"), dict) else {}
            sid = int(r.get("scene_id") or res.get("scene_id") or 0)
            if sid <= 0:
                sid = i + 1
            scene_frames[sid] = _frame_urls_from_scene_result_row(r)
        cards = [shot_card_v2_item_dict_from_scene(scene=r, analysis=r.get("result") or {}) for r in rows]
        await video_analysis_db_service.upsert_history_item(
            {
                "id": video_key,
                "name": video_name,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "video_url": video_url,
                "cards": cards,
                "scene_frames": scene_frames,
            },
            shot_cards_version=SHOT_CARDS_VERSION,
        )
    except Exception:
        print(f"[PERSIST_V2_DB_ERROR] video_key={video_key!r}\n{traceback.format_exc()}")


async def _try_load_scenes_from_db_without_overwrite(video_key: str) -> Optional[List[SceneSplitResult]]:
    """overwrite=False 时：从 MySQL 读已有分镜时间轴 + 参考帧 URL，跳过抽帧与分镜检测。"""
    try:
        item = await video_analysis_db_service.get_history_item(video_key, shot_cards_version=SHOT_CARDS_VERSION)
    except Exception:
        return None
    if not isinstance(item, dict):
        return None
    cards = item.get("cards") or []
    if not cards:
        return None
    scenes: List[SceneSplitResult] = []
    for c in sorted(cards, key=lambda x: int(x.get("scene_id") or 0)):
        fus = c.get("frame_urls") or []
        if not fus:
            return None
        scenes.append(
            SceneSplitResult(
                scene_id=int(c.get("scene_id") or 0),
                frame_url_list=list(fus),
                start_time=float(c.get("start_time") or 0.0),
                end_time=float(c.get("end_time") or 0.0),
                duration_seconds=float(c.get("duration_seconds") or 0.0),
            )
        )
    return scenes or None


async def _upload_source_video_once(video_path: str) -> str:
    """
    Upload the original video to OBS under:
      ai_picture/car_video_analysis/source_video/{CAR_MODEL}/<basename>

    不按 analysis 的 video_id 分子目录，便于用文件名推断 OBS 路径。
    上传缓存按 ``file_name``（basename）查 ``video_source_upload_cache``。

    Returns the OBS URL (https://.../key).
    """
    abs_path = os.path.abspath(video_path)
    fname = os.path.basename(video_path)
    try:
        st = os.stat(video_path)
        mtime = int(st.st_mtime)
        size = int(st.st_size)
    except Exception:
        mtime = 0
        size = 0

    # Ensure table exists (best-effort). If DB is down, we still attempt upload.
    try:
        await create_tables_if_not_exists()
    except Exception:
        pass

    try:
        cached = await video_upload_cache_service.get_by_file_name(fname)
    except Exception:
        cached = None

    if isinstance(cached, dict):
        url = str(cached.get("obs_url") or "").strip()
        key = str(cached.get("obs_key") or "").strip()
        if url:
            if key and obs_key_exists(key):
                print(f"[VIDEO_UPLOAD_DB_CACHE_HIT] file_name={fname!r} url={url}")
                return url
            if not key:
                # No key stored; trust url as best-effort.
                print(f"[VIDEO_UPLOAD_DB_CACHE_HIT_NO_KEY] file_name={fname!r} url={url}")
                return url

    obs_prefix = f"ai_picture/car_video_analysis/source_video/{CAR_MODEL}/"
    obs_key = os.path.join(obs_prefix, fname).replace("\\", "/")

    # If already exists on OBS, skip uploading.
    if obs_key_exists(obs_key):
        url = f"{OBS_BASE_URL}/{obs_key}"
        print(f"[VIDEO_UPLOAD_OBS_EXISTS] file_name={fname!r} key={obs_key} url={url}")
    else:
        print(f"[VIDEO_UPLOAD_PUT] file_name={fname!r} key={obs_key}")
        url = await upload_to_obs(video_path, obs_prefix)

    # Persist to DB cache (best-effort; do not break analysis flow).
    try:
        await video_upload_cache_service.upsert(
            file_name=fname,
            abs_path=abs_path,
            file_size=size,
            file_mtime=mtime,
            obs_key=obs_key,
            obs_url=url,
        )
    except Exception:
        pass
    return url


def _get_first_frame_size(local_frames: List[str]) -> tuple[int, int] | None:
    for p in (local_frames or []):
        try:
            with Image.open(p) as img:
                w, h = img.size
                if w and h:
                    return int(w), int(h)
        except Exception:
            continue
    return None


def _aspect_label(w: int, h: int) -> str:
    if not w or not h:
        return "未知"
    # Rough bucketing (good enough for filtering)
    r = w / h
    if abs(r - (16 / 9)) < 0.08:
        return "横版16:9"
    if abs(r - (9 / 16)) < 0.08:
        return "竖版9:16"
    return "其他比例"


async def analyze_one(
    video_path: str,
    *,
    frame_interval: float,
    frame_overwrite: Optional[bool] = None,
) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        return {"video": video_path, "success": False, "error": "file not found"}

    # Cache: if we've already analyzed this exact video under the same spec, reuse result.
    cache_key = _analysis_cache_key(video_path, frame_interval=frame_interval)
    cache = _load_json(ANALYSIS_CACHE_PATH, default={})
    cached = cache.get(cache_key)
    if (
        USE_ANALYSIS_CACHE
        and isinstance(cached, dict)
        and cached.get("success")
        and isinstance(cached.get("scene_results"), list)
    ):
        vk = str(cached.get("video_key") or cached.get("video_id") or _video_id_for(video_path))
        # Normalize cached results in case a previous run crashed before normalization was introduced.
        for sr in cached.get("scene_results") or []:
            if isinstance(sr, dict) and isinstance(sr.get("result"), dict):
                sr["result"] = _normalize_model_output(sr.get("result"))
        print(f"[ANALYSIS_CACHE_HIT] key={cache_key} video={video_path}")
        # 缓存命中仍同步写 v2 分镜表，否则 OpenSearch 有文档但 MySQL v2 无 cards，print 脚本等会落空。
        vid_cached = Path(video_path)
        ws_dir_hit = Path(__file__).resolve().parent / "workspace" / "frames" / str(vk)
        await _persist_v2_shot_cards_db(
            video_key=vk,
            video_name=vid_cached.name,
            video_url=str(cached.get("obs_video_url") or "").strip() or video_path,
            scene_results=cached.get("scene_results"),
        )
        return {
            "video": video_path,
            "video_key": vk,
            "video_id": vk,
            "db_video_id": cached.get("db_video_id"),
            "obs_video_url": cached.get("obs_video_url") or "",
            "workspace_dir": str(ws_dir_hit),
            "frames": cached.get("frames") or [],
            "scene_results": cached.get("scene_results") or [],
            "success": True,
            "cached": True,
        }

    vid = Path(video_path)
    fname = vid.name

    try:
        await create_tables_if_not_exists()
    except Exception:
        pass

    db_video_id = 0
    video_key = ""
    try:
        db_video_id, video_key = await video_analysis_db_service.resolve_video_v2_for_source_file(file_name=fname)
    except Exception:
        db_video_id, video_key = 0, ""
        print(f"[RESOLVE_VIDEO_V2_EXCEPTION] file={fname!r}\n{traceback.format_exc()}")

    if db_video_id <= 0 or not video_key:
        return {
            "video": video_path,
            "success": False,
            "error": "mysql video_analysis_video_v2 resolve failed (resolve_video_v2_for_source_file); "
            "see [RESOLVE_VIDEO_V2_EXCEPTION] traceback or add column video_analysis_video_v2.source_file_name.",
        }

    # 本地抽帧目录：仅用 video_key（= str(video_id)），避免中文/特殊字符路径导致 cv2 写盘失败
    workspace_dir = Path(__file__).resolve().parent / "workspace" / "frames" / str(video_key)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Upload original source video to OBS (cached), and store OBS url into DB history.
    obs_video_url = ""
    try:
        obs_video_url = await _upload_source_video_once(video_path)
    except Exception:
        obs_video_url = ""

    # Insert history row early so we can trace OpenSearch doc ids back to DB.
    try:
        await video_analysis_db_service.upsert_history_item(
            {
                "id": video_key,
                "name": vid.name,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                # Store OBS source video path for mix/cut workflows; fallback to local path if upload failed.
                "video_url": obs_video_url or video_path,
                "cards": [],
            },
            shot_cards_version=SHOT_CARDS_VERSION,
        )
    except Exception:
        # Non-blocking: analysis can continue even if DB is unavailable.
        pass

    frame_overwrite = bool(FRAME_OVERWRITE if frame_overwrite is None else frame_overwrite)

    scenes: Optional[List[SceneSplitResult]] = None
    if not frame_overwrite:
        scenes = await _try_load_scenes_from_db_without_overwrite(video_key)
        if scenes:
            print(f"[SCENE_DB_REUSE] video_key={video_key} scenes={len(scenes)} overwrite=False")
            try:
                await _ensure_local_split_frames_for_scenes(
                    workspace_dir, video_db_id=db_video_id, scenes=scenes
                )
            except Exception as e:
                print(f"[SPLIT_FRAME_CACHE_HYDRATE_WARN] video_key={video_key} err={e}")

    if scenes is None:
        local_frames = _existing_frames(workspace_dir)
        if not local_frames:
            print(f"[FRAMES_CACHE_MISS] video_key={video_key} workspace_dir={workspace_dir}")
            if _extract_sem is None:
                scenes = await asyncio.to_thread(
                    get_video_scenes, video_path, frame_interval, 30.0, str(workspace_dir)
                )
            else:
                async with _extract_sem:
                    scenes = await asyncio.to_thread(
                        get_video_scenes, video_path, frame_interval, 30.0, str(workspace_dir)
                    )
        else:
            print(
                f"[FRAMES_CACHE_HIT] video_key={video_key} reused_frames={len(local_frames)} workspace_dir={workspace_dir}"
            )
            try:
                scenes = await asyncio.to_thread(
                    get_video_scenes, video_path, frame_interval, 30.0, str(workspace_dir)
                )
            except Exception:
                scenes = None

    if not scenes:
        return {"video": video_path, "success": False, "error": "no scenes extracted"}

    if SPLIT_SCENES and MIN_SCENE_SECONDS and len(scenes) > 1:
        scenes = _merge_short_scenes(scenes, min_seconds=float(MIN_SCENE_SECONDS))

    try:
        await _ensure_local_split_frames_for_scenes(
            workspace_dir, video_db_id=db_video_id, scenes=scenes
        )
    except Exception as e:
        print(f"[SPLIT_FRAME_CACHE_HYDRATE_WARN] video_key={video_key} err={e}")

    local_frames: List[str] = []
    for s in scenes:
        for p in s.frame_url_list or []:
            if not _is_remote_frame_path(str(p)):
                local_frames.append(str(p))

    missing = [p for p in local_frames if not os.path.exists(p)]
    if missing:
        return {
            "video": video_path,
            "video_key": video_key,
            "video_id": video_key,
            "db_video_id": db_video_id,
            "success": False,
            "error": f"missing {len(missing)} extracted frames",
            "missing": missing[:5],
            "workspace_dir": str(workspace_dir),
        }

    locals_for_dim = [p for p in local_frames if os.path.exists(p)]
    wh = _get_first_frame_size(locals_for_dim)
    frame_w, frame_h = wh if wh else (0, 0)
    resolution = f"{frame_w}x{frame_h}" if frame_w and frame_h else "未知"
    frame_size = _aspect_label(frame_w, frame_h) if frame_w and frame_h else "未知"

    scene_frame_urls_by_sid: Dict[int, List[str]] = {}
    obs_frame_urls: List[str] = []
    if db_video_id <= 0:
        return {
            "video": video_path,
            "video_key": video_key,
            "video_id": video_key,
            "db_video_id": 0,
            "success": False,
            "error": "mysql video_analysis_video_v2 id missing (resolve_video_v2_for_source_file returned invalid id)",
        }

    if local_frames:
        for s in scenes:
            sid = int(getattr(s, "scene_id", 1) or 1)
            scene_locals = [str(p) for p in (s.frame_url_list or []) if not _is_remote_frame_path(str(p))]
            if scene_locals:
                # 抽帧结果与理解 schema 解耦：路径仅用稳定 video_key + scene_id（v1/v2 可共用同一套 OBS 帧）
                prefix = f"ai_picture/video_analysis_frames/{video_key}/{sid}/"
                urls = await batch_upload_to_obs(
                    file_paths=scene_locals,
                    obs_key_prefix=prefix,
                    max_concurrency=50,
                )
                scene_frame_urls_by_sid[sid] = urls
                obs_frame_urls.extend(urls)
            else:
                scene_frame_urls_by_sid[sid] = [str(u) for u in (s.frame_url_list or []) if u]
                obs_frame_urls.extend(scene_frame_urls_by_sid[sid])
    else:
        for s in scenes:
            sid = int(getattr(s, "scene_id", 1) or 1)
            urls = [str(u) for u in (s.frame_url_list or []) if u]
            scene_frame_urls_by_sid[sid] = urls
            obs_frame_urls.extend(urls)

    try:
        await video_analysis_db_service.replace_split_frame_obs_cache(db_video_id, scene_frame_urls_by_sid)
    except Exception as e:
        print(f"[SPLIT_FRAME_CACHE_PERSIST_WARN] video_key={video_key} err={e}")

    schema_json = SceneAnalysisResultV2.model_json_schema()

    sem_scene = asyncio.Semaphore(MAX_SCENE_CONCURRENCY)
    fn = os.path.basename(video_path)

    async def _analyze_scene(scene_obj: SceneSplitResult):
        async with sem_scene:
            sid = int(getattr(scene_obj, "scene_id", 1) or 1)
            st = float(getattr(scene_obj, "start_time", 0.0) or 0.0)
            et = float(getattr(scene_obj, "end_time", 0.0) or 0.0)
            dur = float(getattr(scene_obj, "duration_seconds", max(0.0, et - st)) or 0.0)
            scene_frame_urls = list(scene_frame_urls_by_sid.get(sid) or [])
            if not scene_frame_urls:
                return {
                    "scene_id": sid,
                    "success": False,
                    "error": "empty scene frame urls",
                    "failure_response": None,
                }

            raw = None
            try:
                raw = await call_doubao_vision(PROMPT_V2 + f"\n可以额外参考视频名:{fn}", scene_frame_urls, schema_json)
                if raw is None:
                    return {
                        "scene_id": sid,
                        "success": False,
                        "error": "doubao_vision returned None",
                        "failure_response": None,
                    }
                data = raw if isinstance(raw, dict) else json.loads(raw)
                if not isinstance(data, dict):
                    return {
                        "scene_id": sid,
                        "success": False,
                        "error": f"unexpected model output type: {type(data).__name__}",
                        "failure_response": raw,
                    }
                data = _normalize_model_output(data)

                data["id"] = f"{video_key}_scene_{sid:03d}"
                data.setdefault("car_model", "未知")
                data["frame_size"] = frame_size
                data["resolution"] = resolution
                data["video_duration"] = float(dur or 0.0)
                data["start_time"] = float(st or 0.0)
                data["end_time"] = float(et or 0.0)
                return {
                    "scene_id": sid,
                    "start_time": st,
                    "end_time": et,
                    "duration_seconds": dur,
                    "frame_urls": scene_frame_urls,
                    "result": data,
                    "success": True,
                }
            except Exception as e:
                return {
                    "scene_id": sid,
                    "success": False,
                    "error": str(e),
                    "failure_response": raw,
                }

    scene_results = await asyncio.gather(*[asyncio.create_task(_analyze_scene(s)) for s in scenes])
    for sr in scene_results:
        if isinstance(sr, dict) and isinstance(sr.get("result"), dict):
            sr["result"] = _normalize_model_output(sr.get("result"))
    await _persist_v2_shot_cards_db(
        video_key=video_key,
        video_name=vid.name,
        video_url=obs_video_url or video_path,
        scene_results=scene_results,
    )

    out = {
        "video": video_path,
        "video_key": video_key,
        "video_id": video_key,
        "db_video_id": db_video_id,
        "obs_video_url": obs_video_url,
        "workspace_dir": str(workspace_dir),
        "frames": obs_frame_urls,
        "scene_results": scene_results,
        "success": True,
    }

    try:
        global _cache_lock
        if _cache_lock is None:
            _cache_lock = asyncio.Lock()
        async with _cache_lock:
            cache = _load_json(ANALYSIS_CACHE_PATH, default={})
            cache[cache_key] = {
                "success": True,
                "video_key": video_key,
                "video_id": video_key,
                "db_video_id": db_video_id,
                "obs_video_url": obs_video_url,
                "frames": obs_frame_urls,
                "scene_results": scene_results,
                "video": video_path,
                "workspace_dir": str(workspace_dir),
            }
            _save_json(ANALYSIS_CACHE_PATH, cache)
    except Exception:
        pass

    return out


def _list_all_mp4(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []
    return sorted([p for p in base_dir.rglob("*.mp4") if p.is_file()])


def build_test_set() -> List[str]:
    # If persisted testset exists, reuse it to keep runs stable and avoid re-analysis churn.
    cached = _load_json(TESTSET_CACHE_PATH, default=None)
    if isinstance(cached, list) and cached and all(isinstance(x, str) for x in cached):
        return cached

    all_videos = _list_all_mp4(BASE_VIDEO_DIR)
    if not all_videos:
        return []

    rng = random.Random(RANDOM_SEED)

    # Pre-lower for matching (but keep original path)
    candidates = [(p, p.name.lower()) for p in all_videos]

    selected: List[Path] = []
    selected_set = set()

    for _, keywords in BUCKETS:
        kws = [k.lower() for k in keywords]
        bucket = [p for (p, name_l) in candidates if any(k in name_l for k in kws)]
        # Shuffle to avoid always picking the same numbers/suffixes
        rng.shuffle(bucket)
        for p in bucket[:PER_BUCKET]:
            if str(p) in selected_set:
                continue
            selected.append(p)
            selected_set.add(str(p))

    # Fallback: add some random leftovers to broaden coverage
    leftovers = [p for (p, _) in candidates if str(p) not in selected_set]
    rng.shuffle(leftovers)
    selected.extend(leftovers[:20])

    # Keep deterministic order for readability (grouped by bucket selection order first, then leftovers)
    out = [str(p) for p in selected]
    _save_json(TESTSET_CACHE_PATH, out)
    return out


async def main():
    # Easy to run from PyCharm: just hit Run.
    # IMPORTANT: asyncio.to_thread uses the event loop's default executor.
    # Increase it so video extraction + OBS SDK uploads can run concurrently.
    loop = asyncio.get_running_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS))

    frame_interval = 2.0
    out_dir = Path(__file__).resolve().parent / "sample"
    # Analyze override videos when provided; otherwise analyze all under BASE_VIDEO_DIR.
    # videos = [str(p) for p in _list_all_mp4(BASE_VIDEO_DIR)]
    videos = [str(p) for p in random_pick(50, _list_all_mp4(BASE_VIDEO_DIR))]
    print("test videos")
    for v in videos:
        print(v)

    out_dir.mkdir(parents=True, exist_ok=True)

    if not videos:
        print(f"Base video dir not found or empty: {BASE_VIDEO_DIR}")
        return

    print(f"Found {len(videos)} test videos. Output dir: {out_dir.resolve()}")

    global _extract_sem
    _extract_sem = asyncio.Semaphore(MAX_EXTRACT_CONCURRENCY)
    sem = asyncio.Semaphore(MAX_VIDEO_CONCURRENCY)

    def _out_file_for(vp: str) -> Path:
        stem = Path(vp).stem
        safe_stem = "".join([c if (c.isalnum() or c in ("-", "_", ".", " ")) else "_" for c in stem]).strip()
        if not safe_stem:
            safe_stem = "video"
        return out_dir / f"{safe_stem}.json"

    async def _run_one(vp: str) -> Dict[str, Any]:
        async with sem:
            print(f"\n=== Analyzing: {vp}")
            try:
                r = await analyze_one(vp, frame_interval=float(frame_interval))
            except Exception as e:
                r = {"video": vp, "success": False, "error": str(e), "failure_response": None}

            _out_file_for(vp).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

            if r.get("success"):
                # analyze_one 返回的是 scene_results，没有顶层 result；避免 OK: null 误导
                srs = r.get("scene_results") or []
                ok_n = len(_ok_scene_result_rows(srs))
                print(
                    "OK:",
                    json.dumps(
                        {
                            "video_id": r.get("video_id"),
                            "cached": bool(r.get("cached")),
                            "scene_results_ok": ok_n,
                            "scene_results_total": len(srs),
                        },
                        ensure_ascii=False,
                    ),
                )
            else:
                print("FAIL:", r.get("error"))
            return r

    results = await asyncio.gather(*[asyncio.create_task(_run_one(vp)) for vp in videos])
    ok = sum(1 for r in results if r.get("success"))
    fail = len(results) - ok
    print(f"\nSummary: ok={ok}, fail={fail}, total={len(results)}")

    # Quick sanity checks for label purity (legacy issues we saw in the old index).
    # This helps verify prompt-level normalization rules are taking effect.
    def _count_bad() -> Dict[str, int]:
        bad = {
            "shot_style_is_shot_type": 0,  # e.g. shot_style == "特写"
            "weather_is_time": 0,  # e.g. weather == "白天"
            "car_color_has_色_suffix": 0,  # e.g. 黑色/白色/蓝色/银色
            "video_usage_non_enum_synonym": 0,  # brand/rights variants that should be normalized
        }
        shot_type_set = set(index_v2_enums.SHOT_TYPE_CHOICES)
        time_set = set(index_v2_enums.TIME_CHOICES)
        bad_usage_aliases = {"品牌传达", "品牌形象传达", "权益说明", "路跑场景展示"}

        for r in results:
            if not r.get("success"):
                continue
            for sr in (r.get("scene_results") or []):
                data = (sr or {}).get("result") if isinstance(sr, dict) else None
                if not isinstance(data, dict):
                    continue

                shot_style = str(data.get("shot_style") or "")
                if shot_style and shot_style in shot_type_set:
                    bad["shot_style_is_shot_type"] += 1

                weather = str(data.get("weather") or "")
                if weather and weather in time_set:
                    bad["weather_is_time"] += 1

                car_color = str(data.get("car_color") or "")
                if car_color.endswith("色"):
                    bad["car_color_has_色_suffix"] += 1

                vu = data.get("video_usage") or []
                if isinstance(vu, str):
                    vu = [vu]
                if isinstance(vu, list) and any((str(x) in bad_usage_aliases) for x in vu):
                    bad["video_usage_non_enum_synonym"] += 1

        return bad

    bad_counts = _count_bad()
    print("\nLabel purity checks:", bad_counts)

    # Persist failure reasons for quick triage.
    failures: List[Dict[str, Any]] = []
    fail_reason_counter: Dict[str, int] = {}
    for r in results:
        if r.get("success"):
            continue
        err = str(r.get("error") or "")
        reason = err.splitlines()[0].strip() if err else "unknown_error"
        failures.append(
            {
                "video": r.get("video"),
                "video_id": r.get("video_id"),
                "error": r.get("error"),
                "failure_response": r.get("failure_response"),
                # keep extra debug fields if present
                "missing": r.get("missing"),
                "workspace_dir": r.get("workspace_dir"),
            }
        )
        fail_reason_counter[reason] = int(fail_reason_counter.get(reason, 0)) + 1

    if failures:
        failures_path = out_dir / "run_video_analysis_v2_failures.json"
        failures_summary_path = out_dir / "run_video_analysis_v2_failures_summary.json"
        failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        failures_summary_path.write_text(
            json.dumps(
                {
                    "total": len(results),
                    "ok": ok,
                    "fail": fail,
                    "top_reasons": sorted(
                        [{"reason": k, "count": v} for k, v in fail_reason_counter.items()],
                        key=lambda x: x["count"],
                        reverse=True,
                    )[:50],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nFailures written: {failures_path.resolve()}")
        print(f"Failure summary: {failures_summary_path.resolve()}")

    # Write a CSV snapshot for quick inspection (exclude any vector fields).
    def _flat(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, list):
            return " | ".join([str(x) for x in v if x is not None])
        return str(v)

    def _is_vector_key(k: str) -> bool:
        kk = (k or "").lower()
        return "vector" in kk or kk.endswith("_vec") or kk.endswith("_embedding")

    # Use IndexV2 doc fields as the canonical CSV columns (exclude vector fields).
    index_cols = [
        k
        for k in CarInteriorAnalysisV2.model_fields.keys()
        if k and (not _is_vector_key(k))
    ]

    csv_rows: List[Dict[str, Any]] = []
    base_cols = ["file_name", "video_path", "obs_video_url", "video_id", "scene_id", "obs_frames"]
    csv_cols = base_cols + index_cols

    for r in results:
        if not r.get("success"):
            continue
        vp = r.get("video") or ""
        obs_video_url = r.get("obs_video_url") or ""
        video_id = r.get("video_id") or ""
        for sr in (r.get("scene_results") or []):
            if not isinstance(sr, dict) or not sr.get("success"):
                continue
            data = sr.get("result") or {}
            frames = sr.get("frame_urls") or []
            row: Dict[str, Any] = {
                "video_path": vp,
                "obs_video_url": obs_video_url,
                "video_id": video_id,
                "scene_id": sr.get("scene_id"),
                # CSV: separate URLs by comma (Excel-friendly)
                "obs_frames": ",".join([str(x) for x in frames if x]),
            }
            if isinstance(data, dict):
                for k in index_cols:
                    row[k] = _flat(data.get(k))
            csv_rows.append(row)

    csv_path = out_dir / "run_video_analysis_v2.csv"
    try:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
            w.writeheader()
            for row in csv_rows:
                w.writerow(row)
        print(f"\nCSV written: {csv_path.resolve()}")
    except Exception as e:
        print("CSV write failed:", str(e))

    try:
        # Ingest into OpenSearch index_v2 (optional).
        if ENABLE_INGEST:
            try:
                print("\n=== Ingesting to OpenSearch (car_interior_analysis_v2) ...")
                # Ensure index exists (no-op if already exists; overwrite handled elsewhere).
                await index_manager.create_index(model_class=CarInteriorAnalysisV2, overwrite=False)

                emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                docs = []
                for r in results:
                    if not r.get("success"):
                        continue
                    for sr in (r.get("scene_results") or []):
                        if not isinstance(sr, dict) or not sr.get("success"):
                            continue
                        data = sr.get("result") or {}
                        if isinstance(data, dict):
                            data = _normalize_model_output(data)
                        if not isinstance(data, dict):
                            continue
                        # id is injected as {video_id}_scene_{sid:03d} already
                        doc = CarInteriorAnalysisV2.from_analysis_result(data, emb)
                        docs.append(doc)

                resp = await bulk_index(CarInteriorAnalysisV2, docs, refresh=True)
                # Print a small, useful summary for debugging dashboard visibility.
                r = resp.get("response") or {}
                items = r.get("items") or []
                errors = bool(r.get("errors"))
                took = r.get("took")
                print(
                    "Ingest done:",
                    {"success": resp.get("success"), "errors": errors, "took": took, "items": len(items)},
                )
                if errors and items:
                    # show first failure reason if any
                    for it in items:
                        action = (it.get("index") or it.get("create") or it.get("update") or {})
                        if action.get("error"):
                            print("First bulk error:", action.get("error"))
                            break
            finally:
                await opensearch_connector.close()
    finally:
        # Avoid "Event loop is closed" warnings from aiomysql connection __del__.
        try:
            await mysql_connector.close()
        except Exception:
            pass

    print(f"\nDone. Results written to: {out_dir.resolve()}")


if __name__ == "__main__":
    # sample = build_test_set()
    # print("Sampled videos:")
    # for v in sample:
    #     print(v)
    asyncio.run(main())
