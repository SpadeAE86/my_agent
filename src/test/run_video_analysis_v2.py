"""
Test runner: analyze selected videos with SceneAnalysisResultV2 schema.

- Skips scene splitting (single segment)
- Extracts key frames
- Uploads frames to OBS
- Calls Doubao vision with JSON schema = SceneAnalysisResultV2
- Writes outputs to a local JSONL file for inspection

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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure we can import from src/ when running as a module or directly.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models.pydantic.model_output_schema.video_analysis_schema import SceneAnalysisResultV2
from utils.video_process_utils import get_video_single_scene_frames
from utils.obs_utils import batch_upload_to_obs
from utils.call_model_utils import call_doubao_vision
from PIL import Image
from services.video_analysis_db_service import video_analysis_db_service
from models.pydantic.opensearch_index.index_v2_enums import KEY_TRAITS_CHOICES
# --- Optional ingestion (OpenSearch IndexV2) ---
from sentence_transformers import SentenceTransformer
from infra.storage.opensearch.create_index import index_manager
from infra.storage.opensearch.document_writer import bulk_index
from infra.storage.opensearch_connector import opensearch_connector
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2


BASE_VIDEO_DIR = Path(r"C:\Users\25065\Downloads\汽车\ls6_video\LS6视频")

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

# Ingest into OpenSearch after analysis (can turn off quickly).
ENABLE_INGEST = True


PROMPT_V2 = f"""
你是一个专业的视频分镜分析师，擅长把“可检索的结构化标签”从画面中抽取出来，支持后续营销脚本混剪检索。

请仅根据画面可见信息输出 JSON（必须符合给定 schema），不要输出解释。

关键要求：
- movement：只写“核心动作”（单值），必须标准化，不带环境词、不带评价。例：掉头/转弯/泊车/充电/静态展示
- footage_type：画面类型（固定枚举），如：CG/原创实拍/KOL拍摄/TVC切片/直播切片/海报/未知
- shot_style：镜头风格/拍摄方式（固定枚举），如：车内POV/车外跟拍/固定机位/航拍/屏幕录制/展台转盘/未知
- shot_type：镜头景幅/景别（固定枚举）：大远景/远景/中景/特写/大特写/未知
- scene_location：画面场景/路况/空间类型（1-6 个），短词名词化，如：地库/公路/冰雪/现代城区/赛道/展厅 等
- car_color + car_color_detail：车色（枚举）+ 可选细节（如哑光黑/珠光白/涂装/贴膜）
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
- key_traits：客户要求的额外标签（枚举列表，可多值），没有看到对应的要素就不要填，只能从给定的枚举范围里选：{",".join(KEY_TRAITS_CHOICES)}

关于key_trait的特殊标签的额外说明:
    看到带人的内饰，人开车，人谈话、休息，可以打上安静和降噪的标签
    看到车的音响和喇叭，可以打上声道和音响的标签
    看到空调出风口，屏幕上有空调的标志，可以打上空调的标签
    看到轮胎转弯，可以打上转弯的标签
    有人坐后排，可以打上后排的标签
    看到路跑，充电，可以打上续航，低油耗，大电池，充电快的标签

禁止：
- 不要编造画面看不到的具体数值参数（如续航km、电池kWh等）

一致性提示（用于避免语义涣散）：
- design_* 只写“看得见/摸得着”的实体与外观：如 轮毂/车漆/门把手/座椅/中控台/屏幕/灯组/线条/材质
- function_* 只写“能力/功能/算法/性能”：如 一键AI泊车/雨夜模式/NOA/爆胎稳定控制/四轮转向/快充/主动降噪
- 同一个词不要同时出现在 design_* 与 function_*（必要时放到更匹配的一侧）
""".strip()


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

def _safe_stem(name: str) -> str:
    """
    Make a filesystem-friendly stem for workspace folder names.
    Keep ascii letters/numbers and common separators; replace others with '_'.
    """
    s = "".join([c if (c.isalnum() or c in ("-", "_", ".", " ")) else "_" for c in (name or "")]).strip()
    return s or "video"

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


async def analyze_one(video_path: str, *, frame_interval: float) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        return {"video": video_path, "success": False, "error": "file not found"}

    # Cache: if we've already analyzed this exact video under the same spec, reuse result.
    cache_key = _analysis_cache_key(video_path, frame_interval=frame_interval)
    cache = _load_json(ANALYSIS_CACHE_PATH, default={})
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and cached.get("success") and isinstance(cached.get("result"), dict):
        return {
            "video": video_path,
            "run_id": cached.get("run_id") or f"v2_cache_{_now_id()}_{_short_hash(video_path)}",
            "frames": cached.get("frames") or [],
            "result": cached.get("result"),
            "success": True,
            "cached": True,
        }

    vid = Path(video_path)
    # IMPORTANT (Windows/OpenCV): avoid non-ascii paths for cv2.imwrite.
    # Some OpenCV builds fail to write images to Unicode paths.
    stem_hash = _short_hash(vid.stem)
    # Use this run_id as DB history_id, and as the OpenSearch id prefix.
    run_id = f"v2_{_now_id()}_{stem_hash}"
    # Use a stable *ASCII-only* workspace folder per input filename hash so re-runs can reuse extracted frames
    # without hitting OpenCV Unicode-path issues on Windows.
    workspace_dir = (Path(__file__).resolve().parent / "workspace" / f"{stem_hash}")
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Insert history row early so we can trace OpenSearch doc ids back to DB.
    try:
        await video_analysis_db_service.upsert_history_item(
            {
                "id": run_id,
                "name": vid.name,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "video_url": video_path,
                "cards": [],
            }
        )
    except Exception:
        # Non-blocking: analysis can continue even if DB is unavailable.
        pass

    # If workspace already has extracted frames, reuse them and skip detection/extraction.
    local_frames = _existing_frames(workspace_dir)
    video_duration = 0.0
    if not local_frames:
        # Extract frames (single segment, no scene split)
        if _extract_sem is None:
            scenes = await asyncio.to_thread(
                get_video_single_scene_frames, video_path, frame_interval, str(workspace_dir)
            )
        else:
            async with _extract_sem:
                scenes = await asyncio.to_thread(
                    get_video_single_scene_frames, video_path, frame_interval, str(workspace_dir)
                )
        if not scenes:
            return {"video": video_path, "success": False, "error": "no frames extracted"}

        scene = scenes[0]
        local_frames = scene.frame_url_list or []
        try:
            video_duration = float(scene.duration_seconds or 0.0)
        except Exception:
            video_duration = 0.0
        if not local_frames:
            return {"video": video_path, "success": False, "error": "empty frame list"}
    else:
        # Workspace cache hit: we still try to compute duration cheaply.
        try:
            scenes = await asyncio.to_thread(get_video_single_scene_frames, video_path, frame_interval, str(workspace_dir))
            if scenes:
                video_duration = float(scenes[0].duration_seconds or 0.0)
        except Exception:
            video_duration = 0.0

    # Validate local files exist before uploading.
    missing = [p for p in local_frames if not os.path.exists(p)]
    if missing:
        return {
            "video": video_path,
            "run_id": run_id,
            "success": False,
            "error": f"missing {len(missing)} extracted frames",
            "missing": missing[:5],
            "workspace_dir": str(workspace_dir),
        }

    # Compute resolution/aspect from extracted frames (deterministic metadata).
    wh = _get_first_frame_size(local_frames)
    frame_w, frame_h = wh if wh else (0, 0)
    resolution = f"{frame_w}x{frame_h}" if frame_w and frame_h else "未知"
    frame_size = _aspect_label(frame_w, frame_h) if frame_w and frame_h else "未知"

    # Upload to OBS (frames only)
    obs_key_prefix = f"ai_picture/video_analysis_v2/{run_id}"
    frame_urls = await batch_upload_to_obs(
        file_paths=local_frames,
        obs_key_prefix=obs_key_prefix,
        max_concurrency=50,
    )

    # Call Doubao with V2 schema
    schema_json = SceneAnalysisResultV2.model_json_schema()
    raw = await call_doubao_vision(PROMPT_V2, frame_urls, schema_json)
    data = raw if isinstance(raw, dict) else json.loads(raw)

    # Inject deterministic fields that users provide or we compute (not asked from the model).
    # - car_model: user-provided later (keep "未知" here)
    # - frame_size/resolution: computed from frames
    if isinstance(data, dict):
        # Use DB history id as OpenSearch id prefix for later DB backtrace.
        data["id"] = f"{run_id}_scene_001"
        data.setdefault("car_model", "未知")
        data["frame_size"] = frame_size
        data["resolution"] = resolution
        data["video_duration"] = float(video_duration or 0.0)

    # Store one shot card into DB (single-scene mode)
    try:
        await video_analysis_db_service.upsert_history_item(
            {
                "id": run_id,
                "name": vid.name,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "video_url": video_path,
                "cards": [
                    {
                        "scene_id": 1,
                        "start_time": 0.0,
                        "end_time": float(video_duration or 0.0),
                        "duration_seconds": float(video_duration or 0.0),
                        "thumbnail": frame_urls[0] if frame_urls else None,
                        "frame_urls": frame_urls,
                        # Keep a subset of analysis fields for quick UI inspection
                        "description": (data or {}).get("description") if isinstance(data, dict) else None,
                        "subject": (data or {}).get("subject") if isinstance(data, dict) else None,
                        "object": (data or {}).get("object") if isinstance(data, dict) else None,
                        "movement": (data or {}).get("movement") if isinstance(data, dict) else None,
                        "adjective": (data or {}).get("adjective") if isinstance(data, dict) else None,
                        "search_tags": (data or {}).get("search_tags") if isinstance(data, dict) else None,
                        "marketing_tags": (data or {}).get("marketing_tags") if isinstance(data, dict) else None,
                        "appealing_audience": (data or {}).get("appealing_audience") if isinstance(data, dict) else None,
                        "visual_quality": (data or {}).get("visual_quality") if isinstance(data, dict) else None,
                    }
                ],
            }
        )
    except Exception:
        pass

    out = {
        "video": video_path,
        "run_id": run_id,
        "frames": frame_urls,
        "result": data,
        "success": True,
    }

    # Persist to cache (best-effort; keep it simple and robust).
    try:
        global _cache_lock
        if _cache_lock is None:
            _cache_lock = asyncio.Lock()
        async with _cache_lock:
            cache = _load_json(ANALYSIS_CACHE_PATH, default={})
            cache[cache_key] = {
                "success": True,
                "run_id": run_id,
                "frames": frame_urls,
                "result": data,
                "video": video_path,
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
    # Analyze ALL videos under BASE_VIDEO_DIR (no sampling)
    videos = [str(p) for p in _list_all_mp4(BASE_VIDEO_DIR)]
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
                r = {"video": vp, "success": False, "error": str(e)}

            _out_file_for(vp).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

            if r.get("success"):
                print("OK:", json.dumps(r.get("result"), ensure_ascii=False))
            else:
                print("FAIL:", r.get("error"))
            return r

    results = await asyncio.gather(*[asyncio.create_task(_run_one(vp)) for vp in videos])
    ok = sum(1 for r in results if r.get("success"))
    fail = len(results) - ok
    print(f"\nSummary: ok={ok}, fail={fail}, total={len(results)}")

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
                "run_id": r.get("run_id"),
                "error": r.get("error"),
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
    base_cols = ["file_name", "video_path", "run_id", "obs_frames"]
    csv_cols = base_cols + index_cols

    for r in results:
        if not r.get("success"):
            continue
        vp = r.get("video") or ""
        data = r.get("result") or {}
        frames = r.get("frames") or []
        row: Dict[str, Any] = {
            "video_path": vp,
            "run_id": r.get("run_id") or "",
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
                data = r.get("result") or {}
                if not isinstance(data, dict):
                    continue
                # Use run_id as doc id to avoid collisions in quick testing.
                data.setdefault("id", r.get("run_id"))
                doc = CarInteriorAnalysisV2.from_analysis_result(data, emb)
                docs.append(doc)

            resp = await bulk_index(CarInteriorAnalysisV2, docs, refresh=True)
            # Print a small, useful summary for debugging dashboard visibility.
            r = resp.get("response") or {}
            items = r.get("items") or []
            errors = bool(r.get("errors"))
            took = r.get("took")
            print("Ingest done:", {"success": resp.get("success"), "errors": errors, "took": took, "items": len(items)})
            if errors and items:
                # show first failure reason if any
                for it in items:
                    action = (it.get("index") or it.get("create") or it.get("update") or {})
                    if action.get("error"):
                        print("First bulk error:", action.get("error"))
                        break
        finally:
            await opensearch_connector.close()

    print(f"\nDone. Results written to: {out_dir.resolve()}")


if __name__ == "__main__":
    # sample = build_test_set()
    # print("Sampled videos:")
    # for v in sample:
    #     print(v)
    asyncio.run(main())
