"""
Batch runner for v2 video analysis.

Edit the JSON input list only. The runner will:
- read URLs from the JSON files
- infer car model from the OBS path (L6 / LS9)
- upsert `video_source_upload_cache`
- call the existing service pipeline
- keep logs short for long-running batches
"""

from __future__ import annotations

import asyncio
import logging
import json
import os
import zipfile
import shutil
import sys
import time
import contextlib
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

os.environ.setdefault("AI_BATCH_RUNNER_QUIET", "1")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.logging.logger import logger as log
from infra.storage.sqlmodel_init import create_tables_if_not_exists
from models.pydantic.video_analysis_request import ShotCard
from services.analysis_video import analyze_video, index_shotcards_to_opensearch
from services.video_analysis_db_service import video_analysis_db_service
from services.video_upload_cache_service import video_upload_cache_service
from utils.obs_utils import OBS_BASE_URL, download_from_obs, download_url_to_file


def _quiet_logs() -> None:
    # Keep this batch runner focused on progress output only.
    # Service-level loggers are muted; this file prints its own progress lines.
    try:
        from loguru import logger as loguru_logger

        loguru_logger.remove()
        loguru_logger.add(sys.stderr, level="CRITICAL", enqueue=True)
    except Exception:
        pass
    for name in ("pyscenedetect", "httpx", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)


JSON_INPUT_FILES: List[str] = [
    r"C:\Job\AI\AI_agent_project\my_bot_advance\my_agent\ls9data_out.json",
    r"C:\Job\AI\AI_agent_project\my_bot_advance\my_agent\LS9无关数据.json",
    r"C:\Job\AI\AI_agent_project\my_bot_advance\my_agent\L6无关数据.json",
    r"C:\Job\AI\AI_agent_project\my_bot_advance\my_agent\l6data_out(1).json",
]

WORKSPACE = "v2"
FRAME_INTERVAL = 2.0
THRESHOLD = 30.0
SPLIT_SCENES = True
MAX_CONCURRENCY = 5
DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_DELAY_SEC = 1.5
TMP_DIR = Path(__file__).resolve().parent / "workspace" / "tmp_sources"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".qt", ".webm"}


def _source_name(source: str) -> str:
    raw = str(source or "").strip().replace("\\", "/")
    if not raw:
        return "video.mp4"
    if "://" in raw:
        return Path(urlparse(raw).path).name or "video.mp4"
    return Path(raw).name or "video.mp4"


def _resolve_source_url(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("obs://"):
        value = value[len("obs://") :].lstrip("/")
        if value.startswith("freeuuu/"):
            value = value[len("freeuuu/") :]
        return f"{OBS_BASE_URL}/{value.lstrip('/')}"
    if value.startswith("freeuuu/"):
        return f"{OBS_BASE_URL}/{value}"
    if value.startswith("ai_picture/"):
        return f"{OBS_BASE_URL}/{value}"
    return value


def _extract_car_model_from_url(url: str) -> Optional[str]:
    text = str(url or "")
    upper = text.upper()
    if "/L6/" in upper or "LS6" in upper or "L6" in upper:
        return "L6"
    if "/LS9/" in upper or "LS9" in upper:
        return "LS9"
    return None


def _iter_json_urls(path: Path) -> Iterable[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    yield s
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str):
                        s = v.strip()
                        if s:
                            yield s
                    elif isinstance(v, list):
                        for x in v:
                            if isinstance(x, str) and x.strip():
                                yield x.strip()
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, str):
                s = v.strip()
                if s:
                    yield s
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, str) and x.strip():
                        yield x.strip()
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, str) and vv.strip():
                        yield vv.strip()
                    elif isinstance(vv, list):
                        for x in vv:
                            if isinstance(x, str) and x.strip():
                                yield x.strip()


def _load_sources() -> List[str]:
    out: List[str] = []
    seen = set()
    for file_path in JSON_INPUT_FILES:
        p = Path(file_path)
        if not p.exists():
            print(f"[SKIP] missing json: {p}")
            continue
        for url in _iter_json_urls(p):
            resolved = _resolve_source_url(url)
            if not resolved:
                continue
            key = resolved.strip()
            if key in seen:
                continue
            seen.add(key)
            out.append(resolved)
    return out


def _is_local_file(source: str) -> bool:
    return Path(str(source)).expanduser().exists()


def _find_video_file(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.parts), p.name.lower()))
    return candidates[0]


async def _materialize_source(source: str) -> Tuple[str, List[str], str]:
    """
    Return (local_video_path, cleanup_targets, source_name).
    """
    if _is_local_file(source):
        local_path = str(Path(source).expanduser().resolve())
        path = Path(local_path)
        if path.suffix.lower() != ".zip":
            return local_path, [], _source_name(local_path)

        extract_dir = TMP_DIR / f"{path.stem}_unzipped"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(local_path, "r") as zf:
            zf.extractall(extract_dir)
        video_path = _find_video_file(extract_dir)
        if video_path is None:
            raise RuntimeError(f"zip does not contain a supported video file: {source}")
        return str(video_path), [local_path, str(extract_dir)], _source_name(local_path)

    source_url = _resolve_source_url(source)
    source_name = _source_name(source_url or source)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    local_path = TMP_DIR / f"{os.urandom(6).hex()}_{source_name}"
    last_err: Optional[Exception] = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            await download_url_to_file(source_url, str(local_path))
            last_err = None
            break
        except Exception as e:
            last_err = e
            if attempt < DOWNLOAD_RETRIES:
                await asyncio.sleep(DOWNLOAD_RETRY_DELAY_SEC * attempt)
    if last_err is not None:
        raise last_err

    if local_path.suffix.lower() != ".zip":
        return str(local_path), [str(local_path)], source_name

    extract_dir = TMP_DIR / f"{local_path.stem}_unzipped"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_path, "r") as zf:
        zf.extractall(extract_dir)
    video_path = _find_video_file(extract_dir)
    if video_path is None:
        raise RuntimeError(f"zip does not contain a supported video file: {source_url}")
    return str(video_path), [str(local_path), str(extract_dir)], source_name


async def _upsert_source_cache(*, source_name: str, source_url: str, local_path: str) -> None:
    try:
        st = os.stat(local_path)
        obs_key = source_url.replace(OBS_BASE_URL + "/", "") if source_url.startswith(OBS_BASE_URL) else source_url
        await video_upload_cache_service.upsert(
            file_name=source_name,
            abs_path=os.path.abspath(local_path),
            file_size=int(st.st_size),
            file_mtime=int(st.st_mtime),
            obs_key=obs_key,
            obs_url=source_url,
        )
    except Exception as e:
        log.warning("failed to upsert video_source_upload_cache for %s: %s", source_name, e)


async def _persist_v2_result(
    *,
    video_key: str,
    source_name: str,
    source_url: str,
    car_model: str,
    cards: List[ShotCard],
) -> None:
    scene_frames: Dict[int, List[str]] = {}
    payload_cards: List[Dict[str, Any]] = []
    for card in cards or []:
        if not isinstance(card, ShotCard):
            continue
        scene_frames[int(card.scene_id)] = [str(x) for x in (card.frame_urls or []) if x]
        payload_cards.append(card.model_dump())

    await video_analysis_db_service.upsert_history_item(
        {
            "id": video_key,
            "name": source_name,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_url": source_url,
            "workspace": WORKSPACE,
            "car_model": car_model,
            "cards": payload_cards,
            "scene_frames": scene_frames,
        },
        shot_cards_version="v2",
    )


async def _should_skip_existing_analysis(source_name: str) -> Tuple[bool, str]:
    """
    Reuse the v2 history mapping to skip already analyzed videos on reruns.

    Returns (should_skip, video_key).
    """
    _, video_key = await video_analysis_db_service.resolve_video_v2_for_source_file(file_name=source_name)
    if not video_key:
        return False, ""

    try:
        item = await video_analysis_db_service.get_history_item(video_key, shot_cards_version="v2")
        if isinstance(item, dict):
            status = str(item.get("status") or "").upper()
            cards = item.get("cards") or []
            if status == "SUCCESS" and isinstance(cards, list) and cards:
                return True, video_key
    except Exception as e:
        log.warning("failed to inspect existing analysis history for %s: %s", source_name, e)

    return False, video_key


async def _run_one_source(source: str, index: int, total: int) -> Dict[str, Any]:
    local_path = ""
    cleanup_targets: List[str] = []
    source_name = _source_name(source)
    source_url = _resolve_source_url(source)
    car_model = _extract_car_model_from_url(source_url or source) or "L6"

    try:
        started_at = time.perf_counter()
        should_skip, existing_video_key = await _should_skip_existing_analysis(source_name)
        if should_skip:
            elapsed_sec = time.perf_counter() - started_at
            print(f"[{index}/{total}] skip {source_name} elapsed={elapsed_sec:.1f}s reason=already_analysed")
            return {
                "source": source,
                "source_name": source_name,
                "video_key": existing_video_key,
                "car_model": car_model,
                "success": True,
                "skipped": True,
                "elapsed_sec": round(elapsed_sec, 3),
            }

        local_path, cleanup_targets, source_name = await _materialize_source(source)
        print(f"[{index}/{total}] start {source_name} model={car_model}")
        source_url = source_url or source

        await _upsert_source_cache(
            source_name=source_name,
            source_url=source_url,
            local_path=local_path,
        )

        _, video_key = await video_analysis_db_service.resolve_video_v2_for_source_file(
            file_name=source_name
        )
        if not video_key:
            raise RuntimeError(f"failed to resolve v2 video key for {source_name}")

        print(f"[{index}/{total}] analyze {source_name} ...")
        analyze_started = time.perf_counter()
        cards = await analyze_video(
            local_video_path=local_path,
            project_id=video_key,
            frame_interval=FRAME_INTERVAL,
            threshold=THRESHOLD,
            split_scenes=SPLIT_SCENES,
            cleanup_workspace=True,
            workspace=WORKSPACE,
            car_model=car_model,
        )
        analyze_elapsed = time.perf_counter() - analyze_started
        print(
            f"[{index}/{total}] analyze_done {source_name} cards={len(cards)} "
            f"elapsed={analyze_elapsed:.1f}s"
        )

        persist_started = time.perf_counter()
        await _persist_v2_result(
            video_key=video_key,
            source_name=source_name,
            source_url=source_url,
            car_model=car_model,
            cards=cards,
        )
        persist_elapsed = time.perf_counter() - persist_started
        print(f"[{index}/{total}] persist_done {source_name} elapsed={persist_elapsed:.1f}s")

        index_started = time.perf_counter()
        print(f"[{index}/{total}] index {source_name} ...")
        index_resp = await index_shotcards_to_opensearch(
            cards,
            id_prefix=video_key,
            workspace=WORKSPACE,
        )
        index_elapsed = time.perf_counter() - index_started

        elapsed_sec = time.perf_counter() - started_at

        print(
            f"[{index}/{total}] success {source_name} cards={len(cards)} "
            f"elapsed={elapsed_sec:.1f}s index_elapsed={index_elapsed:.1f}s"
        )
        return {
            "source": source,
            "source_name": source_name,
            "video_key": video_key,
            "car_model": car_model,
            "success": True,
            "cards": len(cards),
            "indexed": index_resp.get("items", 0) if isinstance(index_resp, dict) else 0,
            "elapsed_sec": round(elapsed_sec, 3),
        }
    except Exception as e:
        elapsed_sec = time.perf_counter() - started_at if "started_at" in locals() else 0.0
        print(f"[{index}/{total}] fail {source_name} elapsed={elapsed_sec:.1f}s err={e}")
        log.exception("video analysis failed for %s", source)
        return {
            "source": source,
            "source_name": source_name,
            "car_model": car_model,
            "success": False,
            "error": str(e),
            "elapsed_sec": round(elapsed_sec, 3),
        }
    finally:
        if cleanup_targets:
            try:
                for target in cleanup_targets:
                    path = Path(target)
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                if TMP_DIR.exists() and not any(TMP_DIR.iterdir()):
                    shutil.rmtree(TMP_DIR, ignore_errors=True)
            except Exception as e:
                log.warning("failed to clean temp source %s: %s", local_path, e)


async def main() -> None:
    _quiet_logs()
    await create_tables_if_not_exists()

    sources = _load_sources()
    total = len(sources)
    if not sources:
        print("No sources found in JSON input files.")
        return

    print(f"Loaded {total} sources from {len(JSON_INPUT_FILES)} json files.")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _guarded_run(i: int, source: str) -> Dict[str, Any]:
        async with sem:
            return await _run_one_source(source, i, total)

    results = await asyncio.gather(
        *[asyncio.create_task(_guarded_run(i, source)) for i, source in enumerate(sources, start=1)]
    )

    ok = sum(1 for r in results if r.get("success"))
    skipped = sum(1 for r in results if r.get("skipped"))
    failed_results = [r for r in results if not r.get("success")]
    fail = len(failed_results)
    print(f"\nSummary: ok={ok}, skip={skipped}, fail={fail}, total={total}")
    if failed_results:
        failed_names = ", ".join(str(r.get("source_name") or r.get("source") or "?") for r in failed_results)
        print(f"Failed items ({len(failed_results)}): {failed_names}")


if __name__ == "__main__":
    asyncio.run(main())
