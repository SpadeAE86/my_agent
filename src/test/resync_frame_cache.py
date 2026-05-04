# -*- coding: utf-8 -*-
"""
resync_frame_cache.py
=====================

两步 resync：

Step 1  —  帧缓存回填 (resync_frames)
    读取 workspace/analysis_cache_v2.json，把每条记录里的 per-scene frame_urls
    重新写入 `video_analysis_scene_frames` 表（改名前叫 _v2）。

    触发场景：
      - `video_analysis_scene_frames_v2` 表改名为 `video_analysis_scene_frames` 后，
        SQLModel 创建了空表，旧数据留在旧表里，导致联表查帧 URL 返回空 frame_urls。
      - 直接从本地 analysis_cache_v2.json 读取并回填，无需重新上传/抽帧。

Step 2  —  os_index_status 状态同步 (resync_os_index_status)
    扫描 OpenSearch 中已存在的文档 id，把 MySQL video_analysis_shot_cards_v2
    中对应行的 os_index_status 更新为 'OK'。

    触发场景：
      - run_video_analysis_v2.py 测试脚本直接写 OpenSearch，但不会调
        update_cards_index_status()，导致 MySQL 一直停在 PENDING，前端一直显示「待入库」。

Run:
    python -m src.test.resync_frame_cache
    python -m src.test.resync_frame_cache --dry-run          # 只打印，不写库
    python -m src.test.resync_frame_cache --skip-frames      # 只同步 os_index_status
    python -m src.test.resync_frame_cache --skip-status      # 只回填帧缓存

Options (env vars):
    CACHE_FILE=...   默认 <script_dir>/workspace/analysis_cache_v2.json
    OS_INDEX=...     默认 car_interior_analysis_v2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ── path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from infra.logging.logger import logger as log
from infra.storage.mysql_connector import mysql_connector
from infra.storage.opensearch_connector import opensearch_connector
from models.sqlmodel.video_analysis import VideoAnalysisSceneFrames, VideoAnalysisShotCardV2
from sqlmodel import select, delete


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: frame cache resync
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache(cache_path: Path) -> Dict[str, Any]:
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


def _extract_scene_frames(entry: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Returns {scene_id: [frame_url, ...]} from one cache entry.
    Falls back to grouping the flat `frames` list by path prefix when
    `scene_results` is absent.
    """
    by_scene: Dict[int, List[str]] = {}

    scene_results = entry.get("scene_results") or []
    for sr in scene_results:
        sid = int(sr.get("scene_id") or 0)
        if sid <= 0:
            continue
        urls = [u for u in (sr.get("frame_urls") or []) if u]
        if urls:
            by_scene[sid] = urls

    # fallback: parse scene_id from URL path segment
    # URL pattern: .../video_analysis_frames/{video_key}/{scene_id}/...
    if not by_scene:
        for url in entry.get("frames") or []:
            parts = url.split("/")
            try:
                # find segment index that matches video_key
                vkey = str(entry.get("video_key") or entry.get("video_id") or "")
                idx = parts.index(vkey)
                sid = int(parts[idx + 1])
                by_scene.setdefault(sid, []).append(url)
            except (ValueError, IndexError):
                pass

    return by_scene


async def resync_frames(cache_path: Path, *, dry_run: bool = False) -> None:
    """
    Re-insert per-scene frame_paths from analysis_cache_v2.json into
    `video_analysis_scene_frames`.
    """
    log.info("[resync_frames] loading cache: %s", cache_path)
    cache = _load_cache(cache_path)

    total_entries = sum(1 for e in cache.values() if e.get("success"))
    log.info("[resync_frames] cache entries (success=True): %d / %d", total_entries, len(cache))

    inserted = skipped = errors = 0

    async with mysql_connector.session_scope() as session:
        for sha1, entry in cache.items():
            if not entry.get("success"):
                continue

            db_video_id: Optional[int] = entry.get("db_video_id")
            video_key = str(entry.get("video_key") or entry.get("video_id") or "")

            if not db_video_id:
                log.warning("[resync_frames] skip %s: no db_video_id", sha1[:8])
                skipped += 1
                continue

            by_scene = _extract_scene_frames(entry)
            if not by_scene:
                log.warning("[resync_frames] skip video_id=%s: no scene frames", db_video_id)
                skipped += 1
                continue

            log.info(
                "[resync_frames] video_id=%s  video_key=%s  scenes=%d",
                db_video_id, video_key, len(by_scene),
            )

            if dry_run:
                for sid, urls in sorted(by_scene.items()):
                    log.info("  [DRY] scene_%03d  %d frames", sid, len(urls))
                continue

            try:
                # Replace all scene-frame rows for this video
                await session.execute(
                    delete(VideoAnalysisSceneFrames).where(
                        VideoAnalysisSceneFrames.video_id == db_video_id
                    )
                )
                for sid, urls in sorted(by_scene.items()):
                    session.add(
                        VideoAnalysisSceneFrames(
                            video_id=int(db_video_id),
                            scene_id=sid,
                            frame_paths=urls,
                        )
                    )
                await session.commit()
                inserted += len(by_scene)
                log.info("  → inserted %d scene rows for video_id=%s", len(by_scene), db_video_id)
            except Exception as exc:
                await session.rollback()
                log.error("  ERROR video_id=%s: %s", db_video_id, exc)
                errors += 1

    log.info(
        "[resync_frames] done. inserted_scene_rows=%d  skipped=%d  errors=%d  dry_run=%s",
        inserted, skipped, errors, dry_run,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: os_index_status sync
# ─────────────────────────────────────────────────────────────────────────────

def _parse_video_id_and_scene(doc_id: str) -> Optional[Tuple[int, int]]:
    """
    doc_id format for v2: "{video_id}_scene_{scene_id:03d}"
    e.g. "64_scene_001" → (64, 1)
    """
    marker = "_scene_"
    if marker not in doc_id:
        return None
    try:
        prefix, suffix = doc_id.split(marker, 1)
        return int(prefix), int(suffix)
    except ValueError:
        return None


async def _fetch_all_doc_ids(client: Any, index: str, page_size: int = 500) -> List[str]:
    """Paginate through OpenSearch with search_after to get all _ids."""
    out: List[str] = []
    search_after = None
    while True:
        body: Dict[str, Any] = {
            "size": page_size,
            "_source": False,
            "query": {"match_all": {}},
            "sort": [{"_id": "asc"}],
        }
        if search_after is not None:
            body["search_after"] = search_after

        resp = await client.search(index=index, body=body)
        hits = (((resp or {}).get("hits") or {}).get("hits") or [])
        if not hits:
            break
        for h in hits:
            doc_id = str(h.get("_id") or "")
            if doc_id:
                out.append(doc_id)
        last_sort = hits[-1].get("sort")
        if not isinstance(last_sort, list) or not last_sort:
            break
        search_after = last_sort
    return out


async def resync_os_index_status(
    os_index: str,
    *,
    dry_run: bool = False,
) -> None:
    """
    Scan OpenSearch for all indexed doc_ids, then update MySQL
    video_analysis_shot_cards_v2.os_index_status = 'OK' for matching rows.

    Only rows currently in PENDING or FAILED state are touched.
    """
    log.info("[resync_os_status] fetching doc ids from index: %s", os_index)

    await opensearch_connector.ensure_init()
    client = await opensearch_connector.get_client()

    doc_ids = await _fetch_all_doc_ids(client, os_index)
    log.info("[resync_os_status] found %d docs in OpenSearch", len(doc_ids))

    # Parse (video_id, scene_id) pairs
    pairs: List[Tuple[int, int]] = []
    skip_parse = 0
    for did in doc_ids:
        p = _parse_video_id_and_scene(did)
        if p:
            pairs.append(p)
        else:
            skip_parse += 1

    log.info(
        "[resync_os_status] parsed %d (video_id, scene_id) pairs, skipped %d unparseable",
        len(pairs), skip_parse,
    )

    if not pairs:
        log.warning("[resync_os_status] nothing to update — no parseable pairs")
        return

    # Group by video_id for batched updates
    by_video: Dict[int, List[int]] = defaultdict(list)
    for vid, sid in pairs:
        by_video[vid].append(sid)

    updated_total = 0

    async with mysql_connector.session_scope() as session:
        for video_id, scene_ids in by_video.items():
            log.info(
                "[resync_os_status] video_id=%s  %d scenes in OpenSearch",
                video_id, len(scene_ids),
            )

            if dry_run:
                log.info("  [DRY] would update %d rows to OK for video_id=%s", len(scene_ids), video_id)
                updated_total += len(scene_ids)
                continue

            try:
                result = await session.execute(
                    select(VideoAnalysisShotCardV2).where(
                        VideoAnalysisShotCardV2.video_id == video_id,
                        VideoAnalysisShotCardV2.scene_id.in_(scene_ids),
                        VideoAnalysisShotCardV2.os_index_status != "OK",
                    )
                )
                rows = result.scalars().all()
                for row in rows:
                    row.os_index_status = "OK"
                    row.os_index_error = None
                    session.add(row)
                await session.commit()
                if rows:
                    log.info("  → updated %d rows to OK for video_id=%s", len(rows), video_id)
                    updated_total += len(rows)
            except Exception as exc:
                await session.rollback()
                log.error("  ERROR video_id=%s: %s", video_id, exc)

    log.info(
        "[resync_os_status] done. updated=%d  dry_run=%s",
        updated_total, dry_run,
    )


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

async def main(
    *,
    dry_run: bool,
    skip_frames: bool,
    skip_status: bool,
) -> None:
    cache_file = Path(os.getenv("CACHE_FILE", str(SCRIPT_DIR / "workspace" / "analysis_cache_v2.json")))
    os_index = os.getenv("OS_INDEX", "car_interior_analysis_v2")

    await mysql_connector.ensure_init()

    if not skip_frames:
        if not cache_file.exists():
            log.error("[resync] cache file not found: %s", cache_file)
        else:
            await resync_frames(cache_file, dry_run=dry_run)

    if not skip_status:
        await resync_os_index_status(os_index, dry_run=dry_run)

    await opensearch_connector.close()
    await mysql_connector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resync frame cache and/or os_index_status")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done, don't write")
    parser.add_argument("--skip-frames", action="store_true", help="Skip frame cache resync (Step 1)")
    parser.add_argument("--skip-status", action="store_true", help="Skip os_index_status sync (Step 2)")
    args = parser.parse_args()

    asyncio.run(main(
        dry_run=args.dry_run,
        skip_frames=args.skip_frames,
        skip_status=args.skip_status,
    ))
