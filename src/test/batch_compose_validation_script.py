"""
Batch compose runner.

Reads:
- data/transcribe_validation/*.json
- data/match_result/*.json

Creates a temporary video_match_job / shot rows, then calls the existing
video mix compose service so the final output is produced by the normal
service path.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIBE_DIR = PROJECT_ROOT / "data" / "transcribe_validation"
MATCH_DIR = PROJECT_ROOT / "data" / "match_result"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output_video"
TMP_VIDEO_DIR = PROJECT_ROOT / "data" / "tmp_mix_jobs"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_name(name: str, index: int) -> str:
    text = str(name or "").strip()
    if not text:
        return f"{index:03d}"
    return f"{index:03d}_{''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in text)[:80]}"


def _coerce_duration(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _normalize_string_list(values: Any, expected_len: int) -> List[str]:
    out: List[str] = []
    if isinstance(values, list):
        for value in values[:expected_len]:
            if value is None:
                out.append("")
            else:
                out.append(str(value).strip())
    while len(out) < expected_len:
        out.append("")
    return out


def _first_video_url(seg: Dict[str, Any]) -> str:
    if not isinstance(seg, dict):
        return ""
    top1 = str(seg.get("top1_video_path") or "").strip()
    if top1:
        return top1
    top5 = seg.get("top5") or []
    if isinstance(top5, list):
        for h in top5:
            if not isinstance(h, dict):
                continue
            u = str(h.get("video_path") or "").strip()
            if u:
                return u
    return ""


def _segments_from_match(match_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    segs = match_data.get("segments") if isinstance(match_data, dict) else []
    if not isinstance(segs, list):
        return []
    return [s for s in segs if isinstance(s, dict)]


async def _create_temp_job(source: Dict[str, Any], transcribe: Dict[str, Any], match: Dict[str, Any], index: int) -> str:
    from infra.storage.mysql_connector import mysql_connector
    from models.sqlmodel.video_match import VideoMatchJob, VideoMatchShotRow

    job_id = str(uuid.uuid4())
    title = str((source or {}).get("title") or (source or {}).get("topic") or f"job-{index}").strip()
    topic = str((source or {}).get("topic") or "").strip()
    car_model = str((source or {}).get("car_model") or "").strip()
    stage1 = transcribe.get("stage1") if isinstance(transcribe, dict) else {}
    audio_urls = transcribe.get("audio_urls") if isinstance(transcribe, dict) else []
    durations = transcribe.get("segment_durations") if isinstance(transcribe, dict) else []
    segments = _segments_from_match(match)
    audio_urls = _normalize_string_list(audio_urls, len(segments))
    if not isinstance(durations, list):
        durations = []

    async with mysql_connector.session_scope() as session:
        job = VideoMatchJob(
            id=job_id,
            workspace="v2",
            script=str((stage1 or {}).get("storyboard") or ""),
            topic=topic or None,
            title=title or None,
            car_model=car_model or None,
            frame_size="9:16",
            frame_orientation="竖屏",
            parse_status="done",
            extract_status="done",
            search_status="pending",
        )
        session.add(job)
        for order, seg in enumerate(segments):
            top1 = _first_video_url(seg)
            dur = _coerce_duration(seg.get("duration"))
            if dur <= 0 and order < len(durations):
                dur = _coerce_duration(durations[order])
            if dur <= 0:
                dur = 2.0
            tags_json = dict(seg)
            tags_json["duration"] = dur
            audio_url = audio_urls[order] if order < len(audio_urls) else ""
            row = VideoMatchShotRow(
                job_id=job_id,
                shot_order=order,
                storyboard_id=int(seg.get("segment_id") or (order + 1)),
                segment_text=str(seg.get("segment_text") or ""),
                duration_sec=dur,
                description=str(seg.get("description") or ""),
                tags_json=tags_json,
                extract_status="done",
                search_status="done",
                top1_obs_url=top1,
                obs_audio_url=str(audio_url or "").strip(),
            )
            session.add(row)
        await session.commit()
    return job_id


async def _copy_mock_output_to_file(url: str, dest: Path) -> None:
    from utils.obs_utils import download_url_to_file

    dest.parent.mkdir(parents=True, exist_ok=True)
    await download_url_to_file(url, str(dest))


async def _compose_one(match_path: Path, transcribe_path: Path, index: int, total: int) -> Dict[str, Any]:
    from infra.storage.mysql_connector import mysql_connector
    from services.video_compose_services import get_mix_compose_job, start_mix_compose_for_job

    match_data = _load_json(match_path)
    transcribe_data = _load_json(transcribe_path)
    source = transcribe_data.get("source") if isinstance(transcribe_data, dict) else {}
    if not isinstance(source, dict):
        source = {}

    print(f"[{index}/{total}] start {match_path.name}")
    job_id = await _create_temp_job(source, transcribe_data, match_data, index)
    try:
        compose = await start_mix_compose_for_job(job_id, mix_mock=True, prefer_srt=False)
        compose_id = str(compose.get("compose_id") or "").strip()
        if not compose_id:
            raise RuntimeError("compose_id missing")

        status = None
        result_url = None
        for _ in range(120):
            detail = await get_mix_compose_job(compose_id)
            if detail:
                status = detail.get("status")
                result_url = detail.get("result_obs_url")
                if status == "done" and result_url:
                    break
                if status == "failed":
                    raise RuntimeError(str(detail.get("error_message") or "compose failed"))
            await asyncio.sleep(0.5)

        if not result_url:
            raise RuntimeError("compose result url missing")

        out_name = f"{_safe_name(str(source.get('title') or source.get('topic') or match_path.stem), index)}.mp4"
        out_path = OUTPUT_DIR / out_name
        await _copy_mock_output_to_file(str(result_url), out_path)
        print(f"[{index}/{total}] success -> {out_path.name}")
        return {"success": True, "path": str(out_path), "compose_id": compose_id}
    finally:
        try:
            async with mysql_connector.session_scope() as session:
                from sqlalchemy import delete
                from models.sqlmodel.video_match import VideoMatchJob, VideoMatchShotRow

                await session.execute(delete(VideoMatchShotRow).where(VideoMatchShotRow.job_id == job_id))
                await session.execute(delete(VideoMatchJob).where(VideoMatchJob.id == job_id))
                await session.commit()
        except Exception:
            pass


async def main() -> None:
    match_files = {p.name: p for p in sorted([p for p in MATCH_DIR.glob("*.json") if p.is_file()])}
    transcribe_files = {p.name: p for p in sorted([p for p in TRANSCRIBE_DIR.glob("*.json") if p.is_file()])}
    paired_files = [(match_files[name], transcribe_files[name]) for name in sorted(match_files.keys() & transcribe_files.keys())][:10]
    total = len(paired_files)
    if total == 0:
        print("No match/transcribe json files found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (match_path, transcribe_path) in enumerate(paired_files, start=1):
        results.append(await _compose_one(match_path, transcribe_path, i, total))

    ok = sum(1 for r in results if r.get("success"))
    fail = len(results) - ok
    print(f"\nSummary: ok={ok}, fail={fail}, total={len(results)}")
    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
