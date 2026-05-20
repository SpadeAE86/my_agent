"""
Batch transcribe runner for `validation_script.csv`.

Thin orchestration only:
- read the first 10 rows from the CSV
- call the existing local rewrite service
- force portrait constraints
- keep stage1 + stage2 outputs together
- keep per-segment audio URLs and durations together
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "validation_script.csv"
OUT_DIR = PROJECT_ROOT / "data" / "transcribe_validation"

OUTPUT_LIMIT = 10
DEFAULT_FRAME_SIZE = "9:16"
DEFAULT_FRAME_ORIENTATION = "竖屏"
MAX_CONCURRENCY = 1


class TranscribeOutput(BaseModel):
    success: bool
    source: Dict[str, Any]
    stage1: Dict[str, Any]
    stage2: Dict[str, Any]
    segment_and_filters: List[Dict[str, Any]]
    audio_urls: List[str]
    segment_durations: List[float]


def _safe_name(text: str, *, max_len: int = 80) -> str:
    t = re.sub(r"\s+", " ", str(text or "").strip())
    t = re.sub(r'[\\/:*?"<>|]', "_", t)
    t = t.strip(" ._")
    return t[:max_len] if t else "item"


def _pick(row: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if not isinstance(raw, dict):
                continue
            rows.append({str(k).strip(): str(v).strip() for k, v in raw.items() if k is not None})
    return rows


def _build_script_text(row: Dict[str, str]) -> str:
    parts = [
        _pick(row, "口播"),
        _pick(row, "中段混剪"),
        _pick(row, "结尾口播"),
    ]
    return "\n\n".join([p for p in parts if p]).strip()


def _build_frame_context(row: Dict[str, str]) -> tuple[str, str]:
    frame_size = _pick(row, "frame_size", "画幅比例", "素材索引 frame_size", "比例")
    frame_orientation = _pick(
        row,
        "frame_orientation",
        "横竖屏",
        "横竖版",
        "素材索引 frame_orientation",
        "竖横屏",
    )
    return frame_size or DEFAULT_FRAME_SIZE, frame_orientation or DEFAULT_FRAME_ORIENTATION


def _build_input_item(row: Dict[str, str], *, index: int) -> Dict[str, Any]:
    topic = _pick(row, "主题", "topic")
    title = _pick(row, "标题", "title")
    car_model = _pick(row, "车型", "car_model")
    frame_size, frame_orientation = _build_frame_context(row)
    script = _build_script_text(row)
    return {
        "index": index,
        "topic": topic,
        "title": title,
        "car_model": car_model,
        "frame_size": frame_size,
        "frame_orientation": frame_orientation,
        "script": script,
    }


def _dump_any(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, list):
        return [_dump_any(v) for v in value]
    if isinstance(value, dict):
        return {k: _dump_any(v) for k, v in value.items()}
    return value


def _segment_and_filter(seg: Dict[str, Any], car_model: str) -> Dict[str, Any]:
    frame_orientation = str(seg.get("frame_orientation") or "").strip() or DEFAULT_FRAME_ORIENTATION
    return {
        "operator": "AND",
        "fields": [
            {"field": "frame_orientation", "value": frame_orientation},
            {"field": "car_model", "value": car_model},
            {"field": "movement", "value": str(seg.get("movement") or "").strip()},
            {"field": "product_status_scene", "value": str(seg.get("product_status_scene") or "").strip()},
        ],
    }


async def _rewrite_one(item: Dict[str, Any], *, total: int) -> Dict[str, Any]:
    from services.script_rewrite_service import rewrite_script_to_storyboard_and_tags

    idx = int(item.get("index") or 0)
    topic = str(item.get("topic") or "").strip()
    title = str(item.get("title") or "").strip()
    car_model = str(item.get("car_model") or "").strip()
    frame_size = str(item.get("frame_size") or "").strip() or DEFAULT_FRAME_SIZE
    frame_orientation = str(item.get("frame_orientation") or "").strip() or DEFAULT_FRAME_ORIENTATION
    script = str(item.get("script") or "").strip()

    file_base = f"{idx:03d}_{_safe_name(title or topic or car_model or str(idx))}"
    out_path = OUT_DIR / f"{file_base}.json"

    print(f"[{idx}/{total}] start {title or topic or car_model}")

    try:
        tts_audio_urls: List[Optional[str]] = []
        storyboard, tags = await rewrite_script_to_storyboard_and_tags(
            script=script,
            topic=topic or None,
            title=title or None,
            car_model=car_model or None,
            frame_size=frame_size,
            frame_orientation=frame_orientation,
            index=idx,
            tts_obs_project_id=f"transcribe_{idx:03d}",
            out_obs_audio_urls=tts_audio_urls,
        )

        stage1_dump = _dump_any(storyboard)
        stage2_dump = _dump_any(tags)
        segments = stage2_dump.get("segment_result") if isinstance(stage2_dump, dict) else []
        if not isinstance(segments, list):
            segments = []

        payload = TranscribeOutput(
            success=True,
            source={
                "row_index": idx,
                "topic": topic,
                "title": title,
                "car_model": car_model,
                "frame_size": frame_size,
                "frame_orientation": frame_orientation,
            },
            stage1=stage1_dump,
            stage2=stage2_dump,
            segment_and_filters=[
                _segment_and_filter(seg, car_model) for seg in segments if isinstance(seg, dict)
            ],
            audio_urls=list(tts_audio_urls),
            segment_durations=[
                float(seg.get("duration") or 0.0) if isinstance(seg, dict) else 0.0 for seg in segments
            ],
        ).model_dump(exclude_none=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{idx}/{total}] success -> {out_path.name}")
        return {"success": True, "path": str(out_path), "index": idx}
    except Exception as e:
        payload = {"success": False, "error": str(e)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{idx}/{total}] fail -> {out_path.name} err={e}")
        return {"success": False, "path": str(out_path), "index": idx, "error": str(e)}


async def main() -> None:
    rows = _load_rows(CSV_PATH)
    items: List[Dict[str, Any]] = []
    for i, row in enumerate(rows[:OUTPUT_LIMIT], start=1):
        item = _build_input_item(row, index=i)
        if not item["script"]:
            continue
        items.append(item)

    if not items:
        print("No transcribe items found.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _guarded_run(item: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            return await _rewrite_one(item, total=len(items))

    results = await asyncio.gather(*[asyncio.create_task(_guarded_run(item)) for item in items])
    ok = sum(1 for r in results if r.get("success"))
    fail = len(results) - ok
    print(f"\nSummary: ok={ok}, fail={fail}, total={len(results)}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
