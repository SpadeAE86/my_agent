"""
batch_rewrite_requirements_scripts.py

Batch rewrite "需求脚本.json" into:
- per-item storyboard json
- per-item tags json
- a flattened CSV for quick inspection (one row per segment)

Outputs are written under: src/test/scripts/

Run (py312):
  python -m src.test.batch_rewrite_requirements_scripts
or:
  python src/test/batch_rewrite_requirements_scripts.py
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from services.script_rewrite_service import rewrite_script_to_storyboard_and_tags  # noqa: E402


REQ_PATH = Path(__file__).resolve().parent / "需求脚本.json"
OUT_DIR = Path(__file__).resolve().parent / "scripts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "requirements_rewrite_segments.csv"

MAX_CONCURRENCY = 3  # SeedText calls are expensive; keep it small.


def _safe_name(s: str, *, max_len: int = 60) -> str:
    t = (s or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[\\\\/:*?\"<>|]", "_", t)
    t = t.strip(" ._")
    if not t:
        return "item"
    return t[:max_len]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _flat(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " | ".join([str(x) for x in v if x is not None and str(x).strip()])
    return str(v)


async def _run_one(item: Dict[str, Any], sem: asyncio.Semaphore) -> Dict[str, Any]:
    async with sem:
        sid = int(item.get("id") or 0)
        topic = str(item.get("topic") or "").strip()
        title = str(item.get("title") or "").strip()
        car_model = str(item.get("car_model") or "").strip()
        script = str(item.get("mid_mix") or "").strip()
        if not script:
            return {"id": sid, "success": False, "error": "empty mid_mix script"}

        storyboard, tags = await rewrite_script_to_storyboard_and_tags(
            script,
            topic=topic or None,
            title=title or None,
            car_model=car_model or None,
            index=sid,  # use requirement id as index for traceability
        )

        base = f"{sid:03d}_{_safe_name(title or topic or car_model or str(sid))}"
        sb_path = OUT_DIR / f"{base}_storyboard.json"
        tag_path = OUT_DIR / f"{base}_tags.json"

        sb_payload = {
            "id": sid,
            "topic": topic,
            "title": title,
            "car_model": car_model,
            "storyboard": storyboard.model_dump(exclude_none=True),
        }
        tag_payload = {
            "id": sid,
            "topic": topic,
            "title": title,
            "car_model": car_model,
            "tags": tags.model_dump(exclude_none=True),
        }

        sb_path.write_text(json.dumps(sb_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tag_path.write_text(json.dumps(tag_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "id": sid,
            "success": True,
            "topic": topic,
            "title": title,
            "car_model": car_model,
            "storyboard": storyboard,
            "tags": tags,
            "storyboard_path": str(sb_path),
            "tags_path": str(tag_path),
        }


async def main():
    if not REQ_PATH.exists():
        raise FileNotFoundError(f"requirements json not found: {REQ_PATH}")

    data = _load_json(REQ_PATH)
    items = (data or {}).get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("需求脚本.json must be an object with items: []")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    results = await asyncio.gather(*[asyncio.create_task(_run_one(it, sem)) for it in items[:5] if isinstance(it, dict)])

    ok = [r for r in results if r.get("success")]
    fail = [r for r in results if not r.get("success")]
    print(f"Done. ok={len(ok)} fail={len(fail)} out_dir={OUT_DIR}")
    if fail:
        (OUT_DIR / "requirements_rewrite_failures.json").write_text(
            json.dumps(fail, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Failures written:", str(OUT_DIR / "requirements_rewrite_failures.json"))

    # Flatten tags into CSV (one row per segment)
    rows: List[Dict[str, Any]] = []
    for r in ok:
        sid = int(r.get("id") or 0)
        topic = str(r.get("topic") or "")
        title = str(r.get("title") or "")
        car_model = str(r.get("car_model") or "")
        tags = r.get("tags")
        segs = getattr(tags, "segment_result", None)
        if not segs:
            continue
        for seg in segs:
            d = seg.model_dump(exclude_none=True)
            row = {
                "req_id": sid,
                "topic": topic,
                "title": title,
                "car_model": car_model,
                "segment_id": d.get("id"),
                "index": d.get("index"),
                "duration": d.get("duration"),
                "segment_text": d.get("segment_text"),
                "video_usage": _flat(d.get("video_usage")),
                "shot_style": d.get("shot_style"),
                "shot_type": d.get("shot_type"),
                "movement": d.get("movement"),
                "subject": d.get("subject"),
                "scene_location": _flat(d.get("scene_location")),
                "object": _flat(d.get("object")),
                "topic_tag": d.get("topic"),
                "text": _flat(d.get("text")),
                "key_fields_summary": _flat(d.get("function_selling_points")) + " || " + _flat(d.get("design_selling_points")),
                "marketing_phrases": _flat(d.get("marketing_phrases")),
                "extra_tags": _flat(d.get("extra_tags")),
            }
            rows.append(row)

    if rows:
        cols = list(rows[0].keys())
        with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                # Remove newlines to keep CSV viewer stable
                row2 = {k: (str(v).replace("\r", " ").replace("\n", " ") if v is not None else "") for k, v in row.items()}
                w.writerow(row2)
        print("CSV written:", str(OUT_CSV))


if __name__ == "__main__":
    asyncio.run(main())

