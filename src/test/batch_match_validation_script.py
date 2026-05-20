"""
Batch matcher for transcribe outputs under `data/transcribe_validation/`.

Rules:
- read 10 transcribe JSON files
- do not enable road-run fallback by default
- keep BM25 as default weights
- boost vectors for description / marketing_phrases / selling_points style fields
- write 10 match JSON files under `data/match_result/`
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIBE_DIR = PROJECT_ROOT / "data" / "transcribe_validation"
MATCH_DIR = PROJECT_ROOT / "data" / "match_result"

DEFAULT_TOP_K = 5
DEFAULT_MODE = "field_aligned_hybrid"
DEFAULT_CONCURRENCY = 1


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_name(name: str, index: int) -> str:
    text = str(name or "").strip()
    if not text:
        return f"{index:03d}"
    return f"{index:03d}_{''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in text)[:80]}"


def _match_segments_input(stage2: Dict[str, Any]) -> List[Dict[str, Any]]:
    segs = stage2.get("segment_result") if isinstance(stage2, dict) else []
    if not isinstance(segs, list):
        return []
    out: List[Dict[str, Any]] = []
    for seg in segs:
        if not isinstance(seg, dict):
            continue
        seg2 = dict(seg)
        seg2["duration"] = 0.0
        out.append(seg2)
    return out


async def _match_one(path: Path, index: int, total: int) -> Dict[str, Any]:
    from services.script_match_service import match_script_tags_segments

    data = _load_json(path)
    source = data.get("source") if isinstance(data, dict) else {}
    stage2 = data.get("stage2") if isinstance(data, dict) else {}
    segments = _match_segments_input(stage2 if isinstance(stage2, dict) else {})
    car_model = str((source or {}).get("car_model") or "").strip()
    top_k = DEFAULT_TOP_K

    print(f"[{index}/{total}] start {path.name}")
    try:
        results = await match_script_tags_segments(
            segments,
            top_k=top_k,
            global_k=200,
            search_pipeline="nlp-search-pipeline",
            vector_field="description_vector",
            text_fields=[
                "description",
                "marketing_phrases",
                "design_selling_points",
                "function_selling_points",
                "subject",
                "object",
                "scene_location",
                "scenario_a",
                "scenario_b",
                "text",
            ],
            mode=DEFAULT_MODE,
            shot_cards_version="v2",
            concurrency=1,
            bm25_factor=0.5,
            vector_factor=0.5,
            use_rrf=False,
            with_timings=True,
            enable_road_run_fallback=False,
        )

        out_segments: List[Dict[str, Any]] = []
        for seg_in, res in zip(segments, results):
            top_hits = res.get("top_hits") or []
            out_segments.append(
                {
                    "segment_id": seg_in.get("id"),
                    "segment_text": seg_in.get("segment_text"),
                    "duration": seg_in.get("duration"),
                    "description": seg_in.get("description"),
                    "top5": [
                        {
                            "video_path": h.get("video_path"),
                            "history_id": h.get("history_id"),
                            "score": h.get("_score"),
                            "id": h.get("_id"),
                        }
                        for h in top_hits[:5]
                        if isinstance(h, dict)
                    ],
                    "top1_video_path": (top_hits[0].get("video_path") if top_hits and isinstance(top_hits[0], dict) else None),
                    "search_status": "done" if top_hits else "failed",
                    "raw": res,
                }
            )

        payload = {
            "success": True,
            "source": {
                "source_file": path.name,
                "topic": (source or {}).get("topic"),
                "title": (source or {}).get("title"),
                "car_model": car_model,
            },
            "match_mode": {
                "mode": DEFAULT_MODE,
                "use_rrf": False,
                "bm25_factor": 0.5,
                "vector_factor": 0.5,
                "vector_fields_hint": [
                    "description_vector",
                    "marketing_phrases_vector",
                    "function_selling_points_vector",
                    "design_selling_points_vector",
                ],
            },
            "segments": out_segments,
        }
        out_path = MATCH_DIR / path.name
        _dump_json(out_path, payload)
        print(f"[{index}/{total}] success -> {out_path.name}")
        return {"success": True, "path": str(out_path)}
    except Exception as e:
        out_path = MATCH_DIR / path.name
        _dump_json(out_path, {"success": False, "error": str(e)})
        print(f"[{index}/{total}] fail -> {out_path.name} err={e}")
        return {"success": False, "path": str(out_path), "error": str(e)}


async def main() -> None:
    files = sorted([p for p in TRANSCRIBE_DIR.glob("*.json") if p.is_file()])[:10]
    if not files:
        print(f"No transcribe json files found under {TRANSCRIBE_DIR}")
        return
    MATCH_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(DEFAULT_CONCURRENCY)

    async def guarded(p: Path, i: int) -> Dict[str, Any]:
        async with sem:
            return await _match_one(p, i, len(files))

    results = await asyncio.gather(*[asyncio.create_task(guarded(p, i)) for i, p in enumerate(files, start=1)])
    ok = sum(1 for r in results if r.get("success"))
    fail = len(results) - ok
    print(f"\nSummary: ok={ok}, fail={fail}, total={len(results)}")
    print(f"Output dir: {MATCH_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
