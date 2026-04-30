from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from services.script_match_service import match_script_tags_segments  # noqa: E402


TAGS_PATH = Path(
    r"c:\AI\AiGithubProject\DIYProject\src\test\scripts\005_这增程车安静得像纯电？我不信_tags.json"
)


async def _run(mode: str) -> None:
    data = json.loads(TAGS_PATH.read_text(encoding="utf-8"))
    segments = data["tags"]["segment_result"]
    out = await match_script_tags_segments(
        segments,
        top_k=5,
        global_k=200,
        search_pipeline="nlp-search-pipeline",
        mode=mode,
        vector_field="description_vector",
    )
    for i, seg in enumerate(out):
        top1 = (seg.get("top_hits") or [{}])[0]
        print(
            f"[{mode}] seg={i+1} top1_id={top1.get('_id')} score={top1.get('_score')} video={top1.get('video_path')}"
        )


async def main() -> None:
    for mode in ["global_then_segment_zero", "lite", "full"]:
        try:
            await _run(mode)
        except Exception as e:
            print(f"[{mode}] ERROR: {e!r}")


if __name__ == "__main__":
    asyncio.run(main())

