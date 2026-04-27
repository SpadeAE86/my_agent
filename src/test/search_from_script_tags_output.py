"""
search_from_script_tags_output.py

Take script_tags_output.json (Stage2 tags for each script segment),
search OpenSearch index `car_interior_analysis_v2`, then map hit ids back
to MySQL history (VideoAnalysisHistory) to retrieve local video paths.

Run:
  python -m src.test.search_from_script_tags_output
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.mysql_connector import mysql_connector  # noqa: E402
from infra.storage.opensearch_connector import opensearch_connector  # noqa: E402
from services.script_match_service import match_script_tags_segments  # noqa: E402


SCRIPT_TAGS_PATH = Path(__file__).resolve().parent.parent.parent / "script_tags_output.json"
OUT_PATH = Path(__file__).resolve().parent / "script_tags_search_results.json"

SEARCH_PIPELINE: Optional[str] = "nlp-search-pipeline"  # set None to disable (may break hybrid stability)
MODE: str = "lite"  # lite/full

TOP_K = 5

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


async def main():
    if not SCRIPT_TAGS_PATH.exists():
        raise FileNotFoundError(f"script tags not found: {SCRIPT_TAGS_PATH}")

    data = _load_json(SCRIPT_TAGS_PATH)
    segs = (data or {}).get("segment_result") if isinstance(data, dict) else None
    if not isinstance(segs, list):
        raise ValueError("script_tags_output.json must be an object with segment_result: []")

    try:
        results = await match_script_tags_segments(
            [s for s in segs if isinstance(s, dict)],
            top_k=TOP_K,
            search_pipeline=SEARCH_PIPELINE,
            mode=MODE,
        )

        OUT_PATH.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Wrote:", str(OUT_PATH))
    finally:
        # Close OpenSearch client session (script-mode cleanup).
        try:
            await opensearch_connector.close()
        except Exception:
            pass
        # Avoid "Event loop is closed" warnings from aiomysql connection __del__.
        try:
            await mysql_connector.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())

