"""
search_from_script_tags_output.py

Take Stage2-style segment tags (list of dicts), search OpenSearch index
`car_interior_analysis_v2`, then map hit ids back to MySQL history
(VideoAnalysisHistory) to retrieve local video paths.

切换输入方式：改脚本顶部 INPUT_MODE 与对应常量即可（无需命令行参数）。

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


# =============================================================================
# 输入方式（只选一个）：改 INPUT_MODE 即可
#   - "file"  ：从 TAGS_JSON_PATH 读 JSON
#   - "inline"：用下面的 INLINE_SEGMENT_TAGS（不落盘）
#
# JSON 支持三种形状：
#   - {"segment_result": [...]}
#   - {"tags": {"segment_result": [...]}}   # 如 scripts/001_*_tags.json
#   - [{ ... }, ...]                       # 纯数组
# =============================================================================
INPUT_MODE = "file"  # "file" | "inline"

# file 模式下的路径（可改为任意 `_tags.json`；相对路径会先相对本脚本目录再找 cwd）
TAGS_JSON_PATH = Path(__file__).resolve().parent / "scripts" / "001_一周只充一次电？这车也太省心了吧_tags.json"

# 备选：项目根目录旧的 script_tags_output.json（需要时可改掉 TAGS_JSON_PATH 指向它）
# TAGS_JSON_PATH = Path(__file__).resolve().parent.parent.parent / "script_tags_output.json"

# inline 模式：在此处粘贴 segment dict 列表
INLINE_SEGMENT_TAGS: List[Dict[str, Any]] = [
    # {"id": 1, "segment_text": "...", "description": "...", ...},
]


OUT_PATH = Path(__file__).resolve().parent / "script_tags_search_results.json"

SEARCH_PIPELINE: Optional[str] = "nlp-search-pipeline"  # set None to disable (may break hybrid stability)
MODE: str = "global_then_segment"

TOP_K = 5
GLOBAL_K = 200


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_existing_path(p: Path) -> Path:
    """Resolve path: absolute/as-is, then relative to this script dir, then cwd."""
    if p.is_file():
        return p.resolve()
    test_rel = Path(__file__).resolve().parent / p
    if test_rel.is_file():
        return test_rel.resolve()
    cwd_rel = Path.cwd() / p
    if cwd_rel.is_file():
        return cwd_rel.resolve()
    raise FileNotFoundError(f"tags JSON not found: {p}")


def segments_from_payload(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [s for s in obj if isinstance(s, dict)]
    if isinstance(obj, dict):
        sr = obj.get("segment_result")
        if isinstance(sr, list):
            return [s for s in sr if isinstance(s, dict)]
        tags = obj.get("tags")
        if isinstance(tags, dict):
            sr2 = tags.get("segment_result")
            if isinstance(sr2, list):
                return [s for s in sr2 if isinstance(s, dict)]
    raise ValueError(
        "Expected JSON: list of segments, or object with segment_result, "
        "or object with tags.segment_result"
    )


def load_segments() -> Tuple[List[Dict[str, Any]], str]:
    if INPUT_MODE == "inline":
        if not INLINE_SEGMENT_TAGS:
            raise ValueError(
                'INPUT_MODE is "inline" but INLINE_SEGMENT_TAGS is empty — paste segments or switch INPUT_MODE to "file".'
            )
        return list(INLINE_SEGMENT_TAGS), "inline:INLINE_SEGMENT_TAGS"

    if INPUT_MODE != "file":
        raise ValueError(f'INPUT_MODE must be "file" or "inline", got: {INPUT_MODE!r}')

    path = resolve_existing_path(TAGS_JSON_PATH)
    data = _load_json(path)
    return segments_from_payload(data), str(path)


async def run_search(segments: List[Dict[str, Any]], *, meta_input: str) -> None:
    try:
        results = await match_script_tags_segments(
            [s for s in segments if isinstance(s, dict)],
            top_k=TOP_K,
            global_k=GLOBAL_K,
            search_pipeline=SEARCH_PIPELINE,
            mode=MODE,
        )

        OUT_PATH.write_text(
            json.dumps(
                {"input_source": meta_input, "segment_count": len(segments), "results": results},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("Wrote:", str(OUT_PATH))
        print("input_source:", meta_input)
    finally:
        try:
            await opensearch_connector.close()
        except Exception:
            pass
        try:
            await mysql_connector.close()
        except Exception:
            pass


async def main() -> None:
    segments, meta = load_segments()
    if not segments:
        raise ValueError("No segments to search.")
    await run_search(segments, meta_input=meta)


if __name__ == "__main__":
    asyncio.run(main())
