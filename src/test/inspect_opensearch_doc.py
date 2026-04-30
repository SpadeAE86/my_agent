"""
inspect_opensearch_doc.py

PyCharm-friendly helper:
- Set DOC_ID below (default), or override via CLI
- Fetch a single OpenSearch document by _id from index `car_interior_analysis_v2`
- Print all non-vector fields (skip *_vector)
- Also join MySQL `video_analysis_shot_cards` to print frame_urls/thumbnail for that scene.

Run:
  python -m src.test.inspect_opensearch_doc
  python -m src.test.inspect_opensearch_doc --id v2_xxx_scene_001
  python -m src.test.inspect_opensearch_doc v2_xxx_scene_001
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch_connector import opensearch_connector  # noqa: E402
from infra.storage.mysql_connector import mysql_connector  # noqa: E402
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2  # noqa: E402
from models.pydantic.opensearch_index.base_index import get_index_name, get_vector_fields  # noqa: E402
from models.sqlmodel.video_analysis import VideoAnalysisShotCard  # noqa: E402
from sqlmodel import select  # noqa: E402
from pathlib import Path  # noqa: E402


DOC_ID = "v2_fe056c28740e_scene_001"  # <- change me
OUT_JSON = Path(__file__).resolve().parent / "workspace" / "inspect_opensearch_doc.json"


def _strip_vectors(src: Dict[str, Any], vector_fields: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (src or {}).items():
        if k in vector_fields:
            continue
        if k.endswith("_vector"):
            continue
        out[k] = v
    return out


def _parse_history_and_scene_id(doc_id: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Doc id format in this project: {history_id}_scene_{scene_id:03d}
    Example: v2_17a6bc414c8a_scene_001 -> (v2_17a6bc414c8a, 1)
    """
    s = str(doc_id or "").strip()
    if not s:
        return None, None
    if "_scene_" not in s:
        return None, None
    hid, tail = s.split("_scene_", 1)
    try:
        sid = int(str(tail).lstrip("0") or "0")
    except Exception:
        sid = None
    return hid or None, sid


async def _fetch_shot_card_frames(history_id: str, scene_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch frame_urls/thumbnail from MySQL for (history_id, scene_id).
    """
    if not history_id or not scene_id:
        return None
    async with mysql_connector.session_scope() as session:
        res = await session.execute(
            select(VideoAnalysisShotCard).where(
                (VideoAnalysisShotCard.history_id == history_id)
                & (VideoAnalysisShotCard.scene_id == int(scene_id))
            )
        )
        row = res.scalars().first()
        if not row:
            return None
        d = row.model_dump(exclude_none=True)
        # Only keep the bits we want to print
        return {
            "history_id": history_id,
            "scene_id": int(scene_id),
            "thumbnail": d.get("thumbnail"),
            "frame_urls": d.get("frame_urls") or [],
        }

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument(
        "doc_id",
        nargs="?",
        default=None,
        help="OpenSearch _id (optional). If omitted, uses DOC_ID constant in file.",
    )
    p.add_argument(
        "--id",
        dest="doc_id_flag",
        default=None,
        help="OpenSearch _id (overrides positional and DOC_ID).",
    )
    return p.parse_args()


async def main() -> None:
    args = _parse_args()
    doc_id = (args.doc_id_flag or args.doc_id or DOC_ID or "").strip()
    if not doc_id:
        raise ValueError("doc_id is empty. Provide --id <doc_id> or edit DOC_ID in this file.")

    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()

    idx = get_index_name(CarInteriorAnalysisV2)
    vector_fields = get_vector_fields(CarInteriorAnalysisV2)

    resp = await c.get(index=idx, id=doc_id)
    found = bool((resp or {}).get("found"))
    if not found:
        print("NOT FOUND:", doc_id)
        await opensearch_connector.close()
        return

    src = (resp or {}).get("_source") or {}
    src2 = _strip_vectors(src, vector_fields)

    print("index:", idx)
    print("_id:", doc_id)
    print("\n=== _source (non-vector) ===")
    # sort keys for stable viewing
    ordered = {k: src2[k] for k in sorted(src2.keys())}
    print(json.dumps(ordered, ensure_ascii=False, indent=2))

    # Write to file for reliable viewing (avoid console mojibake on Windows)
    try:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nWrote:", str(OUT_JSON))
    except Exception:
        pass

    # Best-effort: join with MySQL shot card to see which frames produced these tags.
    history_id, scene_id = _parse_history_and_scene_id(doc_id)
    if history_id and scene_id:
        try:
            await mysql_connector.ensure_init()
            card = await _fetch_shot_card_frames(history_id, int(scene_id))
            print("\n=== MySQL shot card frames ===")
            if card:
                print(json.dumps(card, ensure_ascii=False, indent=2))
            else:
                print(f"<not found> history_id={history_id} scene_id={scene_id}")
        except Exception as e:
            print("\n=== MySQL shot card frames ===")
            print(f"<error> {e}")
        finally:
            try:
                await mysql_connector.close()
            except Exception:
                pass

    await opensearch_connector.close()


if __name__ == "__main__":
    asyncio.run(main())

