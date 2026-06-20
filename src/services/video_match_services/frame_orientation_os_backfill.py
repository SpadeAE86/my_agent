# -*- coding: utf-8 -*-
"""
为已有 OpenSearch v2 文档补充 ``frame_orientation``（由 ``frame_size`` 派生）。
启动时后台执行一次；也可用（在 ``my_agent/src`` 下）::

  python -m services.frame_orientation_os_backfill
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

from infra.logging.logger import logger as log
from infra.storage.opensearch_connector import opensearch_connector
from models.pydantic.opensearch_index.base_index import get_index_name
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
from utils.frame_orientation import infer_frame_orientation


async def ensure_frame_orientation_mapping(client: Any, index_name: str) -> None:
    m = await client.indices.get_mapping(index=index_name)
    root = m.get(index_name) or next(iter(m.values()), {})
    props = (root.get("mappings") or {}).get("properties") or {}
    if "frame_orientation" in props:
        return
    await client.indices.put_mapping(
        index=index_name,
        body={"properties": {"frame_orientation": {"type": "keyword"}}},
    )
    log.info(f"OpenSearch put_mapping: frame_orientation (keyword) on {index_name}")


async def run_frame_orientation_backfill(
    *,
    index_name: Optional[str] = None,
    batch_size: int = 200,
    scroll_ttl: str = "2m",
) -> int:
    """
    Scroll 全索引，按 ``frame_size`` 写回 ``frame_orientation``。
    返回执行了 partial update 的文档条数。
    """
    await opensearch_connector.ensure_init()
    client = await opensearch_connector.get_client()
    idx = (index_name or "").strip() or get_index_name(CarInteriorAnalysisV2)
    await ensure_frame_orientation_mapping(client, idx)

    resp = await client.search(
        index=idx,
        body={
            "query": {"match_all": {}},
            "_source": ["frame_size", "frame_orientation"],
            "size": batch_size,
            "sort": ["_doc"],
        },
        scroll=scroll_ttl,
    )
    scroll_id = resp.get("_scroll_id")
    updated = 0
    try:
        while True:
            hits = ((resp.get("hits") or {}).get("hits")) or []
            if not hits:
                break
            lines: list[str] = []
            for h in hits:
                hid = h.get("_id")
                src = h.get("_source") or {}
                if not hid:
                    continue
                fs = src.get("frame_size")
                want = infer_frame_orientation(str(fs if fs is not None else ""))
                cur = src.get("frame_orientation")
                if cur == want:
                    continue
                lines.append(json.dumps({"update": {"_index": idx, "_id": hid}}))
                lines.append(json.dumps({"doc": {"frame_orientation": want}}))
            if lines:
                bulk_body = "\n".join(lines) + "\n"
                br = await client.bulk(body=bulk_body, refresh=False)
                if br.get("errors"):
                    log.warning(f"frame_orientation backfill bulk had errors: {br}")
                updated += len(lines) // 2
            scroll_id = resp.get("_scroll_id") or scroll_id
            if not scroll_id:
                break
            resp = await client.scroll(scroll_id=scroll_id, scroll=scroll_ttl)
    finally:
        if scroll_id:
            try:
                await client.clear_scroll(scroll_id=scroll_id)
            except Exception as e:
                log.debug("clear_scroll: %s", e)
        try:
            await client.indices.refresh(index=idx)
        except Exception as e:
            log.debug("refresh after backfill: %s", e)

    log.info(f"frame_orientation backfill on {idx}: updated {updated} docs")
    return updated


async def _cli() -> None:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.dirname(CURRENT_DIR)
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    try:
        n = await run_frame_orientation_backfill()
        print("updated:", n)
    finally:
        await opensearch_connector.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(_cli())
