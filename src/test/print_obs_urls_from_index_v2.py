# -*- coding: utf-8 -*-
"""
print_obs_urls_from_index_v2.py

Print OBS URLs (video_url) for documents currently in OpenSearch index
`car_interior_analysis_v2`.

How it works:
- Fetch doc _id list from OpenSearch (match_all, paginated via search_after).
- Convert doc_id -> history_id (doc_id prefix before "_scene_").
- Fetch history item from MySQL and read its `video_url` (OBS URL).

Run:
  python -m src.test.print_obs_urls_from_index_v2

Optional env vars:
  LIMIT=0        # 0 means no limit (default)
  PAGE_SIZE=200  # page size for OpenSearch
"""

from __future__ import annotations

import os
import sys
import asyncio
from typing import Any, Dict, List, Optional, Set

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch_connector import opensearch_connector
from services.video_analysis_db_service import video_analysis_db_service


INDEX_NAME = "car_interior_analysis_v2"


def _history_id_from_doc_id(doc_id: str) -> str:
    if not doc_id:
        return ""
    if "_scene_" in doc_id:
        return doc_id.split("_scene_", 1)[0]
    return doc_id


async def _fetch_doc_ids(
    client: Any,
    *,
    index: str,
    page_size: int,
    limit: int,
) -> List[str]:
    """
    Return doc ids from OpenSearch using search_after pagination.
    We sort by _id asc for stable paging.
    """
    out: List[str] = []
    search_after: Optional[List[Any]] = None

    while True:
        size = int(page_size)
        if limit > 0:
            size = min(size, max(0, limit - len(out)))
            if size <= 0:
                break

        body: Dict[str, Any] = {
            "size": size,
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

        # update cursor for next page
        last_sort = hits[-1].get("sort")
        if not isinstance(last_sort, list) or not last_sort:
            break
        search_after = last_sort

    return out


async def main() -> None:
    limit = int(os.getenv("LIMIT", "0") or "0")
    page_size = int(os.getenv("PAGE_SIZE", "200") or "200")

    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()

    doc_ids = await _fetch_doc_ids(c, index=INDEX_NAME, page_size=page_size, limit=limit)
    history_ids = [_history_id_from_doc_id(i) for i in doc_ids]
    # de-dup preserve order
    seen_h: Set[str] = set()
    history_ids_uniq: List[str] = []
    for hid in history_ids:
        if not hid or hid in seen_h:
            continue
        seen_h.add(hid)
        history_ids_uniq.append(hid)

    url_cache: Dict[str, str] = {}
    urls: List[str] = []
    for hid in history_ids_uniq:
        item = await video_analysis_db_service.get_history_item(hid)
        url = str((item or {}).get("video_url") or "")
        url_cache[hid] = url
        if url and url not in urls:
            urls.append(url)

    print("index:", INDEX_NAME)
    print("docs:", len(doc_ids))
    print("history_ids:", len(history_ids_uniq))
    print("unique_obs_urls:", len(urls))
    print("")
    for u in urls:
        print(u)

    await opensearch_connector.close()


if __name__ == "__main__":
    asyncio.run(main())

