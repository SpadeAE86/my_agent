# -*- coding: utf-8 -*-
"""
排查 /video-analysis/search 命中为 0：通常是 AND→term filter 在 keyword 字段上**要求全等**，
例如请求 car_model=LS6 而索引里是「智己LS6」，或 movement 枚举值与素材里不一致。

在 my_agent/src 下执行（需与线上一致的 config / OpenSearch）::

  C:\\Users\\25065\\.conda\\envs\\py312\\python.exe -m test.debug_video_analysis_search_zero_hits
  C:\\Users\\25065\\.conda\\envs\\py312\\python.exe -m test.debug_video_analysis_search_zero_hits --json path/to/body.json

body.json 示例为 POST /video-analysis/search 的 JSON（tokens / fuzzy / workspace 等）。
不加 --json 时使用内置的一条与「匹配页导航回分析」相近的请求样例。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from pydantic import TypeAdapter  # noqa: E402

from infra.storage.opensearch_connector import opensearch_connector  # noqa: E402
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2  # noqa: E402
from models.pydantic.opensearch_index.base_index import get_index_name  # noqa: E402
from routers.video_analysis import (  # noqa: E402
    VideoAnalysisSearchToken,
    _video_analysis_split_tokens,
)


# 与用户日志同结构的样例（可改用 --json 覆盖）
_BUILTIN_BODY: Dict[str, Any] = {
    "fuzzy": True,
    "use_rrf": False,
    "workspace": "v2",
    "size": 80,
    "tokens": [
        {"text": "LS6", "join": "AND", "not": False, "type": "keyword", "source_field": "car_model"},
        {"text": "竖版9:16", "join": "AND", "not": False, "type": "keyword", "source_field": "frame_size"},
        {"text": "静态内饰", "join": "AND", "not": False, "type": "keyword", "source_field": "product_status_scene"},
        {"text": "TVC切片", "join": "OR", "not": False, "type": "keyword"},
        {"text": "静态展示", "join": "AND", "not": False, "type": "keyword", "source_field": "movement"},
        {"text": "MiniLED车载屏", "join": "OR", "not": False, "type": "keyword"},
        {"text": "中控屏", "join": "OR", "not": False, "type": "keyword"},
        {"text": "固定机位", "join": "OR", "not": False, "type": "keyword"},
        {"text": "特写", "join": "OR", "not": False, "type": "keyword"},
        {"text": "室内", "join": "OR", "not": False, "type": "keyword"},
    ],
}


def _term_fields_from_filters(term_filters: List[dict]) -> List[str]:
    out: List[str] = []
    for t in term_filters or []:
        term = t.get("term") if isinstance(t, dict) else None
        if isinstance(term, dict) and len(term) == 1:
            out.append(str(next(iter(term.keys()))))
    return out


async def _count(client: Any, index: str, query: dict) -> int:
    body = {"size": 0, "track_total_hits": True, "query": query}
    r = await client.search(index=index, body=body)
    total = (r.get("hits") or {}).get("total") or {}
    if isinstance(total, dict):
        return int(total.get("value") or 0)
    return int(total or 0)


async def _facet_terms(client: Any, index: str, field: str, size: int = 30) -> List[tuple[str, int]]:
    body = {
        "size": 0,
        "aggs": {f"f_{field}": {"terms": {"field": field, "size": size, "missing": "__missing__"}}},
    }
    r = await client.search(index=index, body=body)
    buckets = (((r.get("aggregations") or {}).get(f"f_{field}") or {}).get("buckets")) or []
    return [(str(b.get("key")), int(b.get("doc_count") or 0)) for b in buckets]


async def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose video-analysis /search 0 hits (term filters vs index).")
    ap.add_argument("--json", dest="json_path", default="", help="POST body JSON 文件路径")
    args = ap.parse_args()

    raw: Dict[str, Any]
    if (args.json_path or "").strip():
        with open(args.json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = dict(_BUILTIN_BODY)

    ta = TypeAdapter(List[VideoAnalysisSearchToken])
    tokens = ta.validate_python(raw.get("tokens") or [])
    ws = str(raw.get("workspace") or "v2").strip() or "v2"
    index_is_v2 = ws == "v2"
    query_text, term_filters, _must_not = _video_analysis_split_tokens(
        tokens, index_is_v2=index_is_v2
    )

    await opensearch_connector.ensure_init()
    client = await opensearch_connector.get_client()
    idx = get_index_name(CarInteriorAnalysisV2)

    print("=== Parsed (same as /search) ===")
    print("workspace:", ws, "index:", idx)
    print("query_text (first 200 chars):", (query_text or "")[:200], "..." if len(query_text or "") > 200 else "")
    print("term_filters:", json.dumps(term_filters, ensure_ascii=False))

    n_total = await _count(client, idx, {"match_all": {}})
    print("\n=== Counts ===")
    print("match_all:", n_total)

    if not term_filters:
        print("(no term filters)")
    else:
        n_all = await _count(client, idx, {"bool": {"filter": term_filters}})
        print("bool.filter ALL term_filters:", n_all)

        for i, f in enumerate(term_filters):
            n_one = await _count(client, idx, {"bool": {"filter": [f]}})
            print(f"  only filter[{i}] {f}: {n_one}")

        if len(term_filters) > 1:
            for i in range(len(term_filters)):
                subset = [term_filters[j] for j in range(len(term_filters)) if j != i]
                n_leave = await _count(client, idx, {"bool": {"filter": subset}})
                print(f"  ALL except filter[{i}]: {n_leave}")

    fields = _term_fields_from_filters(term_filters)
    print("\n=== Index facet (top values per filtered field) ===")
    for fld in dict.fromkeys(fields):
        try:
            pairs = await _facet_terms(client, idx, fld, size=25)
            print(f"  {fld}:", pairs[:15], ("..." if len(pairs) > 15 else ""))
        except Exception as e:
            print(f"  {fld}: <aggregation failed: {e}>")

    print(
        "\nHint: keyword term 须与索引一致。横竖屏约束请优先用索引字段 ``frame_orientation``（横屏/竖屏/未知）"
        "并在转写模板 AND 中选该字段；``frame_size`` 仍保留具体比例。新集群启动后会自动回填 ``frame_orientation``。"
    )

    await opensearch_connector.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
