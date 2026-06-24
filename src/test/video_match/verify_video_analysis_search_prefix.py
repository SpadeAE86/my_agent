# -*- coding: utf-8 -*-
"""
Plan 2A–2C: OpenSearch 文档 id、MySQL history_id vs video_key、prefix filter A/B.

使用项目的 config.yml ENV（如 test）连接 OpenSearch + MySQL。

在 my_agent/src 目录下::

  C:\\Users\\25065\\.conda\\envs\\py312\\python.exe -m test.verify_video_analysis_search_prefix
  C:\\Users\\25065\\.conda\\envs\\py312\\python.exe -m test.verify_video_analysis_search_prefix --history-id <界面里的历史id>

可选：若只想看 OpenSearch、跳过 MySQL（例如本地无库），加 ``--skip-mysql``。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from sqlalchemy import select  # noqa: E402

from infra.storage.opensearch_connector import opensearch_connector  # noqa: E402
from infra.storage.mysql_connector import mysql_connector  # noqa: E402
from models.pydantic.opensearch_index.car_interior_analysis_v2 import (  # noqa: E402
    CarInteriorAnalysisV2,
)
from models.pydantic.opensearch_index.base_index import get_index_name, get_searchable_fields  # noqa: E402
from models.sqlmodel.video_analysis import VideoAnalysisHistory, VideoAnalysisVideoV2  # noqa: E402


def _prefix_from_doc_id(doc_id: str) -> str:
    s = (doc_id or "").strip()
    if "_" not in s:
        return s
    return s.rsplit("_", 1)[0]


async def main() -> None:
    ap = argparse.ArgumentParser(description="Verify video-analysis search prefix / DB alignment.")
    ap.add_argument(
        "--history-id",
        default="",
        help="界面所选历史的 id：用于对 OpenSearch 做 prefix filter 命中数对比",
    )
    ap.add_argument("--skip-mysql", action="store_true", help="仅跑 OpenSearch 部分")
    args = ap.parse_args()

    await opensearch_connector.ensure_init()
    oc = await opensearch_connector.get_client()
    idx = get_index_name(CarInteriorAnalysisV2)

    # --- 2A OpenSearch ---
    cnt = await oc.count(index=idx, body={"query": {"match_all": {}}})
    sr = await oc.search(
        index=idx,
        body={"size": 15, "query": {"match_all": {}}, "_source": ["id"]},
    )
    hits = ((sr.get("hits") or {}).get("hits") or [])
    doc_ids = [str(h.get("_id") or "") for h in hits if h.get("_id")]
    prefixes = sorted({_prefix_from_doc_id(d) for d in doc_ids if d})
    print("=== 2A OpenSearch ===")
    print("index:", idx)
    print("count:", cnt.get("count"))
    print("sample _id (up to 5):", doc_ids[:5])
    print("distinct prefixes in sample:", prefixes[:8])

    if not args.skip_mysql:
        # --- 2B MySQL ---
        print("\n=== 2B MySQL (workspace=v2) ===")
        await mysql_connector.ensure_init()
        mismatches: list[tuple[str, str, str]] = []
        async with mysql_connector.session_scope() as session:
            res = await session.execute(
                select(VideoAnalysisHistory.id, VideoAnalysisHistory.name).where(
                    VideoAnalysisHistory.workspace == "v2"
                )
            )
            hist_rows = list(res.all())
            print("v2 history rows scanned:", len(hist_rows))
            for hid, name in hist_rows[:200]:
                hid_s = str(hid or "").strip()
                if not hid_s:
                    continue
                base = os.path.basename((name or "").strip().replace("\\", "/"))
                r1 = await session.execute(
                    select(VideoAnalysisVideoV2).where(VideoAnalysisVideoV2.video_key == hid_s)
                )
                if r1.scalar_one_or_none() is not None:
                    continue
                if not base:
                    continue
                r2 = await session.execute(
                    select(VideoAnalysisVideoV2).where(VideoAnalysisVideoV2.source_file_name == base)
                )
                vrow = r2.scalar_one_or_none()
                if vrow is None:
                    continue
                vk = str(vrow.video_key or "").strip()
                if vk and vk != hid_s:
                    mismatches.append((hid_s, base, vk))

        print("cases where history.id != video_key (resolved by source_file_name):", len(mismatches))
        for row in mismatches[:12]:
            print(" ", row)

    # --- 2C prefix A/B：与 /search 里 bool.filter prefix id 一致 ---
    text_fields = get_searchable_fields(CarInteriorAnalysisV2)
    weighted_fields = [f"{f}^1.0" for f in text_fields[:12]]
    base_q: dict = {
        "multi_match": {
            "query": "内饰",
            "fields": weighted_fields,
            "type": "best_fields",
        }
    }

    print("\n=== 2C OpenSearch prefix A/B (multi_match + optional prefix on field id) ===")
    body_no_prefix = {"size": 10, "query": base_q, "_source": False}
    ra = await oc.search(index=idx, body=body_no_prefix)
    ha = ((ra.get("hits") or {}).get("hits") or [])
    print("without prefix filter hits:", len(ha))

    if doc_ids:
        good_p = _prefix_from_doc_id(doc_ids[0])
        body_good = {
            "size": 10,
            "query": {
                "bool": {
                    "must": [base_q],
                    "filter": [{"prefix": {"id": f"{good_p}_"}}],
                }
            },
            "_source": False,
        }
        rb = await oc.search(index=idx, body=body_good)
        hb = ((rb.get("hits") or {}).get("hits") or [])
        print(f"with CORRECT prefix id={good_p!r}_ hits:", len(hb))

        fake = f"__no_such_prefix_{good_p}__"
        body_bad = {
            "size": 10,
            "query": {
                "bool": {
                    "must": [base_q],
                    "filter": [{"prefix": {"id": f"{fake}_"}}],
                }
            },
            "_source": False,
        }
        rc = await oc.search(index=idx, body=body_bad)
        hc = ((rc.get("hits") or {}).get("hits") or [])
        print(f"with bogus prefix id={fake!r}_ hits:", len(hc))

    ui_hid = (args.history_id or "").strip()
    if ui_hid:
        body_ui = {
            "size": 10,
            "query": {
                "bool": {
                    "must": [base_q],
                    "filter": [{"prefix": {"id": f"{ui_hid}_"}}],
                }
            },
            "_source": False,
        }
        rd = await oc.search(index=idx, body=body_ui)
        hd = ((rd.get("hits") or {}).get("hits") or [])
        print(f"with --history-id={ui_hid!r} as prefix hits:", len(hd))
        if len(ha) > 0 and len(hd) == 0:
            print(
                "-> 无 prefix 有命中、带 UI history_id 的 prefix 无命中：说明索引文档 id 前缀与该 id 不一致；"
                "视频分析 /search 已不再按 history 收窄（全 workspace 索引检索）。"
            )

    await opensearch_connector.close()
    if not args.skip_mysql:
        await mysql_connector.close()

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
