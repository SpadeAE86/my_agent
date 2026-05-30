# -*- coding: utf-8 -*-
"""
build_elasticsearch_index_v2.py

在 Elasticsearch 环境下创建 ``car_interior_analysis_v2`` 映射，
并扫 MySQL ``video_analysis_shot_cards_v2`` 重新 embedding 并 bulk 写入 Elasticsearch。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

# Ensure we can import from src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _apply_env_from_argv() -> None:
    args = sys.argv[1:]
    if "--env" in args:
        ix = args.index("--env")
        if ix + 1 < len(args):
            os.environ["env"] = args[ix + 1].strip()
            return
    if not (os.getenv("env") or "").strip():
        os.environ["env"] = "local"  # Default to local for ES test


_apply_env_from_argv()

# Set search_provider to elasticsearch to force ES execution path
from config.config import ENV, MY_CONFIG  # noqa: E402
MY_CONFIG["search_provider"] = "elasticsearch"

from infra.storage.elasticsearch.create_index import index_manager  # noqa: E402
from infra.storage.elasticsearch_connector import elasticsearch_connector  # noqa: E402
from infra.storage.mysql_connector import mysql_connector  # noqa: E402
from models.elasticsearch_index.base_index import build_field_types_from_markers  # noqa: E402
from models.elasticsearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2  # noqa: E402
from models.elasticsearch_index.base_index import get_index_name  # noqa: E402
from models.pydantic.video_analysis_request import ShotCard as PydShotCard  # noqa: E402
from services.analysis_video import index_shotcards_to_opensearch  # noqa: E402
from services.video_analysis_db_service import video_analysis_db_service  # noqa: E402


async def _reindex_v2_from_mysql(
    *,
    index_name: str,
    workspace: Optional[str],
    per_history_refresh: bool = False,
    final_refresh: bool = True,
) -> None:
    rows = await video_analysis_db_service.list_all_cards(shot_cards_version="v2", workspace=workspace)
    print(f"[reindex] MySQL v2 卡片行数: {len(rows)} workspace_filter={workspace!r}")

    by_hid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped_err = 0
    for r in rows:
        if r.get("error"):
            skipped_err += 1
            continue
        hid = str(r.get("history_id") or r.get("video_key") or "").strip()
        if not hid:
            continue
        by_hid[hid].append(r)

    print(f"[reindex] 按 history 分组: {len(by_hid)} 个（略过 error 标记行 {skipped_err}）")

    ok_h = fail_h = 0
    for hid in sorted(by_hid.keys()):
        group = by_hid[hid]
        group.sort(key=lambda x: int(x.get("scene_id") or 0))
        keys = [(hid, int(x.get("scene_id") or 0)) for x in group]

        cards: List[PydShotCard] = []
        for x in group:
            try:
                cards.append(PydShotCard(**x))
            except Exception as e:
                print(f"[reindex] 解析卡片失败 {hid} scene={x.get('scene_id')}: {e}")

        if not cards:
            await video_analysis_db_service.update_cards_index_status(
                keys, status="FAILED", error="reindex: 无有效 ShotCard", shot_cards_version="v2"
            )
            fail_h += 1
            continue

        try:
            await video_analysis_db_service.update_cards_index_status(
                keys, status="PENDING", error=None, shot_cards_version="v2"
            )
            await index_shotcards_to_opensearch(
                cards,
                id_prefix=hid,
                refresh=per_history_refresh,
                workspace="v2",
                opensearch_index_name=index_name,
            )
            await video_analysis_db_service.update_cards_index_status(
                keys, status="OK", error=None, shot_cards_version="v2"
            )
            ok_h += 1
            if ok_h % 20 == 0:
                print(f"[reindex] … 已写入 {ok_h} 个 history")
        except Exception as e:
            print(f"[reindex] FAILED history {hid}: {e}")
            await video_analysis_db_service.update_cards_index_status(
                keys, status="FAILED", error=str(e), shot_cards_version="v2"
            )
            fail_h += 1

    if final_refresh:
        client = await elasticsearch_connector.get_client()
        await client.indices.refresh(index=index_name)
        print(f"[reindex] ES indices.refresh({index_name!r})")

    print(f"[reindex] 完成: 成功 history={ok_h} 失败 history={fail_h} → Elasticsearch index={index_name!r}")


async def main() -> None:
    default_index = get_index_name(CarInteriorAnalysisV2)
    parser = argparse.ArgumentParser(description="Create / inspect car_interior_analysis_v2 ES index.")
    parser.add_argument("--index-name", default=default_index, help=f"索引名（默认 {default_index}）")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="索引已存在时是否删除后重建（默认: 是）",
    )
    parser.add_argument(
        "--skip-reindex",
        action="store_true",
        help="建索引后不扫 MySQL 重灌（默认会重灌）",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="仅重灌该 workspace 的历史（不设则全部 v2）",
    )
    args = parser.parse_args()

    index_name = (args.index_name or default_index).strip()
    
    mysql_touched = False
    try:
        await elasticsearch_connector.ensure_init()
        client = await elasticsearch_connector.get_client()

        exists = await client.indices.exists(index=index_name)
        print(f"ES index exists: {exists} name={index_name!r}")

        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }

        print(f"开始在 ES 中创建索引 {index_name!r} (overwrite={args.overwrite}) ...")
        created = await index_manager.create_index(
            model_class=CarInteriorAnalysisV2,
            field_types=None,
            settings=settings,
            overwrite=args.overwrite,
            index_name_override=index_name,
        )
        print(f"ES 索引创建结果: {created}")

        if args.skip_reindex:
            print("\n[skip-reindex] 未扫 MySQL")
            return

        await mysql_connector.ensure_init()
        mysql_touched = True
        print("\n--- 扫表重灌 Elasticsearch ---")
        await _reindex_v2_from_mysql(
            index_name=index_name,
            workspace=args.workspace,
            per_history_refresh=False,
            final_refresh=True,
        )

    finally:
        await elasticsearch_connector.close()
        if mysql_touched:
            await mysql_connector.close()


if __name__ == "__main__":
    asyncio.run(main())
