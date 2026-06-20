# -*- coding: utf-8 -*-
"""
build_car_interior_index_v2.py

在指定 OpenSearch 环境（默认 test）下创建 ``car_interior_analysis_v2`` 映射，
与 Pydantic 模型 ``CarInteriorAnalysisV2`` 一致（含 knn_vector）。**默认在重建索引后
扫 MySQL ``video_analysis_shot_cards_v2``，按 history 分组重新 embedding 并 bulk 写入
OpenSearch**，并更新 ``os_index_status``（与 ``POST /video-analysis/reindex`` 同源逻辑）。

云端若曾用旧脚本/手工建索引，可能出现 ``marketing_phrases_vector`` 等字段不是
``knn_vector`` 的情况，与本地/模型不一致。可先用本脚本 **仅打印 mapping** 对照。

**连接哪个集群**：由 ``config.yml`` 的 ``env``（或环境变量 ``env``）决定 ``opensearch.{env}``。
本脚本在导入配置前若发现未设置 ``env``，会默认 ``env=test``，便于连测试集群。

Run::

  # 测试集群：打印向量相关字段类型
  python -m src.test.build_car_interior_index_v2 --print-mapping-only

  # 指定解释器（Windows conda 示例）
  C:\\Users\\...\\.conda\\envs\\py312\\python.exe -m src.test.build_car_interior_index_v2 --env test --print-mapping-only

  # 删除并重建 + 扫表重灌（默认行为）
  python -m src.test.build_car_interior_index_v2 --env test --overwrite

  # 只建空索引，不重灌
  python -m src.test.build_car_interior_index_v2 --skip-reindex

  # 已有正确 mapping，仅扫表重灌
  python -m src.test.build_car_interior_index_v2 --reindex-only

  # 仅重灌某一 workspace 的历史
  python -m src.test.build_car_interior_index_v2 --reindex-only --workspace v2

  # 新索引名（须自行做别名/改配置才能让业务读到）
  python -m src.test.build_car_interior_index_v2 --index-name car_interior_analysis_v2_knn --overwrite
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
    """Must run before importing config (ENV / opensearch 小节在 import 时定型)."""
    args = sys.argv[1:]
    if "--env" in args:
        ix = args.index("--env")
        if ix + 1 < len(args):
            os.environ["env"] = args[ix + 1].strip()
            return
    if not (os.getenv("env") or "").strip():
        os.environ["env"] = "test"


_apply_env_from_argv()

from config.config import ENV, MY_CONFIG  # noqa: E402
from infra.storage.opensearch.create_index import index_manager  # noqa: E402
from infra.storage.opensearch_connector import opensearch_connector  # noqa: E402
from infra.storage.mysql_connector import mysql_connector  # noqa: E402
from models.pydantic.opensearch_index.base_index import build_field_types_from_markers  # noqa: E402
from models.pydantic.opensearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2  # noqa: E402
from models.pydantic.opensearch_index.base_index import get_index_name  # noqa: E402
from models.pydantic.video_analysis_request import ShotCard as PydShotCard  # noqa: E402
from services.video_match_services.analysis_video import index_shotcards_to_opensearch  # noqa: E402
from services.video_match_services.video_analysis_db_service import video_analysis_db_service  # noqa: E402


def _walk_props(props: dict, prefix: str = "") -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    if not isinstance(props, dict):
        return out
    for name, spec in props.items():
        if not isinstance(spec, dict):
            continue
        path = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
        out.append((path, spec))
        nested = spec.get("properties")
        if isinstance(nested, dict):
            out.extend(_walk_props(nested, path))
    return out


def _vectorish_mapping_summary(mapping_response: dict, index_key: str) -> dict[str, str]:
    summary: dict[str, str] = {}
    try:
        idx = mapping_response.get(index_key)
        if not isinstance(idx, dict) and mapping_response:
            idx = next(iter(mapping_response.values()))
        m = (idx or {}).get("mappings") or {}
        props = m.get("properties")
        if not isinstance(props, dict) and isinstance(m.get("_doc"), dict):
            props = m["_doc"].get("properties")
        if not isinstance(props, dict):
            return summary
        for path, spec in _walk_props(props):
            low = path.lower()
            if "vector" in low or spec.get("type") == "knn_vector":
                summary[path] = str(spec.get("type") or "?")
    except Exception:
        pass
    return summary


async def _print_mapping(client, index_name: str, dump_full: bool) -> None:
    try:
        raw = await client.indices.get_mapping(index=index_name)
    except Exception as e:
        print(f"[mapping] index={index_name!r} get_mapping failed: {e}")
        return
    if dump_full:
        print(json.dumps(raw, ensure_ascii=False, indent=2))
        return

    key = index_name if index_name in raw else next(iter(raw.keys()), index_name)
    vec = _vectorish_mapping_summary(raw, key)
    print(f"\n--- 向量相关字段 (index key={key!r}, ENV={ENV}) ---")
    for k in sorted(vec.keys()):
        print(f"  {k}: {vec[k]}")
    if not vec:
        print("  (none — 索引可能不存在或无 vector 字段)")
    knn_expected = sorted(
        k
        for k, v in build_field_types_from_markers(CarInteriorAnalysisV2).items()
        if v.get("type") == "knn_vector"
    )
    print("\n模型期望的 knn_vector 字段名:", knn_expected)


def _print_cluster_banner() -> None:
    cfg = (MY_CONFIG.get("opensearch") or {}).get(ENV) or {}
    print(
        f"OpenSearch: ENV={ENV!r} → host={cfg.get('host')} port={cfg.get('port')} "
        f"use_ssl={cfg.get('use_ssl')} verify_certs={cfg.get('verify_certs')}"
    )
    dbc = (MY_CONFIG.get("mysql") or {}).get(ENV) or {}
    if dbc:
        print(
            f"MySQL:      ENV={ENV!r} → {dbc.get('host')}:{dbc.get('port')} / {dbc.get('database')}"
        )
    print()


async def _reindex_v2_from_mysql(
    *,
    index_name: str,
    workspace: Optional[str],
    per_history_refresh: bool = False,
    final_refresh: bool = True,
) -> None:
    """
    扫 v2 分镜表 → 按 history_id（video_key）分组 → index_shotcards_to_opensearch（写指定索引）。
    """
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

    print(
        f"[reindex] 按 history 分组: {len(by_hid)} 个（略过 error 标记行 {skipped_err}）"
    )

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
        client = await opensearch_connector.get_client()
        await client.indices.refresh(index=index_name)
        print(f"[reindex] indices.refresh({index_name!r})")

    print(f"[reindex] 完成: 成功 history={ok_h} 失败 history={fail_h} → OpenSearch index={index_name!r}")


async def main() -> None:
    default_index = get_index_name(CarInteriorAnalysisV2)
    parser = argparse.ArgumentParser(description="Create / inspect car_interior_analysis_v2 OpenSearch index.")
    parser.add_argument(
        "--env",
        default=None,
        help="与 config.yml 的 opensearch.{env} / mysql.{env} 对应（导入前已由 argv 解析）",
    )
    parser.add_argument("--index-name", default=default_index, help=f"索引名（默认 {default_index}）")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="索引已存在时是否删除后重建（默认: 是）",
    )
    parser.add_argument(
        "--print-mapping-only",
        action="store_true",
        help="只连接并打印 mapping，不创建索引、不重灌",
    )
    parser.add_argument(
        "--print-mapping",
        action="store_true",
        help="创建前若索引已存在，先打印向量字段摘要",
    )
    parser.add_argument(
        "--full-mapping-json",
        action="store_true",
        help="与 --print-mapping-only（或打印时）输出完整 get_mapping JSON",
    )
    parser.add_argument(
        "--skip-reindex",
        action="store_true",
        help="建索引后不扫 MySQL 重灌（默认会重灌）",
    )
    parser.add_argument(
        "--reindex-only",
        action="store_true",
        help="不创建/删除索引，仅扫表按当前索引名重灌（索引须已存在且 mapping 正确）",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="仅重灌该 workspace 的历史（传入 list_all_cards；不设则全部 v2）",
    )
    parser.add_argument(
        "--per-history-refresh",
        action="store_true",
        help="每个 history bulk 后 refresh（慢；默认只最后在索引上 refresh 一次）",
    )
    args = parser.parse_args()

    _print_cluster_banner()
    index_name = (args.index_name or default_index).strip()
    canonical = get_index_name(CarInteriorAnalysisV2)
    if index_name != canonical:
        print(
            f"[warn] 目标索引 {index_name!r} ≠ 模型默认 {canonical!r}；"
            "业务侧 script_match / 搜索若写死索引名，需做别名或改配置。\n"
        )

    mysql_touched = False
    try:
        await opensearch_connector.ensure_init()
        client = await opensearch_connector.get_client()

        exists = await client.indices.exists(index=index_name)
        print(f"index exists: {exists} name={index_name!r}")

        if args.print_mapping_only:
            if not exists:
                print("(索引不存在，get_mapping 可能报错；可先 --overwrite 建空索引)")
            await _print_mapping(client, index_name, dump_full=args.full_mapping_json)
            return

        if args.reindex_only:
            if not exists:
                print("[error] --reindex-only 要求索引已存在")
                return
            await mysql_connector.ensure_init()
            mysql_touched = True
            await _reindex_v2_from_mysql(
                index_name=index_name,
                workspace=args.workspace,
                per_history_refresh=args.per_history_refresh,
                final_refresh=True,
            )
            return

        if args.print_mapping and exists:
            await _print_mapping(client, index_name, dump_full=args.full_mapping_json)
            print()

        settings = {
            "index": {"knn": True},
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }

        print(f"开始创建索引 {index_name!r} (overwrite={args.overwrite}) ...")
        created = await index_manager.create_index(
            model_class=CarInteriorAnalysisV2,
            field_types=None,
            settings=settings,
            overwrite=args.overwrite,
            index_name_override=index_name,
        )
        print(f"索引创建结果: {created} （False 表示已存在且未 overwrite）")
        print("\n--- 建库后 mapping 摘要 ---")
        await _print_mapping(client, index_name, dump_full=False)

        if args.skip_reindex:
            print("\n[skip-reindex] 未扫 MySQL")
            return

        await mysql_connector.ensure_init()
        mysql_touched = True
        print("\n--- 扫表重灌 OpenSearch ---")
        await _reindex_v2_from_mysql(
            index_name=index_name,
            workspace=args.workspace,
            per_history_refresh=args.per_history_refresh,
            final_refresh=True,
        )

    finally:
        await opensearch_connector.close()
        if mysql_touched:
            await mysql_connector.close()


if __name__ == "__main__":
    asyncio.run(main())
