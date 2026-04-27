# -*- coding: utf-8 -*-
"""
tiny_hybrid_repro.py

Create a tiny index (1 text + 1 knn_vector), index a few docs with fixed vectors,
then run a minimal hybrid query (BM25 + KNN) to see whether the cluster returns
negative huge scores / duplicate _id.

This is meant to answer: "Is hybrid+knn broken in this cluster in general?"

Run:
  python -m src.test.tiny_hybrid_repro
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from pydantic import Field
from typing import Annotated, Optional

from infra.storage.opensearch_connector import opensearch_connector
from infra.storage.opensearch.create_index import index_manager
from infra.storage.opensearch.document_writer import bulk_index
from models.pydantic.opensearch_index.base_index import BaseIndex, get_index_name
from models.pydantic.opensearch_index.markers import Text, Vector


OUT = Path(__file__).resolve().parent / "tiny_hybrid_repro_result.json"


class TinyHybridDoc(BaseIndex):
    class Meta:
        index_name = "tiny_hybrid_repro"

    id: Optional[str] = Field(None)
    text: Annotated[str, Text(1.0, analyzer="standard")] = Field("")
    vec: Annotated[Optional[List[float]], Vector(4, 1.0)] = Field(None)


def _detect_anomaly(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    seen = {}
    dups = []
    neg = []
    for i, h in enumerate(hits[:50]):
        doc_id = h.get("_id")
        score = h.get("_score")
        if doc_id in seen:
            dups.append({"_id": doc_id, "pos": i, "prev_pos": seen[doc_id]})
        else:
            seen[doc_id] = i
        try:
            if score is not None and float(score) < -1e6:
                neg.append({"_id": doc_id, "_score": score, "pos": i})
        except Exception:
            pass
    return {"huge_negative_scores": neg, "duplicate_ids": dups}


async def create_hybrid_pipeline(client):
    """
    创建一个用于分数归一化的 Search Pipeline。
    这是 Hybrid 搜索必须的步骤。
    """
    pipeline_name = "nlp-search-pipeline"
    pipeline_body = {
        "description": "Post-processing pipeline for hybrid search",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},  # 将分数归一化到 0-1
                    "combination": {
                        "technique": "arithmetic_mean",  # 算术平均加权
                        "parameters": {"weights": [0.3, 0.7]}  # 这里的权重对应下面 queries 的顺序
                    }
                }
            }
        ]
    }
    # 显式创建 pipeline
    await client.http.put(f"/_search/pipeline/{pipeline_name}", body=pipeline_body)
    print(f"INFO: Search Pipeline '{pipeline_name}' created/updated.")
    return pipeline_name


async def main():
    # 0) 初始化连接并创建 Pipeline
    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()

    # --- 新增步骤：创建 Pipeline ---
    pipeline_name = await create_hybrid_pipeline(c)

    # 1) Create index (overwrite)
    settings = {"index": {"knn": True}, "number_of_shards": 1, "number_of_replicas": 0}
    await index_manager.create_index(TinyHybridDoc, settings=settings, overwrite=True)

    # 2) Index tiny docs
    docs = [
        TinyHybridDoc(id="d1", text="地库 一键 泊车", vec=[1.0, 0.0, 0.0, 0.0]),
        TinyHybridDoc(id="d2", text="商场 停车 泊车 辅助", vec=[0.9, 0.1, 0.0, 0.0]),
        TinyHybridDoc(id="d3", text="雨夜 模式 盲区 补盲", vec=[0.0, 1.0, 0.0, 0.0]),
        TinyHybridDoc(id="d4", text="充电 低温 电池", vec=[0.0, 0.0, 1.0, 0.0]),
    ]
    bulk_resp = await bulk_index(TinyHybridDoc, docs, refresh=True)

    # 3) Run hybrid query
    idx = get_index_name(TinyHybridDoc)
    query_text = "地库 一键泊车"
    query_vec = [1.0, 0.0, 0.0, 0.0]

    body = {
        "size": 10,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["text"],
                            "type": "best_fields"
                        }
                    },
                    {
                        "knn": {
                            "vec": {
                                "vector": query_vec,
                                "k": 10
                            }
                        }
                    },
                ]
            }
        },
        "_source": {"excludes": ["vec"]},
        # 注意：这里务必删除 "profile": True，Hybrid 查询在某些版本开启 profile 会直接 Crash
    }

    try:
        # --- 关键：在请求参数中带上 search_pipeline ---
        resp = await c.search(
            index=idx,
            body=body,
            params={"search_pipeline": pipeline_name}
        )
    finally:
        await opensearch_connector.close()

    # ... 后续处理逻辑保持不变 ...
    hits = ((resp.get("hits") or {}).get("hits") or [])
    top = [{"_id": h.get("_id"), "_score": h.get("_score")} for h in hits[:10]]
    # (保持原有的打印逻辑)
    print("Top hits:", json.dumps(top, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())

