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


async def main():
    # 1) Create index (overwrite)
    settings = {"index": {"knn": True}, "number_of_shards": 1, "number_of_replicas": 0}
    await index_manager.create_index(TinyHybridDoc, settings=settings, overwrite=True)

    # 2) Index tiny docs (fixed vectors)
    docs = [
        TinyHybridDoc(id="d1", text="地库 一键 泊车", vec=[1.0, 0.0, 0.0, 0.0]),
        TinyHybridDoc(id="d2", text="商场 停车 泊车 辅助", vec=[0.9, 0.1, 0.0, 0.0]),
        TinyHybridDoc(id="d3", text="雨夜 模式 盲区 补盲", vec=[0.0, 1.0, 0.0, 0.0]),
        TinyHybridDoc(id="d4", text="充电 低温 电池", vec=[0.0, 0.0, 1.0, 0.0]),
    ]
    bulk_resp = await bulk_index(TinyHybridDoc, docs, refresh=True)

    # 3) Run hybrid query directly (no embeddings, we provide query vector)
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
                            "type": "best_fields",
                            "boost": 0.5,
                        }
                    },
                    {
                        "knn": {
                            "vec": {
                                "vector": query_vec,
                                "k": 10,
                                "boost": 0.5,
                            }
                        }
                    },
                ]
            }
        },
        "_source": {"excludes": ["vec"]},
        "profile": True,
    }

    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()
    try:
        resp = await c.search(index=idx, body=body)
    finally:
        await opensearch_connector.close()

    hits = ((resp.get("hits") or {}).get("hits") or [])
    top = [{"_id": h.get("_id"), "_score": h.get("_score")} for h in hits[:10]]

    out = {
        "index": idx,
        "bulk_success": bulk_resp.get("success"),
        "bulk_errors": bool((bulk_resp.get("response") or {}).get("errors")),
        "query": {"text": query_text, "vec": query_vec},
        "hits_total": (resp.get("hits") or {}).get("total"),
        "max_score": (resp.get("hits") or {}).get("max_score"),
        "top_hits": top,
        "anomaly": _detect_anomaly(hits),
        # keep raw response for deep debugging
        "raw": resp,
    }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {OUT}")
    print("Top hits:", json.dumps(top, ensure_ascii=False))
    print("Anomaly:", json.dumps(out["anomaly"], ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())

