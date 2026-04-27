# -*- coding: utf-8 -*-
"""
run_vector_request.py

Run a hybrid search (BM25 + single KNN) from in-file inputs:
- Define a list of query strings
- Embed each query (SentenceTransformer)
- Execute hybrid search against a single vector field + a few text fields
- Persist raw + summary JSON per query

Run:
  python -m src.test.run_vector_request
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch_connector import opensearch_connector


# === Configure here ===
INDEX = "car_interior_analysis_v2"
SEARCH_PIPELINE: Optional[str] = "nlp-search-pipeline"  # set None to disable

VECTOR_FIELD = "marketing_phrases_vector"  # pick ONE vector field
TEXT_FIELDS = ["description", "marketing_phrases", "function_selling_points", "scene_location"]

BM25_BOOST = 0.5
KNN_BOOST = 0.5
SIZE = 10

# Define your input queries here.
INPUTS: List[str] = [
    "地库 一键泊车",
    "雨夜 补盲",
    "冰雪 爬坡",
]

OUT_DIR = Path(__file__).resolve().parent / "vector_requests_outputs"


async def main():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is required for embedding queries in this script. "
            "Install it in your env (pip/conda) and retry."
        ) from e

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()

    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        for q in [s.strip() for s in (INPUTS or []) if s and s.strip()]:
            qv = model.encode(q).tolist()

            body: Dict[str, Any] = {
                "size": SIZE,
                "_source": {"excludes": ["*vector*"]},
                "query": {
                    "hybrid": {
                        "queries": [
                            {
                                "multi_match": {
                                    "query": q,
                                    "fields": TEXT_FIELDS,
                                    "type": "best_fields",
                                    "boost": BM25_BOOST,
                                }
                            },
                            {
                                "knn": {
                                    VECTOR_FIELD: {
                                        "vector": qv,
                                        "k": SIZE,
                                        "boost": KNN_BOOST,
                                    }
                                }
                            },
                        ]
                    }
                },
            }

            params = {"search_pipeline": SEARCH_PIPELINE} if SEARCH_PIPELINE else None
            resp = await c.search(index=INDEX, body=body, params=params)

            hits = ((resp.get("hits") or {}).get("hits") or [])
            top = [{"_id": h.get("_id"), "_score": h.get("_score")} for h in hits[:10]]

            # anomaly detection
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

            summary = {
                "index": INDEX,
                "search_pipeline": SEARCH_PIPELINE,
                "query": q,
                "vector_field": VECTOR_FIELD,
                "text_fields": TEXT_FIELDS,
                "took": resp.get("took"),
                "timed_out": resp.get("timed_out"),
                "hits_total": (resp.get("hits") or {}).get("total"),
                "max_score": (resp.get("hits") or {}).get("max_score"),
                "top_hits": top,
                "anomaly": {"huge_negative_scores": neg, "duplicate_ids": dups},
            }

            safe = hashlib.sha1(q.encode("utf-8")).hexdigest()[:10]
            raw_path = OUT_DIR / f"{safe}_raw.json"
            sum_path = OUT_DIR / f"{safe}_summary.json"
            raw_path.write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")
            sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print("Wrote:", sum_path)
            print("Top hits:", json.dumps(top, ensure_ascii=False))
    finally:
        await opensearch_connector.close()


if __name__ == "__main__":
    asyncio.run(main())

