# -*- coding: utf-8 -*-
"""
hybrid_ablation_runner.py

Goal:
Reproduce and localize the "hybrid negative score + duplicate _id" anomaly by ablation:

We read a DevTools request stored in `vector.txt` (hybrid query), then run 3 variants:

  A) knn_only_single      : hybrid = [knn(marketing_phrases_vector)] only
  B) bm25_plus_single_knn : hybrid = [multi_match, knn(marketing_phrases_vector)]
  C) knn_only_multi       : hybrid = [knn(marketing_phrases_vector), knn(function_selling_points_vector), knn(description_vector)]

Each run writes:
- raw response JSON
- summary JSON (top hits + anomaly detection)
- profile summary JSON (flattened query nodes head)

Outputs:
  src/test/hybrid_ablation_outputs/
    - ablation_report.json
    - <case>_raw.json
    - <case>_summary.json
    - <case>_profile_summary.json

Run:
  python -m src.test.hybrid_ablation_runner
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch_connector import opensearch_connector


INPUT_TXT = Path(__file__).resolve().parent / "vector.txt"
OUT_DIR = Path(__file__).resolve().parent / "hybrid_ablation_outputs"

PROFILE = True
EXPLAIN = False

# Keep responses small for stability
SIZE = 10

_REQ_LINE_RE = re.compile(r"^(GET|POST)\s+([^\s]+)\s*$", re.IGNORECASE)


def _parse_devtools_request(text: str) -> Tuple[str, str, Dict[str, Any]]:
    lines = [ln.rstrip("\n") for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty request file")
    m = _REQ_LINE_RE.match(lines[0])
    if not m:
        raise ValueError(f"Invalid first line (expected 'GET index/_search'): {lines[0]!r}")
    method = m.group(1).upper()
    path = m.group(2)
    body_text = "\n".join(lines[1:])
    body = json.loads(body_text)
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    return method, path, body


def _extract_index_from_path(path: str) -> str:
    p = path.lstrip("/")
    if "/_search" not in p:
        raise ValueError(f"Only _search supported, got path={path!r}")
    idx = p.split("/_search", 1)[0].strip("/")
    if not idx:
        raise ValueError(f"Failed to parse index from path: {path!r}")
    return idx


def _min_hit(h: Dict[str, Any]) -> Dict[str, Any]:
    return {"_id": h.get("_id"), "_score": h.get("_score")}


def _summarize(resp: Dict[str, Any]) -> Dict[str, Any]:
    hits = ((resp.get("hits") or {}).get("hits") or [])
    top = [_min_hit(h) for h in hits[:10]]

    # anomaly detection
    seen: Dict[str, int] = {}
    dups = []
    neg = []
    for i, h in enumerate(hits[:200]):
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

    return {
        "took": resp.get("took"),
        "timed_out": resp.get("timed_out"),
        "hits_total": (resp.get("hits") or {}).get("total"),
        "max_score": (resp.get("hits") or {}).get("max_score"),
        "top_hits": top,
        "anomaly": {"huge_negative_scores": neg[:20], "duplicate_ids": dups[:20]},
    }


def _flatten_query_tree(node: Dict[str, Any], out: List[Dict[str, Any]], depth: int = 0):
    desc = node.get("description")
    breakdown = node.get("breakdown") or {}
    out.append(
        {
            "depth": depth,
            "type": node.get("type"),
            "description": desc,
            "time": node.get("time"),
            "breakdown_top": {k: breakdown.get(k) for k in list(breakdown.keys())[:8]},
        }
    )
    for child in (node.get("children") or []):
        if isinstance(child, dict):
            _flatten_query_tree(child, out, depth + 1)


def _summarize_profile(resp: Dict[str, Any]) -> Dict[str, Any]:
    profile = resp.get("profile") or {}
    shards = profile.get("shards") or []
    shard_summaries = []
    flat: List[Dict[str, Any]] = []
    for s in shards:
        shard_id = s.get("id")
        searches = s.get("searches") or []
        for search in searches:
            qnodes = search.get("query") or []
            for q in qnodes:
                if isinstance(q, dict):
                    _flatten_query_tree(q, flat, depth=0)
        shard_summaries.append({"id": shard_id, "searches": len(searches)})
    return {
        "shards": shard_summaries,
        "flattened_query_nodes_count": len(flat),
        "flattened_query_nodes_head": flat[:120],
    }


def _find_hybrid_queries(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (((body.get("query") or {}).get("hybrid") or {}).get("queries") or [])


def _pick_first_knn(hybrid_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    for q in hybrid_queries:
        if isinstance(q, dict) and "knn" in q:
            return q
    raise ValueError("No knn query found in hybrid.queries")


def _pick_multi_match(hybrid_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    for q in hybrid_queries:
        if isinstance(q, dict) and "multi_match" in q:
            return q
    raise ValueError("No multi_match query found in hybrid.queries")


def _build_case_body(base_body: Dict[str, Any], *, case: str) -> Dict[str, Any]:
    body = json.loads(json.dumps(base_body))  # deep copy (JSON-safe)
    body["size"] = SIZE
    if PROFILE:
        body["profile"] = True
    if EXPLAIN:
        body["explain"] = True

    qlist = _find_hybrid_queries(body)
    if not qlist:
        raise ValueError("Input request does not look like a hybrid query.")

    mm = _pick_multi_match(qlist)
    first_knn = _pick_first_knn(qlist)
    all_knn = [q for q in qlist if isinstance(q, dict) and "knn" in q]

    if case == "knn_only_single":
        new_qlist = [first_knn]
    elif case == "bm25_plus_single_knn":
        new_qlist = [mm, first_knn]
    elif case == "knn_only_multi":
        new_qlist = all_knn
    else:
        raise ValueError(f"Unknown case: {case}")

    body["query"]["hybrid"]["queries"] = new_qlist
    return body


async def _run_one(index: str, body: Dict[str, Any]) -> Dict[str, Any]:
    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()
    return await c.search(index=index, body=body)


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = INPUT_TXT.read_text(encoding="utf-8")
    method, path, base_body = _parse_devtools_request(raw)
    index = _extract_index_from_path(path)

    cases = ["knn_only_single", "bm25_plus_single_knn", "knn_only_multi"]
    report: Dict[str, Any] = {
        "input": {"method": method, "path": path, "index": index, "profile": PROFILE, "explain": EXPLAIN},
        "cases": {},
    }

    try:
        for case in cases:
            body = _build_case_body(base_body, case=case)
            resp = await _run_one(index, body)

            # filenames
            raw_path = OUT_DIR / f"{case}_raw.json"
            sum_path = OUT_DIR / f"{case}_summary.json"
            prof_path = OUT_DIR / f"{case}_profile_summary.json"

            raw_path.write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")
            summary = _summarize(resp)
            sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            prof = _summarize_profile(resp) if PROFILE else {}
            prof_path.write_text(json.dumps(prof, ensure_ascii=False, indent=2), encoding="utf-8")

            report["cases"][case] = {
                "raw": str(raw_path),
                "summary": str(sum_path),
                "profile_summary": str(prof_path),
                "anomaly": summary.get("anomaly"),
                "top_hits": summary.get("top_hits"),
            }

    finally:
        await opensearch_connector.close()

    (OUT_DIR / "ablation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {OUT_DIR / 'ablation_report.json'}")
    print(json.dumps({k: v.get('anomaly') for k, v in report['cases'].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

