"""
summarize_vector_profile.py

Take a large OpenSearch response JSON that includes `profile`, and generate a small
summary JSON that is easier to read.

Input:
  - src/test/vector_response_raw.json (default)

Output:
  - src/test/vector_profile_summary.json

Run:
  python -m src.test.summarize_vector_profile
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


RAW = Path(__file__).resolve().parent / "vector_response_raw.json"
OUT = Path(__file__).resolve().parent / "vector_profile_summary.json"


def _get(d: Any, path: List[Any], default=None):
    cur = d
    for p in path:
        try:
            if isinstance(p, int):
                cur = cur[p]
            else:
                cur = cur.get(p)
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def _flatten_query_tree(node: Dict[str, Any], out: List[Dict[str, Any]], depth: int = 0):
    desc = node.get("description")
    t = node.get("time")
    breakdown = node.get("breakdown") or {}
    out.append(
        {
            "depth": depth,
            "type": node.get("type"),
            "description": desc,
            "time": t,
            "breakdown_top": {k: breakdown.get(k) for k in list(breakdown.keys())[:8]},
        }
    )
    for child in (node.get("children") or []):
        if isinstance(child, dict):
            _flatten_query_tree(child, out, depth + 1)


def main():
    resp = json.loads(RAW.read_text(encoding="utf-8"))

    hits = _get(resp, ["hits", "hits"], default=[]) or []
    top_hits = [{"_id": h.get("_id"), "_score": h.get("_score")} for h in hits[:20]]

    # anomaly detection
    seen = {}
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

    profile = resp.get("profile") or {}
    shards = profile.get("shards") or []
    shard_summaries = []
    flat_queries: List[Dict[str, Any]] = []

    for s in shards:
        shard_id = s.get("id")
        searches = s.get("searches") or []
        for si, search in enumerate(searches):
            qnodes = search.get("query") or []
            # each qnode is a dict containing type/description/time/children
            for q in qnodes:
                if isinstance(q, dict):
                    _flatten_query_tree(q, flat_queries, depth=0)
        shard_summaries.append({"id": shard_id, "searches": len(searches)})

    summary = {
        "took": resp.get("took"),
        "timed_out": resp.get("timed_out"),
        "hits_total": _get(resp, ["hits", "total"]),
        "max_score": _get(resp, ["hits", "max_score"]),
        "top_hits": top_hits,
        "anomaly": {"huge_negative_scores": neg[:20], "duplicate_ids": dups[:20]},
        "profile": {
            "shards": shard_summaries,
            "flattened_query_nodes_count": len(flat_queries),
            "flattened_query_nodes_head": flat_queries[:80],
        },
    }

    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()

