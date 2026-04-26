# -*- coding: utf-8 -*-
"""
run_vector_request.py

Execute an OpenSearch DevTools-style request stored in a text file (e.g. vector.txt),
then persist the raw response to JSON for debugging.

Supported input format:
  GET index/_search
  { ...json body... }

Run:
  python -m src.test.run_vector_request
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch_connector import opensearch_connector


INPUT_TXT = Path(__file__).resolve().parent / "vector.txt"
# Persist both:
# - raw response (may be very large)
# - small summary for quick inspection (recommended to open first)
OUTPUT_RAW_JSON = Path(__file__).resolve().parent / "vector_response_raw.json"
OUTPUT_SUMMARY_JSON = Path(__file__).resolve().parent / "vector_response_summary.json"


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
    try:
        body = json.loads(body_text)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON body: {e}") from e

    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")

    return method, path, body


async def main():
    raw = INPUT_TXT.read_text(encoding="utf-8")
    method, path, body = _parse_devtools_request(raw)

    # Extract index from "index/_search" path.
    # Examples:
    #   car_interior_analysis_v2/_search
    #   /car_interior_analysis_v2/_search
    p = path.lstrip("/")
    if "/_search" not in p:
        raise ValueError(f"Only _search supported for now, got path={path!r}")
    index = p.split("/_search", 1)[0].strip("/")
    if not index:
        raise ValueError(f"Failed to parse index from path: {path!r}")

    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()

    try:
        # Always use .search for _search requests; method is informational here.
        resp = await c.search(index=index, body=body)
    finally:
        await opensearch_connector.close()

    hits = ((resp.get("hits") or {}).get("hits") or [])
    top = [{"_id": h.get("_id"), "_score": h.get("_score")} for h in hits[:10]]

    # Detect common hybrid anomalies quickly: duplicated ids and huge negative scores.
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
        "method": method,
        "path": path,
        "index": index,
        "took": resp.get("took"),
        "timed_out": resp.get("timed_out"),
        "hits_total": (resp.get("hits") or {}).get("total"),
        "max_score": (resp.get("hits") or {}).get("max_score"),
        "top_hits": top,
        "anomaly": {"huge_negative_scores": neg, "duplicate_ids": dups},
    }

    OUTPUT_RAW_JSON.write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote raw: {OUTPUT_RAW_JSON}")
    print(f"Wrote summary: {OUTPUT_SUMMARY_JSON}")
    print("Top hits:", json.dumps(top, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())

