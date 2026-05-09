# -*- coding: utf-8 -*-
"""
bootstrap_opensearch_pipelines.py

1) **apply** — PUT every ``*.json`` under ``src/opensearch_pipelines/`` to the cluster.
   Filename (without .json) = pipeline id, file content = PUT body (same as manual curl).

2) **export** — GET named pipelines from the cluster and write ``<name>.json`` into the same dir
   (handy to snapshot what is already on localhost before syncing to cloud).

JSON templates ship in-repo under ``my_agent/src/opensearch_pipelines/``:
  - nlp-search-pipeline (+ q3/q4/q5): hybrid **normalization-processor** (min_max + arithmetic_mean)
  - video-analysis-rrf-probe: **score-ranker** RRF (2 sub-queries equal weights), same role as rrf_pipeline_smoke

``ensure_hybrid_pipeline`` upserts the matching pipeline on every search (including **num_queries==2** → ``nlp-search-pipeline``),
so clusters self-heal; ``apply`` is still useful to pre-seed or align with git-tracked JSON.

Production RRF from ``ensure_rrf_pipeline`` uses dynamic names ``video-analysis-rrf-q{N}-{hash}``; the probe file
is for smoke/ops only — video search still creates pipelines on demand when weights differ.

Run::

  cd my_agent/src
  python -m test.bootstrap_opensearch_pipelines apply
  python -m test.bootstrap_opensearch_pipelines apply --dir opensearch_pipelines

  python -m test.bootstrap_opensearch_pipelines export
  python -m test.bootstrap_opensearch_pipelines export --names nlp-search-pipeline,video-analysis-rrf-probe

  python -m test.bootstrap_opensearch_pipelines apply --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, List, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch_connector import opensearch_connector  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_DIR = os.path.join(SRC_DIR, "opensearch_pipelines")

DEFAULT_EXPORT_NAMES = [
    "nlp-search-pipeline",
    "nlp-search-pipeline-q3",
    "nlp-search-pipeline-q4",
    "nlp-search-pipeline-q5",
    "video-analysis-rrf-probe",
]


def _pipeline_dir(arg: Optional[str]) -> str:
    return os.path.abspath(arg or DEFAULT_DIR)


def _json_files(d: str) -> List[str]:
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            out.append(os.path.join(d, fn))
    return out


async def _http_status(resp: Any) -> int:
    for attr in ("status", "status_code"):
        v = getattr(resp, attr, None)
        if v is not None:
            return int(v)
    return 0


async def _apply_one(client: Any, pipeline_id: str, body: dict, dry_run: bool) -> Tuple[bool, str]:
    if dry_run:
        return True, f"[dry-run] PUT /_search/pipeline/{pipeline_id}"
    try:
        resp = await client.http.put(f"/_search/pipeline/{pipeline_id}", body=body)
        sc = await _http_status(resp)
        if sc not in (200, 201):
            try:
                txt = await resp.text() if hasattr(resp, "text") else str(resp)
            except Exception:
                txt = str(resp)
            return False, f"HTTP {sc}: {txt}"
        return True, "ok"
    except Exception as e:
        return False, repr(e)


async def cmd_apply(d: str, *, dry_run: bool) -> int:
    files = _json_files(d)
    if not files:
        print(f"No .json files under {d!r}")
        return 1
    await opensearch_connector.ensure_init()
    client = await opensearch_connector.get_client()
    ok_n = 0
    for path in files:
        pid = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            body = json.load(f)
        ok, msg = await _apply_one(client, pid, body, dry_run)
        print(f"  {pid}: {msg}")
        if ok:
            ok_n += 1
    print(f"\nDone: {ok_n}/{len(files)} succeeded.")
    return 0 if ok_n == len(files) else 2


async def _get_pipeline(client: Any, name: str) -> Tuple[bool, Any]:
    try:
        resp = await client.http.get(f"/_search/pipeline/{name}")
        sc = await _http_status(resp)
        if sc != 200:
            try:
                txt = await resp.text() if hasattr(resp, "text") else str(resp)
            except Exception:
                txt = str(resp)
            return False, f"HTTP {sc}: {txt}"
        data = await resp.json() if hasattr(resp, "json") else None
        if data is None:
            raw = await resp.text() if hasattr(resp, "text") else str(resp)
            try:
                data = json.loads(raw)
            except Exception:
                return False, raw
        return True, data
    except Exception as e:
        return False, repr(e)


async def cmd_export(d: str, names: List[str], *, dry_run: bool) -> int:
    os.makedirs(d, exist_ok=True)
    await opensearch_connector.ensure_init()
    client = await opensearch_connector.get_client()
    ok_n = 0
    for name in names:
        name = name.strip()
        if not name:
            continue
        ok, data = await _get_pipeline(client, name)
        if not ok:
            print(f"  {name}: GET failed — {data}")
            continue
        path = os.path.join(d, f"{name}.json")
        if dry_run:
            print(f"  [dry-run] would write {path}")
            ok_n += 1
            continue
        # Cluster may wrap body; normalize to pretty pipeline body if possible
        to_dump: Any = data
        if isinstance(data, dict) and len(data) == 1 and name in data:
            to_dump = data[name]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_dump, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"  {name} -> {path}")
        ok_n += 1
    print(f"\nExported {ok_n}/{len(names)} pipelines.")
    return 0 if ok_n else 2


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap / export OpenSearch search pipelines")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_apply = sub.add_parser("apply", help="PUT all JSON files from dir")
    p_apply.add_argument("--dir", default=None, help=f"default: {DEFAULT_DIR}")
    p_apply.add_argument("--dry-run", action="store_true")

    p_exp = sub.add_parser("export", help="GET pipelines by name and save as JSON")
    p_exp.add_argument("--dir", default=None, help=f"default: {DEFAULT_DIR}")
    p_exp.add_argument(
        "--names",
        default=",".join(DEFAULT_EXPORT_NAMES),
        help="Comma-separated pipeline ids (default: bundled templates list)",
    )
    p_exp.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    d = _pipeline_dir(getattr(args, "dir", None))

    if args.cmd == "apply":
        raise SystemExit(asyncio.run(cmd_apply(d, dry_run=bool(args.dry_run))))
    if args.cmd == "export":
        names = [x.strip() for x in str(args.names).split(",")]
        raise SystemExit(asyncio.run(cmd_export(d, names, dry_run=bool(args.dry_run))))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
