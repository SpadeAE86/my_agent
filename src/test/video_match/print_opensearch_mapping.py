"""
print_opensearch_mapping.py

PyCharm-friendly helper:
- Print index mapping for debugging field types (e.g. topic/product_status_scene)

Run:
  python -m src.test.print_opensearch_mapping
"""

from __future__ import annotations

import asyncio
import json
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from infra.storage.opensearch_connector import opensearch_connector  # noqa: E402


INDEX = "car_interior_analysis_v2"
ONLY_FIELDS = ["topic", "product_status_scene", "movement", "video_usage"]


def _pick_field(mapping: dict, field: str) -> dict:
    try:
        props = (
            mapping.get(INDEX, {})
            .get("mappings", {})
            .get("properties", {})
        )
        return props.get(field) or {}
    except Exception:
        return {}


async def main() -> None:
    await opensearch_connector.ensure_init()
    c = await opensearch_connector.get_client()
    m = await c.indices.get_mapping(index=INDEX)
    # `m` is {index_name: {mappings: ...}}
    out = {f: _pick_field(m, f) for f in ONLY_FIELDS}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    await opensearch_connector.close()


if __name__ == "__main__":
    asyncio.run(main())

