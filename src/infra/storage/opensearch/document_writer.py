from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from infra.storage.opensearch_connector import opensearch_connector
from infra.logging.logger import logger as log
from models.pydantic.opensearch_index.base_index import BaseIndex, get_index_name


async def bulk_index(
    model_class: Type[BaseIndex],
    docs: List[BaseIndex],
    *,
    refresh: bool = False,
) -> Dict[str, Any]:
    """
    Bulk index documents into OpenSearch.

    This keeps it intentionally simple: build a newline-delimited bulk body list:
      {"index": {"_index": "...", "_id": "..."}}
      {...doc...}
    """
    if not docs:
        return {"success": True, "items": 0}

    await opensearch_connector.ensure_init()
    client = await opensearch_connector.get_client()
    index_name = get_index_name(model_class)

    body: List[Dict[str, Any]] = []
    for d in docs:
        doc = d.model_dump(exclude_none=True)
        doc_id = doc.get("id")
        action: Dict[str, Any] = {"index": {"_index": index_name}}
        if doc_id:
            action["index"]["_id"] = doc_id
        body.append(action)
        body.append(doc)

    try:
        resp = await client.bulk(body=body, refresh=refresh)
        errors = bool(resp.get("errors"))
        if errors:
            log.error(f"bulk_index errors: {resp}")
        return {"success": not errors, "response": resp}
    except Exception as e:
        log.error(f"bulk_index failed: {e}")
        raise

