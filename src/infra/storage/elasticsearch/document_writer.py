from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Type

from infra.storage.elasticsearch_connector import elasticsearch_connector
from infra.logging.logger import logger as log
from models.elasticsearch_index.base_index import BaseIndex, get_index_name

# Elasticsearch bulk setting: batch size + retries
_BULK_DOCS_PER_CHUNK = 20
_BULK_MAX_ATTEMPTS = 5
_BULK_RETRY_BASE_SEC = 1.0
_BULK_INTER_CHUNK_SLEEP_SEC = 0.08


def _is_retryable_bulk_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, asyncio.TimeoutError)):
        return True
    try:
        from aiohttp import ClientError
        if isinstance(exc, ClientError):
            return True
    except ImportError:
        pass
    s = str(exc).lower()
    if any(
        x in s
        for x in (
            "connection reset",
            "connection aborted",
            "broken pipe",
            "timeout",
            "errno",
            "ssl",
            "handshake",
        )
    ):
        return True
    try:
        from elasticsearch.exceptions import TransportError
        if isinstance(exc, TransportError):
            code = getattr(exc, "status_code", None)
            if code is None or code == 0 or str(code).upper() == "N/A":
                return True
    except Exception:
        pass
    return False


def _bulk_body_for_docs(docs: List[BaseIndex], index_name: str) -> List[Dict[str, Any]]:
    body: List[Dict[str, Any]] = []
    for d in docs:
        doc = d.model_dump(exclude_none=True)
        doc_id = doc.get("id")
        action: Dict[str, Any] = {"index": {"_index": index_name}}
        if doc_id:
            action["index"]["_id"] = doc_id
        body.append(action)
        body.append(doc)
    return body


async def _bulk_single_request(
    client: Any,
    body: List[Dict[str, Any]],
    *,
    refresh: bool,
) -> Dict[str, Any]:
    last: Optional[BaseException] = None
    for attempt in range(_BULK_MAX_ATTEMPTS):
        try:
            # Use operations=body for ES 8.x/9.x compatibility
            return await client.bulk(operations=body, refresh=refresh)
        except BaseException as e:
            last = e
            if attempt + 1 >= _BULK_MAX_ATTEMPTS or not _is_retryable_bulk_error(e):
                log.error("es_bulk_index: giving up after %s attempts: %s", attempt + 1, e)
                raise
            wait = _BULK_RETRY_BASE_SEC * (2**attempt)
            log.warning(
                "es_bulk_index: attempt %s/%s failed (%s); retry in %.1fs",
                attempt + 1,
                _BULK_MAX_ATTEMPTS,
                e,
                wait,
            )
            await asyncio.sleep(wait)
    assert last is not None
    raise last


async def bulk_index(
    model_class: Type[BaseIndex],
    docs: List[BaseIndex],
    *,
    refresh: bool = False,
    index_name_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Bulk index documents into Elasticsearch.
    """
    if not docs:
        return {"success": True, "items": 0}

    await elasticsearch_connector.ensure_init()
    client = await elasticsearch_connector.get_client()
    index_name = (index_name_override or "").strip() or get_index_name(model_class)

    any_errors = False
    last_resp: Dict[str, Any] = {}

    n = len(docs)
    if n <= _BULK_DOCS_PER_CHUNK:
        body = _bulk_body_for_docs(docs, index_name)
        try:
            last_resp = await _bulk_single_request(client, body, refresh=refresh)
            any_errors = bool(last_resp.get("errors"))
        except Exception as e:
            log.error(f"es_bulk_index failed: {e}")
            raise
    else:
        for i in range(0, n, _BULK_DOCS_PER_CHUNK):
            chunk = docs[i : i + _BULK_DOCS_PER_CHUNK]
            is_last = i + _BULK_DOCS_PER_CHUNK >= n
            body = _bulk_body_for_docs(chunk, index_name)
            try:
                last_resp = await _bulk_single_request(
                    client,
                    body,
                    refresh=refresh if is_last else False,
                )
            except Exception as e:
                log.error(f"es_bulk_index failed at offset {i}: {e}")
                raise
            if last_resp.get("errors"):
                any_errors = True
                log.error(f"es_bulk_index errors in chunk @{i}: {last_resp}")
            if not is_last:
                await asyncio.sleep(_BULK_INTER_CHUNK_SLEEP_SEC)

        if refresh and n > _BULK_DOCS_PER_CHUNK:
            try:
                await client.indices.refresh(index=index_name)
            except Exception as e:
                log.warning("es_bulk_index: indices.refresh failed (non-fatal): %s", e)

    if last_resp.get("errors"):
        any_errors = True

    if any_errors:
        log.error(f"es_bulk_index errors: {last_resp}")
    return {"success": not any_errors, "response": last_resp}
