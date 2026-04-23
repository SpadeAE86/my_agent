from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx

from infra.logging.logger import logger as log
from utils.obs_utils import upload_to_obs


def _guess_ext_from_url(url: str) -> str:
    try:
        path = urlparse(url).path
        _, ext = os.path.splitext(path)
        ext = (ext or "").lower()
        if ext and len(ext) <= 8:
            return ext
    except Exception:
        pass
    return ""


async def mirror_remote_url_to_obs(
    remote_url: str,
    *,
    obs_prefix: str,
    timeout_s: int = 30,
) -> str:
    """
    Download a remote URL to a temp file, then upload to OBS and return the OBS public URL.
    """
    ext = _guess_ext_from_url(remote_url)
    fd, tmp_path = tempfile.mkstemp(suffix=ext or ".bin")
    os.close(fd)

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_s) as client:
            r = await client.get(remote_url)
            r.raise_for_status()
            content = r.content

        def _write():
            with open(tmp_path, "wb") as f:
                f.write(content)

        await asyncio.to_thread(_write)
        obs_url = await upload_to_obs(tmp_path, obs_prefix=obs_prefix)
        return obs_url
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def is_obs_url(url: str) -> bool:
    if not url:
        return False
    return ("obs.cn-east-3.myhuaweicloud.com" in url) or ("obs.freeuuu.com" in url)

