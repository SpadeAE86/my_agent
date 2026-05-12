"""
网宿 / 非标准中转：文生图 JSON POST，带参考图则为 multipart POST（与 OpenAI images.generate 不同）。
配置见环境变量，勿与 call_gpt_image_utils（OpenAI SDK）混用。
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
from io import BytesIO
from math import gcd
from typing import Optional

import requests
from dotenv import load_dotenv

from infra.logging.logger import logger as log
from utils.obs_utils import upload_to_obs

load_dotenv()

_TEXT2IMG_URL = (os.getenv("GPT_IMAGE_WANGSU_TEXT_URL") or "").strip()
_TEXT2IMG_KEY = (os.getenv("GPT_IMAGE_WANGSU_TEXT_API_KEY") or "").strip()
_IMGEDIT_URL = (os.getenv("GPT_IMAGE_WANGSU_EDIT_URL") or "").strip()
_IMGEDIT_KEY = (os.getenv("GPT_IMAGE_WANGSU_EDIT_API_KEY") or "").strip()

_NO_PROXY = {"http": None, "https": None}
_REQUEST_TIMEOUT = 500


def _ensure_config(*, need_edit: bool) -> None:
    if not _TEXT2IMG_URL or not _TEXT2IMG_KEY:
        raise ValueError(
            "网宿文生图未配置：请设置 GPT_IMAGE_WANGSU_TEXT_URL 与 GPT_IMAGE_WANGSU_TEXT_API_KEY"
        )
    if need_edit and (not _IMGEDIT_URL or not _IMGEDIT_KEY):
        raise ValueError(
            "网宿图生图未配置：请设置 GPT_IMAGE_WANGSU_EDIT_URL 与 GPT_IMAGE_WANGSU_EDIT_API_KEY"
        )


def _size_str_to_aspect_ratio(size: str) -> str:
    s = size.lower().replace("×", "x")
    if "x" not in s:
        return "1:1"
    parts = s.split("x", 1)
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        return "1:1"
    if w <= 0 or h <= 0:
        return "1:1"
    g = gcd(w, h)
    return f"{w // g}:{h // g}"


def _file_tuple_for_reference(img: str):
    if img.startswith(("http://", "https://")):
        r = requests.get(img, timeout=120, proxies=_NO_PROXY)
        r.raise_for_status()
        ct = (r.headers.get("Content-Type") or "image/png").split(";")[0].strip()
        if "/" not in ct:
            ct = "image/png"
        name = img.rstrip("/").rsplit("/", 1)[-1].split("?")[0] or "reference.png"
        return ("image", (name, BytesIO(r.content), ct))
    return ("image", open(img, "rb"))


def _parse_image_url_from_response(result: dict) -> str:
    if "data" not in result:
        raise ValueError(f"无法解析图片返回: {result}")
    item = result["data"][0]
    image_url = item.get("url") or item.get("b64_json")
    if not image_url:
        raise ValueError(f"图片字段不存在: {result}")
    return image_url


def _decode_to_bytes(image_url: str) -> bytes:
    if image_url.startswith("data:image"):
        return base64.b64decode(image_url.split(",", 1)[1])
    if image_url.startswith("http://") or image_url.startswith("https://"):
        r = requests.get(image_url, timeout=120, proxies=_NO_PROXY)
        r.raise_for_status()
        return r.content
    return base64.b64decode(image_url)


async def call_gpt_image_wangsu_edge(
    prompt: str,
    size: str,
    reference_image_list: Optional[list[str]] = None,
    *,
    quality: str = "medium",
) -> str:
    """
    返回可给下游镜像/展示的图片 URL（远端 https，或 base64/非 URL 时上传 OBS 后的公网地址）。
    """
    _ensure_config(need_edit=bool(reference_image_list))
    aspect_ratio = _size_str_to_aspect_ratio(size)
    use_form = bool(reference_image_list)

    def _do_http() -> tuple[str, Optional[bytes]]:
        log.info(f"[WangsuImg] mode={'multipart' if use_form else 'json'} aspect={aspect_ratio} size={size}")

        if use_form:
            files = [_file_tuple_for_reference(img) for img in reference_image_list or []]
            data = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "quality": quality,
                "size": size,
            }
            try:
                resp = requests.post(
                    _IMGEDIT_URL,
                    data=data,
                    files=files,
                    headers={"Authorization": f"Bearer {_IMGEDIT_KEY}"},
                    timeout=_REQUEST_TIMEOUT,
                    proxies=_NO_PROXY,
                )
            finally:
                for _, part in files:
                    fh = part[1] if isinstance(part, tuple) else part
                    if hasattr(fh, "close"):
                        fh.close()
        else:
            payload = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "quality": quality,
                "size": size,
            }
            log.info(f"[WangsuImg] json_payload={json.dumps(payload)[:400]}")
            resp = requests.post(
                _TEXT2IMG_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {_TEXT2IMG_KEY}",
                },
                json=payload,
                timeout=_REQUEST_TIMEOUT,
                proxies=_NO_PROXY,
            )

        if resp.status_code != 200:
            raise ValueError(f"网宿生图 HTTP {resp.status_code}: {resp.text[:800]}")

        result = resp.json()
        log.info(f"[WangsuImg] raw={str(result)[:400]}")
        field = _parse_image_url_from_response(result)

        if field.startswith("http://") or field.startswith("https://"):
            return field, None

        img_bytes = _decode_to_bytes(field)
        return "", img_bytes

    remote_url, raw = await asyncio.to_thread(_do_http)
    if remote_url:
        return remote_url

    if not raw:
        raise ValueError("网宿生图未返回 URL 或图片字节")

    suffix = ".png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:

        def _write():
            with open(tmp_path, "wb") as f:
                f.write(raw)

        await asyncio.to_thread(_write)
        obs_url = await upload_to_obs(tmp_path, obs_prefix="ai_picture/generated_image_wangsu/")
        return obs_url
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    import sys

    prompt = "generate a Japanese animation girl, looks like girls in the image, poster for me, add some Christmas element to her such as present or magical style"
    reference_image_list = ["https://freeuuu.obs.cn-east-3.myhuaweicloud.com/ai_picture/reference_image/1777535181530.jpg"]
    result = asyncio.run(call_gpt_image_wangsu_edge(prompt, size="1440x2560", reference_image_list=reference_image_list))

    print(result[:100])
