import asyncio
import hashlib
import os
import time
from typing import List, Optional

import tos
from tos.exceptions import TosClientError, TosServerError

from config.config import ENV, MY_CONFIG, VPC
from exceptions.infra import ServiceException
from infra.logging.logger import logger as log
from utils.cache_utils import get_from_cache, set_to_cache
import re
import urllib

def decode_chinese_url(url):
    """
    将URL中的中文编码部分解码回中文
    只处理连续3个百分号编码（对应中文字符）
    """
    def decode_match(match):
        try:
            return urllib.parse.unquote(match.group())
        except:
            return match.group()

    # 匹配连续3个百分号编码（对应一个中文字符）
    pattern1 = r'%[A-Fa-f0-9]{2}%[A-Fa-f0-9]{2}%[A-Fa-f0-9]{2}'
    chinese_url = re.sub(pattern1, decode_match, url)
    pattern2 = r'%[A-Fa-f0-9]{2}'
    fixed_url = re.sub(pattern2, decode_match, chinese_url)

    # 尝试修复由于上游/前端错误将 UTF-8 当作 Latin-1 解码导致的乱码现象
    try:
        fixed_url = fixed_url.encode('latin1').decode('utf-8')
    except Exception:
        pass

    return fixed_url

# === Volcengine TOS 配置 ===
AK = os.getenv("VOLC_TOS_AK") or MY_CONFIG.get("storage", {}).get("volcengine", {}).get("access_key") or "YOUR_VOLCENGINE_AK"
SK = os.getenv("VOLC_TOS_SK") or MY_CONFIG.get("storage", {}).get("volcengine", {}).get("secret_key") or "YOUR_VOLCENGINE_SK"
ENDPOINT = os.getenv("VOLC_TOS_ENDPOINT") or MY_CONFIG.get("storage", {}).get("volcengine", {}).get("endpoint") or "tos-s3-cn-beijing.volces.com"
REGION = os.getenv("VOLC_TOS_REGION") or MY_CONFIG.get("storage", {}).get("volcengine", {}).get("region") or "cn-beijing"
BUCKET_NAME = os.getenv("VOLC_TOS_BUCKET") or MY_CONFIG.get("storage", {}).get("volcengine", {}).get("bucket") or "your-volc-bucket"

# 虚拟主机风格的 Base URL
TOS_BASE_URL = f"https://{BUCKET_NAME}.{ENDPOINT.replace('https://', '').replace('http://', '')}"

_tos_client = None
def get_tos_client():
    global _tos_client
    if _tos_client is None:
        _tos_client = tos.TosClientV2(AK, SK, ENDPOINT, REGION)
    return _tos_client

tos_audio_prefix = f"aigc/aigc_{MY_CONFIG.get('env', 'local')}/"

async def download_resource(path_list, output_dir=None):
    decode_path_list = [decode_chinese_url(path) for path in path_list]
    if not output_dir:
        vpc_prefix = VPC + "/"
        path_list = [vpc_prefix + p for p in decode_path_list]
    else:
        download_task = [download_from_tos(p, output_dir) for p in decode_path_list]
        path_list = await asyncio.gather(*download_task)
    return path_list

async def upload_audio(audio_path, project_id="test"):
    log.info(f"开始上传音频到 TOS: {audio_path}")
    if not audio_path:
        return ""
    tos_audio_path = await upload_to_tos(audio_path, tos_audio_prefix, str(project_id))
    tos_audio_path = tos_audio_path.replace("\\", "/")
    return tos_audio_path

async def upload_to_tos(filename: str, tos_prefix: str = "ai_picture/mark/demo/frames_test/", project_id=None) -> str:
    if project_id is not None:
        tos_prefix = tos_prefix + str(project_id)
    fname = os.path.basename(filename)
    tos_key = os.path.join(tos_prefix, fname).replace("\\", "/")

    try:
        client = get_tos_client()
        await asyncio.to_thread(
            client.put_object_from_file,
            bucket=BUCKET_NAME,
            key=tos_key,
            file_path=filename
        )
        return f"{TOS_BASE_URL}/{tos_key}"
    except (TosClientError, TosServerError) as e:
        raise ServiceException(code=461, message=f"TOS 上传异常: {e.message}")
    except Exception as e:
        raise ServiceException(code=457, message=f"TOS 上传异常，请检查 {filename} 文件是否存在", data=str(e))

def sha256_file(filename, chunk_size=512):
    m = hashlib.sha256()
    with open(filename, 'rb') as f:
        while True:
            b = f.read(chunk_size)
            if len(b) == 0:
                break
            m.update(b)
    return m.hexdigest()

async def download_from_tos(path, save_dir: str = "./tos_video") -> str:
    """
    从 TOS 下载文件并保存在本地指定目录，使用 diskcache 管理本地缓存。
    """
    import uuid
    filename = os.path.basename(path)
    temp_local_name = f"{uuid.uuid4().hex}_{filename}"
    local_path = os.path.join(save_dir, temp_local_name)
    fn, ext = os.path.splitext(filename)
    if not ext.lower() in [".mp4", ".mov", ".avi", ".wav", ".mp3", ".MP4", ".qt", ".png", ".jpg", ".jpeg", ".webp"]:
        raise ServiceException(code=461, message=f"{filename} 文件不是合法格式")
    os.makedirs(save_dir, exist_ok=True)

    try:
        # 查询缓存
        cached_path = await asyncio.to_thread(get_from_cache, path, True)
        if cached_path:
            return str(cached_path)

        # 缓存未命中，开始下载
        start = time.time()
        log.info(f"{fn} 开始从 TOS 下载")

        client = get_tos_client()
        await asyncio.to_thread(
            client.get_object_to_file,
            bucket=BUCKET_NAME,
            key=path,
            file_path=local_path
        )
        d = time.time() - start

        log.info(f"{fn} 从 TOS 下载任务执行了 {d:.2f} 秒")
        log.info(f"{local_path}:{sha256_file(local_path)}")

        # 写入缓存并清理临时文件
        def write_cache_and_clean():
            with open(local_path, 'rb') as f:
                set_to_cache(path, f)
            try:
                os.remove(local_path)
            except Exception as cleanup_err:
                log.warning(f"Failed to delete temp file {local_path}: {cleanup_err}")
            return get_from_cache(path, as_path=True)

        final_cached_path = await asyncio.to_thread(write_cache_and_clean)
        return str(final_cached_path) if final_cached_path else local_path
    except (TosClientError, TosServerError) as e:
        raise ServiceException(code=460, message=f"TOS 下载异常: {e.message}，文件路径: {path}")
    except Exception as e:
        raise ServiceException(code=440, message=f"TOS 下载异常，请检查 {filename} 文件是否存在", data=str(e))

async def batch_upload_to_tos(
    file_paths: List[str],
    tos_key_prefix: str,
    max_concurrency: int = 5,
) -> List[str]:
    sem = asyncio.Semaphore(max_concurrency)

    async def _upload(path: str):
        async with sem:
            url = await upload_to_tos(path, tos_key_prefix)
            return url

    tasks = [_upload(p) for p in file_paths]
    tos_keys = await asyncio.gather(*tasks)
    return tos_keys

def tos_key_exists(tos_path: str) -> bool:
    """
    判断 TOS 对象是否存在
    """
    try:
        client = get_tos_client()
        client.head_object(bucket=BUCKET_NAME, key=tos_path)
        return True
    except TosServerError as e:
        if e.status_code == 404:
            return False
        log.exception(f"TOS headObject 异常 (Status Code: {e.status_code}): {e.message}")
        return False
    except Exception as e:
        log.exception(f"TOS 路径 {tos_path} 异常: {e}")
        return False
