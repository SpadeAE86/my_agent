import asyncio
import hashlib
import os
import time
import re
import urllib
from typing import List, Optional

from config.config import ENV, MY_CONFIG, VPC
from exceptions.infra import ServiceException
from infra.logging.logger import logger as log
from utils.cache_utils import get_from_cache, set_to_cache

# === 存储提供商选择 ===
STORAGE_PROVIDER = os.getenv("STORAGE_PROVIDER") or MY_CONFIG.get("storage", {}).get("provider") or "huawei"

if STORAGE_PROVIDER == "volcengine":
    # 动态切换到 Volcano TOS 的实现
    from utils.tos_utils import (
        download_resource,
        upload_audio,
        upload_to_tos as upload_to_obs,
        download_from_tos as download_from_obs,
        batch_upload_to_tos as batch_upload_to_obs,
        tos_key_exists as obs_key_exists,
        TOS_BASE_URL as OBS_BASE_URL,
        decode_chinese_url,
    )
else:
    # 默认使用原版 华为云 OBS 的实现
    from obs import ObsClient

    # === OBS 配置 ===
    AK = os.getenv("HUAWEI_OBS_AK") or MY_CONFIG.get("storage", {}).get("huawei", {}).get("access_key") or "UJDPK31ANIBV0XTEUN5N"
    SK = os.getenv("HUAWEI_OBS_SK") or MY_CONFIG.get("storage", {}).get("huawei", {}).get("secret_key") or "NhQExxv9PUYsvmvGnVReizRksaiHcJdQ6vMMw19d"
    ENDPOINT = os.getenv("HUAWEI_OBS_ENDPOINT") or MY_CONFIG.get("storage", {}).get("huawei", {}).get("endpoint") or "obs.cn-east-3.myhuaweicloud.com"
    BUCKET_NAME = os.getenv("HUAWEI_OBS_BUCKET") or MY_CONFIG.get("storage", {}).get("huawei", {}).get("bucket") or "freeuuu"

    OBS_BASE_URL = f"https://{BUCKET_NAME}.{ENDPOINT}"
    CDN_BASE_URL = "https://obs.freeuuu.com"
    audio_output_dir = "../work"

    _obs_client = None
    def get_obs_client():
        global _obs_client
        if _obs_client is None:
            _obs_client = ObsClient(
                access_key_id=AK,
                secret_access_key=SK,
                server=ENDPOINT
            )
        return _obs_client

    obs_audio_prefix = f"aigc/aigc_{MY_CONFIG.get('env', 'local')}/"

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

        # 尝试修复由于上游/前端错误将 UTF-8 当作 Latin-1 解码导致的乱码现象（如 ç¾é£é¤é¥® 变回 美食餐饮）
        try:
            fixed_url = fixed_url.encode('latin1').decode('utf-8')
        except Exception:
            pass

        return fixed_url

    async def download_resource(path_list, output_dir=None):
        decode_path_list = [decode_chinese_url(path) for path in path_list]  #把中文unicode转换成中文字符串
        if not output_dir:
            vpc_prefix = VPC + "/"
            path_list = [vpc_prefix+p for p in decode_path_list]
        else:
            download_task = [download_from_obs(p, output_dir) for p in decode_path_list]
            path_list = await asyncio.gather(*download_task)
        return path_list

    async def upload_audio(audio_path, project_id="test"):
        log.info(f"开始上传音频到 OBS: {audio_path}")
        if not audio_path:
            return ""
        obs_audio_path = await upload_to_obs(audio_path, obs_audio_prefix, str(project_id))
        obs_audio_path = obs_audio_path.replace("\\", "/")
        return obs_audio_path

    async def upload_to_obs(filename: str, obs_prefix: str = "ai_picture/mark/demo/frames_test/", project_id=None) -> str:
        if project_id is not None:
            obs_prefix = obs_prefix + str(project_id)
        fname = os.path.basename(filename)
        obs_key = os.path.join(obs_prefix, fname).replace("\\", "/")

        try:
            client = get_obs_client()
            resp = await asyncio.to_thread(client.putFile, bucketName=BUCKET_NAME, objectKey=obs_key,
                                           file_path=filename)
            if resp.status < 300:
                return f"{OBS_BASE_URL}/{obs_key}"
            else:
                raise ServiceException(code=461, message=f"obs上传异常，状态码{resp.status}")
        except Exception as e:
            raise ServiceException(code=457, message=f"obs上传异常，请检查{filename}文件是否存在", data=str(e))

    def sha256_file(filename, chunk_size=512):
        m = hashlib.sha256()
        f = open(filename, 'rb')
        while True:
            b = f.read(chunk_size)
            if len(b) == 0:
                break
            m.update(b)
        return m.hexdigest()

    async def download_from_obs(path, save_dir: str = "./obs_video") -> str:
        """
        从 OBS 下载文件并保存在本地指定目录，使用 diskcache 管理本地缓存。
        """
        import uuid
        filename = os.path.basename(path)
        # 给每次下载分配独立的临时文件名，防止同时并发下载时造成文件读写冲突崩溃
        temp_local_name = f"{uuid.uuid4().hex}_{filename}"
        local_path = os.path.join(save_dir, temp_local_name)
        fn, ext = os.path.splitext(filename)
        if not ext.lower() in [".mp4", ".mov", ".avi", ".wav", ".mp3", ".MP4", ".qt"]:
            raise ServiceException(code=461, message=f"{filename}文件不是合法格式")
        os.makedirs(save_dir, exist_ok=True)

        try:
            # 查询 diskcache 缓存
            cached_path = await asyncio.to_thread(get_from_cache, path, True)
            if cached_path:
                return str(cached_path)

            # 缓存未命中，从 OBS 下载
            start = time.time()
            log.info(f"{fn}开始下载")

            client = get_obs_client()
            resp = await asyncio.to_thread(client.getObject, bucketName=BUCKET_NAME, objectKey=path,
                                           downloadPath=local_path)
            d = time.time() - start

            log.info(f"{fn}下载任务执行了{d}秒")
            if resp.status < 300:
                log.debug(f"requestId: {resp.requestId}")
                log.info(f"{fn}下载成功")
                log.info(f"{local_path}:{sha256_file(local_path)}")

                # 写入 diskcache 缓存
                def write_cache_and_clean():
                    with open(local_path, 'rb') as f:
                        set_to_cache(path, f)
                    
                    # 删除临时下载的文件，释放磁盘空间
                    try:
                        os.remove(local_path)
                    except Exception as cleanup_err:
                        log.warning(f"Failed to delete temp file {local_path}: {cleanup_err}")

                    return get_from_cache(path, as_path=True)
                
                final_cached_path = await asyncio.to_thread(write_cache_and_clean)
                return str(final_cached_path) if final_cached_path else local_path
            else:
                raise ServiceException(code=460, message=f"obs下载异常，状态码{resp.status}，文件路径: {path}")
        except ServiceException:
            raise
        except Exception as e:
            raise ServiceException(code=440, message=f"obs下载异常，请检查{filename}文件是否存在", data=str(e))

    async def batch_upload_to_obs(
        file_paths: List[str],
        obs_key_prefix: str,
        max_concurrency: int = 5,
    ) -> List[str]:
        sem = asyncio.Semaphore(max_concurrency)

        async def _upload(path: str):
            async with sem:
                url = await upload_to_obs(path, obs_key_prefix)
                return url

        tasks = [_upload(p) for p in file_paths]
        obs_keys = await asyncio.gather(*tasks)

        return obs_keys

    def obs_key_exists(obs_path: str) -> bool:
        """
        判断 OBS 对象是否存在
        """
        try:
            client = get_obs_client()
            resp = client.headObject(BUCKET_NAME, obs_path)
            return resp.status < 300
        except Exception as e:
            log.exception(f"OBS 路径{obs_path}不存在 异常: {e}")
            return False


# === 始终导出的通用、存储提供商无关的下载函数 ===
async def download_url_to_file(url: str, dest_path: str) -> str:
    """
    将可公网访问的 URL（如 OBS/TOS HTTPS 对象 URL）下载到指定本地路径。
    用于分镜参考帧等图片；不经过扩展名白名单限制。
    """
    import httpx
    from pathlib import Path

    u = decode_chinese_url(str(url or "").strip())
    if not u:
        raise ServiceException(code=440, message="empty download url")
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        resp = await client.get(u)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return str(dest)


if __name__ == "__main__":
    # 手动测试用
    test_paths = [
        "aigc/aigc_local/1998/1998743094727520258/0/video/1765372463420.mp4",
        "aigc/aigc_local/1998/1997943094727520258/0/video/1765372463421.mp4",
    ]
    for path in test_paths:
        exists = obs_key_exists(path)
        print(f"[TEST] obs_path={path}, exists={exists}")
