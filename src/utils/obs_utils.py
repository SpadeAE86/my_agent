import asyncio
import hashlib
import os
import time
from typing import List, Optional

from obs import ObsClient

from config.config import ENV, MY_CONFIG, VPC
from exceptions.infra import ServiceException
from infra.logging.logger import logger as log
from utils.cache_utils import get_from_cache, set_to_cache

# === OBS 配置 ===
BUCKET_NAME = 'freeuuu'
OBS_BASE_URL = 'https://freeuuu.obs.cn-east-3.myhuaweicloud.com'
CDN_BASE_URL = "https://obs.freeuuu.com"
audio_output_dir = "../work"
obs_client = ObsClient(
    access_key_id='UJDPK31ANIBV0XTEUN5N',
    secret_access_key='NhQExxv9PUYsvmvGnVReizRksaiHcJdQ6vMMw19d',
    server='obs.cn-east-3.myhuaweicloud.com'
)
obs_audio_prefix = f"aigc/aigc_{MY_CONFIG['env']}/"

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
    print(f"开始上传音频{audio_path}")
    if not audio_path:
        return ""
    obs_audio_path = await upload_to_obs(audio_path, obs_audio_prefix, project_id)
    obs_audio_path = obs_audio_path.replace("\\", "/")
    return obs_audio_path

async def upload_to_obs(filename: str, obs_prefix: str = "ai_picture/mark/demo/frames_test/", project_id=None) -> str:
    if project_id is not None:
        obs_prefix = obs_prefix + project_id
    fname = os.path.basename(filename)
    obs_key = os.path.join(obs_prefix, fname).replace("\\", "/")

    try:
        resp = await asyncio.to_thread(obs_client.putFile, bucketName=BUCKET_NAME, objectKey=obs_key,
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

    :param path: OBS 对象路径
    :param save_dir: 本地保存目录，默认 ./obs_video
    :return: 本地完整文件路径
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
        # 查询 diskcache 缓存（多进程安全，自带 TTL 续期）
        cached_path = await asyncio.to_thread(get_from_cache, path, True)
        if cached_path:
            return str(cached_path)

        # 缓存未命中，从 OBS 下载
        start = time.time()
        log.info(f"{fn}开始下载")

        resp = await asyncio.to_thread(obs_client.getObject, bucketName=BUCKET_NAME, objectKey=path,
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
            file_name = os.path.basename(path)
            obs_key = f"{obs_key_prefix}/{file_name}"
            url = await upload_to_obs(path, obs_key)
            return url

    tasks = [_upload(p) for p in file_paths]
    obs_keys = await asyncio.gather(*tasks)

    return obs_keys

def obs_key_exists(obs_path: str) -> bool:
    """
    判断 OBS 对象是否存在
    """
    try:
        key = obs_path
        resp = obs_client.headObject(BUCKET_NAME, key)
        return resp.status < 300
    except Exception as e:
        log.exception(f"OBS 路径{obs_path}不存在 异常: {e}")
        return False

if __name__ == "__main__":
    # 手动测试用
    test_paths = [
        "aigc/aigc_local/1998/1998743094727520258/0/video/1765372463420.mp4",
        "aigc/aigc_local/1998/1997943094727520258/0/video/1765372463421.mp4",
    ]
    for path in test_paths:
        exists = obs_key_exists(path)
        print(f"[TEST] obs_path={path}, exists={exists}")