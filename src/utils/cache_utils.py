"""
文件缓存工具
使用 diskcache，进程安全，线程安全
由它自动处理过期，简化 set 和 get，处理竞态
由于主要为了缓存 video 和 audio 等数据，使用全文件保存，优化缓存文件后缀名
"""

import io
import os
import codecs
import os.path as op
from pathlib import Path
from diskcache import Cache, Disk, UNKNOWN
from typing import Union

from config.config import MY_CONFIG, ENV
from infra.logging.logger import logger, time_it


# --- 自定义带文件格式后缀的 Disk 类 ---
class SuffixDisk(Disk):
    def __init__(self, directory, **kwargs):
        super().__init__(directory, **kwargs)

    def filename(self, key=UNKNOWN, value=UNKNOWN):
        """
        重写 suffix 逻辑
        """
        # 提取文件后缀
        suffix = '.val'
        if isinstance(key, str):
            _, suffix = os.path.splitext(key)

        hex_name = codecs.encode(os.urandom(16), 'hex').decode('utf-8')
        sub_dir = op.join(hex_name[:2], hex_name[2:4])
        name = hex_name[4:] + suffix
        filename = op.join(sub_dir, name)
        full_path = op.join(self._directory, filename)
        return filename, full_path


@time_it
def check_in_cache(key: str) -> bool:
    return _CACHE.touch(key, expire=_CACHE_TTL)


@time_it
def set_to_cache(key: str, data: Union[bytes, io.BufferedIOBase]):
    assert data
    if check_in_cache(key):
        return
    # 使用 add 防止高并发下 set 覆写同一 key 导致正在被前一个进程读取的旧物理文件被提前删除报错
    _CACHE.add(key, data, expire=_CACHE_TTL, read=True)


@time_it
def get_from_cache(key: str, as_path: bool = False) -> Union[bytes, Path, None]:
    if not check_in_cache(key):
        return None

    if as_path:
        with _CACHE.get(key, read=True, default=None) as handle:
            if handle is None:
                logger.warning(f'⚠️ 索引存在但句柄获取失败: {key}')
                _CACHE.delete(key)
                return None
            file = Path(handle.name)
            if not file.exists():
                logger.warning(f'⚠️ 物理文件丢失: {file}')
                _CACHE.delete(key)
                return None
            return file
    else:
        data = _CACHE.get(key, default=None)
        if data is None:
            logger.warning(f'⚠️ 续期后触发自动删除: {key}')
            _CACHE.delete(key)
        return data


@time_it
def reconcile_cache_integrity():
    logger.info("🚀 开始执行缓存一致性深度自检...")
    expired_count = _CACHE.expire()
    broken_count = 0
    total_count = 0

    for key in list(_CACHE.iterkeys()):
        total_count += 1
        try:
            with _CACHE.get(key, read=True, default=None) as handle:
                if handle is None:
                    _CACHE.delete(key)
                    broken_count += 1
                    continue
                file_path = Path(handle.name)
                if not file_path.exists():
                    _CACHE.delete(key)
                    broken_count += 1
        except Exception as e:
            _CACHE.delete(key)
            broken_count += 1

    current_size = _CACHE.volume()
    logger.success(
        f'✅ 自检完成! 总计: {total_count}, 清理过期: {expired_count}, '
        f'清理损坏: {broken_count}, 当前大小: {current_size / 1024 ** 3:.2f}GB'
    )

# --- 配置 ---
if 'cache_config' in MY_CONFIG:
    # _CACHE_DIR = MY_CONFIG['cache_config'][ENV]['cache_dir']

    # 暂时统一使用该路径，不使用内存盘路径，避免找不到
    from config.config import PROJECT_ROOT

    _CACHE_DIR = str(PROJECT_ROOT / 'cache')

    _CACHE_MAX_SIZE = MY_CONFIG['cache_config'][ENV]['cache_max_size']
    _CACHE_TTL = MY_CONFIG['cache_config'][ENV]['cache_ttl']
else:
    from config.config import PROJECT_ROOT
    _CACHE_DIR = str(PROJECT_ROOT / 'cache')
    _CACHE_MAX_SIZE = 10737418240
    _CACHE_TTL = 86400

os.makedirs(_CACHE_DIR, exist_ok=True)

_CACHE: Cache = Cache(
    _CACHE_DIR,
    disk=SuffixDisk,
    size_limit=_CACHE_MAX_SIZE,
    disk_min_file_size=0,
    eviction_policy='least-recently-used',
)
