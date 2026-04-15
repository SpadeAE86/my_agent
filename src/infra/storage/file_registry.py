import asyncio
import os
import uuid
import hashlib
import time
from typing import Dict

from infra.cache.redis_connector import redis_manager

class FileRegistry:
    """
    虚拟文件注册中心与分布式锁管理器 (单例)。
    职责:
    1. 生成短 file_id (例如 file-1a2b3c) 映射到极其混乱的绝对路径。
    2. 基于 Redis 提供毫秒级自旋调度的鲁棒性分布式文件锁。
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileRegistry, cls).__new__(cls)
            # 退化为仅映射短指针用。如需多机器共享该映射，未来可上卷至 DB/Redis，由于短指针目前本局会话用完即废，内存足矣。
            cls._instance._id_to_path: Dict[str, str] = {}
            cls._instance._path_to_id: Dict[str, str] = {}
        return cls._instance

    def _generate_short_id(self, path: str) -> str:
        """根据路径 hash 生成 8 位的短 ID"""
        hash_obj = hashlib.md5(path.encode('utf-8'))
        return f"file-{hash_obj.hexdigest()[:8]}"

    def register_file(self, local_path: str) -> str:
        """注册一个物理文件并返回极短的 file_id 提供给 Agent"""
        path = os.path.abspath(local_path)
        if path in self._path_to_id:
            return self._path_to_id[path]
        
        base_file_id = self._generate_short_id(path)
        file_id = base_file_id
        counter = 1
        while file_id in self._id_to_path:
            file_id = f"{base_file_id}-{counter}"
            counter += 1
            
        self._id_to_path[file_id] = path
        self._path_to_id[path] = file_id
        return file_id

    def resolve_file(self, file_id: str) -> str:
        """将短 file_id 翻译回物理路径"""
        if file_id not in self._id_to_path:
            # 容错：万一大模型传进来的本身就是绝对物理路径，直接放行
            if os.path.exists(file_id) or os.path.isabs(file_id):
                return file_id
            raise RuntimeError(f"文件找不到: 无效的短连接 ID ({file_id})，注册表中木有！")
        return self._id_to_path[file_id]

    # ─── Redis 分布式机制同步底层 ─────────────────────────────────
    def _get_lock_key(self, path: str) -> str:
        return f"agent:file_lock:{path}"

    async def acquire_write_lock(self, local_path: str, timeout: float = 5.0) -> str:
        """获取写入锁，支持大模型在后台悄然等待 timeout 期间抢锁"""
        path = self.resolve_file(local_path)
        lock_key = self._get_lock_key(path)
        lock_uuid = str(uuid.uuid4())
        
        client = redis_manager.async_client
        start_time = time.time()
        
        while True:
            # 尝试拿锁，原子化 nx=True。防死锁过期时间设置为 60s
            acquired = await client.set(lock_key, lock_uuid, nx=True, ex=60)
            if acquired:
                return lock_uuid
            
            if time.time() - start_time >= timeout:
                raise TimeoutError(f"获取文件 {path} 写入锁因为等待超过 {timeout} 秒报错超时，可能有严重卡死。")
            
            await asyncio.sleep(0.1)

    async def release_write_lock(self, local_path: str, lock_uuid: str):
        """用 uuid 释放掉写锁，基于 Lua 脚本保障纯净的原子判断。"""
        path = self.resolve_file(local_path)
        lock_key = self._get_lock_key(path)
        client = redis_manager.async_client
        
        # 严格执行自己的解锁权判定
        lua_script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        else
            return 0
        end
        """
        await client.eval(lua_script, 1, lock_key, lock_uuid)

    async def check_can_read(self, local_path: str, timeout: float = 5.0) -> str:
        """自旋直到没人在写它，给 Agent 清爽的读环境"""
        path = self.resolve_file(local_path)
        lock_key = self._get_lock_key(path)
        client = redis_manager.async_client
        
        start_time = time.time()
        while True:
            is_locked = await client.exists(lock_key)
            if not is_locked:
                return path
            
            if time.time() - start_time >= timeout:
                raise TimeoutError(f"读取由于文件长期处于被独占写入状态，阻塞自旋 {timeout} 秒后超时保护！")
                
            await asyncio.sleep(0.1)

    # ─── 文件操作方法 (异步IO层) ──────────────────────────────────
    async def read_lines(self, local_path: str, start_line: int = 1, lines_amount: int = -1) -> str:
        # 首先用阻塞的方式确认绿灯
        real_path = await self.check_can_read(local_path)
        
        if not os.path.exists(real_path):
            raise FileNotFoundError(f"物理文件不存在: {real_path}")
            
        with open(real_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        total_lines = len(lines)
        start_idx = max(start_line - 1, 0)
        end_idx = total_lines if lines_amount == -1 else start_idx + lines_amount
            
        selected_lines = lines[start_idx:end_idx]
        return "".join(selected_lines)

    async def write_to_file(self, local_path: str, content: str, mode: str, offset_line: int = -1):
        real_path = self.resolve_file(local_path)
        lock_id = await self.acquire_write_lock(real_path)
        try:
            dirname = os.path.dirname(real_path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
                
            if mode == 'overwrite':
                with open(real_path, "w", encoding="utf-8") as f:
                    f.write(content)
            elif mode == 'append':
                with open(real_path, "a", encoding="utf-8") as f:
                    if not content.startswith("\n"):
                        f.write("\n")
                    f.write(content)
            elif mode == 'insert':
                if offset_line < 1:
                    raise ValueError("insert 模式下，offset_line 必须 >= 1")
                if not os.path.exists(real_path):
                    with open(real_path, "w", encoding="utf-8") as f:
                        f.write(content)
                else:
                    with open(real_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    insert_idx = offset_line - 1
                    if not content.endswith('\n'):
                        content += '\n'
                    lines.insert(insert_idx, content)
                    with open(real_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
            else:
                raise ValueError(f"不支持的写入模式: {mode}")

        finally:
            # 万一发生异常也一定要强制进入 Redis 解开全局锁
            await self.release_write_lock(real_path, lock_id)

file_registry = FileRegistry()
