import asyncio
import os
import uuid
from typing import Dict

class FileManager:
    """
    文件管理器 (单例)。
    负责管理本地文件读写的锁机制。
    写入锁：每个文件只能有一个写入者，记录形式为 {local_path: uuid}。
    """
    _instance = None
    _init_lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileManager, cls).__new__(cls)
            cls._instance._locks: Dict[str, str] = {}  # {local_path: lock_uuid}
        return cls._instance

    def acquire_write_lock(self, local_path: str) -> str:
        """获取写入锁，如果已被占用抛出异常。"""
        # 标准化路径
        path = os.path.abspath(local_path)
        if path in self._locks:
            raise RuntimeError(f"文件 {local_path} 正在被写入(Lock UUID: {self._locks[path]})，请稍后再试。")
        
        lock_id = str(uuid.uuid4())
        self._locks[path] = lock_id
        return lock_id

    def release_write_lock(self, local_path: str, lock_id: str):
        """释放写入锁。"""
        path = os.path.abspath(local_path)
        if path not in self._locks:
            return  # 已经没有锁了
        if self._locks[path] != lock_id:
            raise RuntimeError(f"释放锁失败：提供的 Lock UUID ({lock_id}) 与占用锁 ({self._locks[path]}) 不匹配！")
        del self._locks[path]

    def check_can_read(self, local_path: str):
        """检查是否可以读取。如果有写锁存在，则不可读。"""
        path = os.path.abspath(local_path)
        if path in self._locks:
            raise RuntimeError(f"文件 {local_path} 正在被写入(Lock UUID: {self._locks[path]})，目前禁止读取，请稍后重试。")
    
    # ─── 文件操作方法 (同步/异步) ──────────────────────────────────
    async def read_lines(self, local_path: str, start_line: int = 1, lines_amount: int = -1) -> str:
        """
        读取文件指定行数。start_line 从 1 开始。lines_amount = -1 读到底。
        """
        self.check_can_read(local_path)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"文件不存在: {local_path}")
            
        with open(local_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        total_lines = len(lines)
        start_idx = max(start_line - 1, 0)
        
        if lines_amount == -1:
            end_idx = total_lines
        else:
            end_idx = start_idx + lines_amount
            
        selected_lines = lines[start_idx:end_idx]
        return "".join(selected_lines)

    async def write_to_file(self, local_path: str, content: str, mode: str, offset_line: int = -1):
        """
        执行文件写入操作 (内部负责取锁与放锁)。
        mode: 
            - 'overwrite': 覆盖写入
            - 'append': 追加到末尾
            - 'insert': 插入到特定行数之后（offset_line 控制）
        """
        lock_id = self.acquire_write_lock(local_path)
        try:
            # 确保目录存在
            dirname = os.path.dirname(local_path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
                
            if mode == 'overwrite':
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(content)
            elif mode == 'append':
                with open(local_path, "a", encoding="utf-8") as f:
                    # 总是换行追加
                    if not content.startswith("\n"):
                        f.write("\n")
                    f.write(content)
            elif mode == 'insert':
                if offset_line < 1:
                    raise ValueError("insert 模式下，offset_line 必须 >= 1")
                
                # 如果文件不存在，就当 overwrite 写
                if not os.path.exists(local_path):
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write(content)
                else:
                    with open(local_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        
                    insert_idx = offset_line - 1
                    # 处理前置换行逻辑，按行插入
                    if not content.endswith('\n'):
                        content += '\n'
                        
                    lines.insert(insert_idx, content)
                    
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
            else:
                raise ValueError(f"不支持的写入模式: {mode}")

        finally:
            # 万一发生异常，也一定要释放锁
            self.release_write_lock(local_path, lock_id)

# 暴露单例实力
file_manager = FileManager()
