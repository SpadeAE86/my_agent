# infra/storage/ — 文件与本地持久化
# 子模块:
#   file_store      — 通用文件读写 (替代旧 file_utils)
#   jsonl_store     — JSONL 格式的追加式存储 (短期记忆)
#   diskcache_store — Diskcache (SQLite backend) 封装, 并发安全的 KV 存储
