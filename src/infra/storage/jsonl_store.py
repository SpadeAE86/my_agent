# infra/storage/jsonl_store.py — JSONL 追加式存储
# 职责:
#   1. 以追加模式写入 JSON Lines (每行一个 JSON 对象)
#   2. 读取时逐行解析, 支持尾部 N 条读取
#   3. 配合 diskcache 管理写入锁
#   4. 用于短期记忆的实时持久化
