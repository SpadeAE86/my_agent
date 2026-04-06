# infra/storage/file_store.py — 通用文件读写
# 职责:
#   1. 封装 pathlib 的常用操作: read, write, append, exists, mkdir
#   2. 自动创建中间目录
#   3. UTF-8 编码统一处理
#   4. 文件锁 (防止多进程写入冲突)
import yaml


def read_yaml(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def save_yaml(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)