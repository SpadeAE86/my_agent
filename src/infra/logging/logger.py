# infra/logging/logger.py — 统一日志封装
# 职责:
#   1. 封装 loguru 或 stdlib logging, 提供统一 API
#   2. 支持结构化日志输出 (JSON 格式)
#   3. 按模块/Agent ID 分级别记录
#   4. 日志文件按天轮转, 自动清理过期日志

from loguru import logger
import sys
import os
from datetime import datetime

def setup_logger():
    logger.remove()  # 移除默认 handler

    logger.add(
        sys.stdout,
        level="INFO",
        colorize=True,
        format=(
            "<green>[{time:YYYY-MM-DD HH:mm:ss}]</green>"
            "<cyan>[{thread.name}]</cyan>"
            "<blue>[{function}]</blue> "
            "<level>{level}:\t{message}</level>"
        ),
    )

setup_logger()