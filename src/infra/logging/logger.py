# infra/logging/logger.py — 统一日志封装
# 职责:
#   1. 封装 loguru 或 stdlib logging, 提供统一 API
#   2. 支持结构化日志输出 (JSON 格式)
#   3. 按模块/Agent ID 分级别记录
#   4. 日志文件按天轮转, 自动清理过期日志
from functools import wraps
from config.config import ENV, MY_CONFIG
from loguru import logger
import sys, threading, time
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
if not ENV:
    ENV = MY_CONFIG['env']
    logger.warning(f"未找到 .env 文件，环境变量加载失败, 回退到config.yml环境{ENV}")

    # --- 日志 配置 ---
    LOG_LEVEL = MY_CONFIG['log'][ENV]['level'].upper()

    # --- 日志 初始化 ---
    logger.remove()  # 移除默认
    logger.add(
        sys.stderr,
        format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:1.1}</level> | <yellow>{process}</yellow>:<yellow>{thread}</yellow> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}',
        level=LOG_LEVEL,
        enqueue=True,  # 🌟 重要：开启异步写入，防止日志 IO 阻塞你的主逻辑（尤其是音视频处理）
    )

def time_it(func):
    """
    记录时间装饰器
    :param func:
    :return:
    """
    max_time: float = 180

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 当日志等级大于 TRACE 时，直接运行，不计时
        if logger.level(LOG_LEVEL).no > logger.level('TRACE').no:
            return func(*args, **kwargs)

        func_name = func.__name__
        stop_event = threading.Event()

        # 监控函数
        def monitor_task():
            count = 0
            while not stop_event.wait(1):  # 每秒检查一次 stop_event
                count += 1
                if count % 10 == 0:
                    logger.trace(f'⏳ [监控] 函数 [{func_name}] 已运行 {count}s...')

                if count >= max_time:
                    # 注意：子线程抛异常不会杀掉主线程
                    # 我们在这里记录并让主线程在结束后感知
                    logger.warning(f'🚨 [监控] 函数 [{func_name}] 运行已达 {max_time}s 限制，已运行 {count}s！')
                    # break

        # 开启守护线程
        t = threading.Thread(target=monitor_task, daemon=True)
        t.start()

        try:
            start_time = time.perf_counter()
            logger.trace(f'🚀 函数 [{func_name}] 开始执行')
            _result = func(*args, **kwargs)
            actual_duration = time.perf_counter() - start_time
            logger.trace(f'✅ 函数 [{func_name}] 执行完毕，总耗时: {actual_duration:.4f}s')

            # # 检查是否真的跑太久了 # todo
            # if actual_duration >= max_time:
            #     from exceptions.FuncException import ExecutionTimeoutError
            #     raise ExecutionTimeoutError(
            #         f'函数 [{func_name}] 实际耗时 {actual_duration:.2f}s，超过 {max_time}s 限制')
            return _result

        finally:
            # 无论成功还是报错，通知监控线程退出
            stop_event.set()
            # 显式等待监控线程结束（可选，防止日志交织）
            t.join(0.1)
            # logger.trace(f'✅ 函数 [{func_name}] 运行结束，监控线程已释放')

    return wrapper

# --- 日志 配置 ---
LOG_LEVEL = MY_CONFIG['log'][ENV]['level'].upper()

# --- 日志 初始化 ---
logger.remove()  # 移除默认
logger.add(
    sys.stderr,
    format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:1.1}</level> | <yellow>{process}</yellow>:<yellow>{thread}</yellow> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}',
    level=LOG_LEVEL,
    enqueue=True,  # 🌟 重要：开启异步写入，防止日志 IO 阻塞你的主逻辑（尤其是音视频处理）
)

setup_logger()