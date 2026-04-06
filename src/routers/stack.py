import os.path
import sys
import threading
import traceback

from fastapi import APIRouter, Header
from tabulate import tabulate

from infra.logging.logger import logger as log

stack_router = APIRouter()

@stack_router.get("/stack")
async def debug_stack():
    """调试接口：打印所有线程调用栈"""
    stacks = []
    for thread_id, frame in sys._current_frames().items():
        stacks.append(f"Thread {thread_id}:\n{''.join(traceback.format_stack(frame))}")
    headers = ["ThreadID", "Name", "State", "Key Stack Point"]
    rows = []
    for tid, frame in sys._current_frames().items():
        stack = traceback.extract_stack(frame)
        last_call = stack[-1] if stack else None
        row = [
            tid,
            next((t.name for t in threading.enumerate() if t.ident == tid), "Unknown"),
            "阻塞" if last_call and "ssl.py" in last_call.filename else "运行",
            f"{os.path.basename(last_call.filename)}:{last_call.lineno}" if last_call else ""
        ]
        rows.append(row)

    log.info(tabulate(rows, headers=headers))
    return {"stacks": stacks}