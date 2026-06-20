# -*- coding: utf-8 -*-
"""
批量本地 MP4 入库（走与 HTTP 相同的后台分析链路），绕开浏览器轮询。

在 ``my_agent/src`` 下执行::

  python -m services.batch_local_video_import

默认扫描目录（可用 ``--dir`` 覆盖）::

  C:\\Users\\25065\\Downloads\\汽车\\ls6_video\\LS6视频

依赖与 Web 服务相同：``connector_loader``、MySQL、OpenSearch、OBS、豆包等。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# 保证以 ``python -m services.batch_local_video_import`` 从 ``my_agent/src`` 运行时能 import
_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from infra.connector_loader import connector_loader  # noqa: E402
from infra.logging.logger import logger as log  # noqa: E402
from infra.storage.sqlmodel_init import create_tables_if_not_exists  # noqa: E402
from routers.video_analysis.analysis import _bg_analyze_video  # noqa: E402
from services.video_match_services.analysis_video import _get_or_upload_source_video  # noqa: E402
from services.taskboard_services.http_request_trace_service import http_request_trace_service  # noqa: E402


DEFAULT_SCAN_DIR = r"C:\Users\25065\Downloads\汽车\ls6_video\LS6视频"


def _collect_mp4(root: Path) -> List[Path]:
    out: List[Path] = []
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".mp4":
            out.append(p)
    return sorted(out)


def _file_project_id(path: Path) -> str:
    md5_hash = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()[:16]


async def _import_one(
    path: Path,
    *,
    frame_interval: float,
    threshold: float,
    split_scenes: bool,
    workspace: str,
    car_model: Optional[str],
    sem: asyncio.Semaphore,
) -> None:
    async with sem:
        file_name = path.name
        local_path = str(path.resolve())
        if not os.path.isfile(local_path):
            log.warning("skip missing file: {}", local_path)
            return
        project_id = _file_project_id(path)
        log.info("[batch] start project_id={} file={}", project_id, file_name)
        t0 = time.monotonic()
        try:
            obs_video_url = await _get_or_upload_source_video(local_path, project_id, car_model)
            if not obs_video_url:
                raise RuntimeError("源视频上传 OBS 失败或未返回 URL")
            trace_id = await http_request_trace_service.create_initial(
                request_url="batch_local_video_import",
                http_method="CLI",
                request_body={
                    "project_id": project_id,
                    "path": local_path,
                    "workspace": workspace,
                    "car_model": car_model,
                },
                business_type="VIDEO_ANALYSIS",
                method_name="batch_local_video_import",
                upstream_task_id=project_id,
            )
            await _bg_analyze_video(
                project_id,
                local_path,
                file_name,
                frame_interval,
                threshold,
                None,
                split_scenes,
                workspace,
                car_model,
                obs_video_url,
                trace_id,
                remove_local_after=False,
            )
            log.info(
                "[batch] done project_id={} file={} in {:.1f}s",
                project_id,
                file_name,
                time.monotonic() - t0,
            )
        except Exception as e:
            log.exception("[batch] FAILED project_id={} file={}: {}", project_id, file_name, e)


async def _amain() -> int:
    ap = argparse.ArgumentParser(description="批量扫描本地 MP4 并执行视频分析入库")
    ap.add_argument("--dir", default=DEFAULT_SCAN_DIR, help="要扫描的根目录（递归 *.mp4）")
    ap.add_argument("--workspace", default="v2", help="与前端 workspace 一致，如 v1 / v2")
    ap.add_argument("--car-model", default="LS6", dest="car_model", help="写入 OBS 子目录与 history.car_model")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并行分析路数 1–8（大则吃满 CPU/GPU）",
    )
    ap.add_argument("--max", type=int, default=0, help="最多处理多少个文件，0 表示不限制")
    ap.add_argument("--frame-interval", type=float, default=2.0, dest="frame_interval")
    ap.add_argument("--threshold", type=float, default=30.0)
    ap.add_argument("--no-split-scenes", action="store_true", dest="no_split_scenes")
    args = ap.parse_args()
    if args.concurrency < 1 or args.concurrency > 8:
        ap.error("--concurrency must be between 1 and 8")
    root = Path(args.dir)
    paths = _collect_mp4(root)
    if args.max and args.max > 0:
        paths = paths[: args.max]
    if not paths:
        log.error("目录下未找到 mp4: {}", root)
        return 1
    log.info(
        "batch import: dir={} files={} concurrency={} workspace={} car_model={}",
        root,
        len(paths),
        args.concurrency,
        args.workspace,
        args.car_model,
    )
    await connector_loader.startup()
    try:
        await create_tables_if_not_exists()
    except Exception as e:
        log.warning("create_tables_if_not_exists: {}", e)
    sem = asyncio.Semaphore(args.concurrency)
    split_scenes = not args.no_split_scenes
    await asyncio.gather(
        *[
            _import_one(
                p,
                frame_interval=args.frame_interval,
                threshold=args.threshold,
                split_scenes=split_scenes,
                workspace=args.workspace,
                car_model=(args.car_model or None),
                sem=sem,
            )
            for p in paths[:50]
        ]
    )
    await connector_loader.shutdown()
    log.info("batch import finished.")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
