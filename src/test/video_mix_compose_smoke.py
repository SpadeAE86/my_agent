# -*- coding: utf-8 -*-
"""
混剪合成链路烟测：单镜 ``MixedVideoRequest`` 序列化 + 写入 ``mix_video_overall_time``（mock）+ 轮询 ``output_url``。

需可连 MySQL（与 ``mix_compose.overall_time_mysql_env`` 指向的库一致），且存在表 ``mix_video_overall_time``（至少含 ``biz_id``、``output_url`` 列）。

Run::

  cd my_agent/src
  python -m test.video_mix_compose_smoke
"""

from __future__ import annotations

import asyncio
import os
import random
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.pydantic.mix import (  # noqa: E402
    AudioConfig,
    Cap,
    CapConfig,
    CropConfig,
    MixedVideoRequest,
)
from services.video_compose_services import (  # noqa: E402
    fetch_mix_result_obs_url,
    mock_write_mix_overall_time,
)
from infra.storage.mysql_connector import mysql_connector  # noqa: E402


async def _amain() -> None:
    await mysql_connector.init()
    try:
        biz_id = random.randint(100, 999)
        video_url = "https://demo-bucket.obs.cn-east-3.myhuaweicloud.com/clips/scene1.mp4"
        audio_url = "https://demo-bucket.obs.cn-east-3.myhuaweicloud.com/tts/line1.wav"
        audio_dur = 2.5
        tail = 0.4

        req = MixedVideoRequest(
            biz_id=biz_id,
            obs_video_path_list=[video_url],
            crop_config=[CropConfig(start=0.0, end=audio_dur + tail)],
            obs_audio_path_list=[audio_url],
            audio_config=[AudioConfig(start=0.0, end=audio_dur, offset=0.0, volume=1.0)],
            cap_config=CapConfig(
                caption_list=[
                    Cap(start=0.0, end=audio_dur, cap="烟测字幕一行"),
                ]
            ),
        )
        payload = req.model_dump(exclude_none=True, mode="json")
        print("MixedVideoRequest JSON keys:", sorted(payload.keys()))
        print("biz_id:", biz_id)

        mock_out = os.getenv("MOCK_MIX_RESULT_URL", "https://example.com/smoke-mix-output.mp4")
        await mock_write_mix_overall_time(str(biz_id), mock_out)
        print("mock row written")

        for i in range(20):
            got = await fetch_mix_result_obs_url(str(biz_id))
            print(f"poll {i}: output_url={got!r}")
            if got:
                break
            await asyncio.sleep(0.3)
    finally:
        await mysql_connector.close()


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
