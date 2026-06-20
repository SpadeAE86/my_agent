"""
Mock 自测：视频匹配 GET job 前的分镜 URL hydrate（纯数字 history_id / 空 video_path）。

运行（在 my_agent 目录下；按本机解释器替换路径）::

  set PYTHONPATH=src
  python src/test/test_video_match_url_hydrate_mock.py

  示例（Conda py312）::

  set PYTHONPATH=src
  C:\\Users\\25065\\.conda\\envs\\py312\\python.exe src/test/test_video_match_url_hydrate_mock.py

不连 MySQL / OpenSearch；仅验证「命中里有 history_id、top1 为空 → 通过 resolve 补全」的流程。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

# 允许从仓库任意 cwd 执行：把 src 加进 path
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


async def _run_hydrate_case() -> None:
    from services.video_match_services.video_match_service import _hydrate_shot_match_urls_for_response

    shot: dict = {
        "shot_order": 0,
        "top1_obs_url": "",
        "search_status": "done",
        "match_top_hits_json": [
            {"_id": "55_1", "_score": 1.2, "history_id": "55"},
            {"_id": "56_1", "_score": 1.0, "history_id": "56"},
        ],
        "match_hit_count": 2,
    }

    async def fake_resolve(key: str, *, shot_cards_version: str = "v1") -> str:
        mapping = {
            "55": "https://mock.obs.example/bucket/a.mp4",
            "56": "https://mock.obs.example/bucket/b.mp4",
        }
        return mapping.get((key or "").strip(), "")

    with patch(
        "services.video_match_service.video_analysis_db_service.resolve_source_video_url_for_index_key",
        new_callable=AsyncMock,
        side_effect=fake_resolve,
    ):
        await _hydrate_shot_match_urls_for_response(shot, shot_cards_version="v2")

    assert shot["top1_obs_url"] == "https://mock.obs.example/bucket/a.mp4", shot
    assert shot["match_top_hits_json"][0]["video_path"] == "https://mock.obs.example/bucket/a.mp4"
    assert shot["match_top_hits_json"][1]["video_path"] == "https://mock.obs.example/bucket/b.mp4"
    assert len(shot.get("top5_video_urls") or []) >= 2


async def _run_skip_when_top1_exists() -> None:
    from services.video_match_services.video_match_service import _hydrate_shot_match_urls_for_response

    shot = {
        "shot_order": 1,
        "top1_obs_url": "https://existing/already.mp4",
        "search_status": "done",
        "match_top_hits_json": [{"history_id": "55"}],
    }
    mock_resolve = AsyncMock(return_value="https://should-not-call")

    with patch(
        "services.video_match_service.video_analysis_db_service.resolve_source_video_url_for_index_key",
        mock_resolve,
    ):
        await _hydrate_shot_match_urls_for_response(shot, shot_cards_version="v2")

    mock_resolve.assert_not_awaited()
    assert shot["top1_obs_url"] == "https://existing/already.mp4"


async def main() -> None:
    await _run_hydrate_case()
    await _run_skip_when_top1_exists()
    print("video_match_url_hydrate_mock: all assertions passed")


if __name__ == "__main__":
    asyncio.run(main())
