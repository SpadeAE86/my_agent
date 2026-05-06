"""
core/workspace.py — 多 workspace 配置注册表

每个 workspace 对应独立的：
  - OpenSearch index (通过 schema 类的 settings 区分)
  - MySQL 分镜卡片表 (video_analysis_shot_cards / video_analysis_shot_cards_v2)

共用的资源 (workspace-agnostic)：
  - video_analysis_video_v2       (素材行，外键源)
  - video_analysis_scene_frames   (抽帧缓存，避免重复上传)
  - video_analysis_history        (分析历史记录)

使用方式
--------
    from core.workspace import get_workspace, WORKSPACE_REGISTRY

    ws = get_workspace("v2")
    table = ws.shot_cards_table          # "video_analysis_shot_cards_v2"
    ver   = ws.shot_cards_version        # "v2"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class WorkspaceConfig:
    """Immutable per-workspace configuration bundle."""

    key: str
    label: str
    description: str

    # MySQL 分镜卡片表名
    shot_cards_table: str
    # 用于 API query param shot_cards_version=
    shot_cards_version: str

    # OpenSearch index 别名/名称
    opensearch_index: str

    # 是否为默认 workspace
    is_default: bool = False


# ─── 注册表 ────────────────────────────────────────────────────────────────
_WORKSPACES: List[WorkspaceConfig] = [
    WorkspaceConfig(
        key="v1",
        label="经典分析 v1",
        description="基础理解 schema，字段包含 search_tags / adjective / visual_quality / marketing_tags",
        shot_cards_table="video_analysis_shot_cards",
        shot_cards_version="v1",
        opensearch_index="car_interior_analysis",
        is_default=True,
    ),
    WorkspaceConfig(
        key="v2",
        label="深度分析 v2",
        description=(
            "对齐 CarInteriorAnalysisV2 schema，字段包含 key_words / footage_type / "
            "shot_style / shot_type / camera_movement / scene_location / "
            "design_adjectives / function_selling_points 等"
        ),
        shot_cards_table="video_analysis_shot_cards_v2",
        shot_cards_version="v2",
        opensearch_index="car_interior_analysis_v2",
        is_default=False,
    ),
]

WORKSPACE_REGISTRY: Dict[str, WorkspaceConfig] = {ws.key: ws for ws in _WORKSPACES}
DEFAULT_WORKSPACE_KEY: str = next(ws.key for ws in _WORKSPACES if ws.is_default)


def get_workspace(key: str | None) -> WorkspaceConfig:
    """Return workspace config for *key*, falling back to default."""
    if key and key in WORKSPACE_REGISTRY:
        return WORKSPACE_REGISTRY[key]
    return WORKSPACE_REGISTRY[DEFAULT_WORKSPACE_KEY]


def list_workspaces() -> List[dict]:
    """Serializable list for API response."""
    return [
        {
            "key": ws.key,
            "label": ws.label,
            "description": ws.description,
            "is_default": ws.is_default,
        }
        for ws in _WORKSPACES
    ]
