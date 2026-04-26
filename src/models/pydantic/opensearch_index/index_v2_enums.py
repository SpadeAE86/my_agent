"""
Centralized enum-like constants for IndexV2.

Purpose:
- Keep "fixed choices" in one place for:
  - OpenSearch keyword fields (filterable)
  - Doubao prompt injection / schema descriptions
  - Server-side validation & normalization

This file intentionally uses plain lists (instead of Enum) so it can be
embedded into prompts easily and serialized without extra work.
"""

from __future__ import annotations

from typing import Final, List


# --- Generic / shared ---
UNKNOWN: Final[str] = "未知"


# --- Media / presentation ---
SHOT_STYLE_CHOICES: Final[List[str]] = [
    UNKNOWN,
    "车内POV",
    "车外跟拍",
    "固定机位",
    "手持",
    "航拍",
    "屏幕录制",
    "展台转盘",
]

FRAME_SIZE_CHOICES: Final[List[str]] = [
    UNKNOWN,
    "横版16:9",
    "竖版9:16",
    "其他比例",
]

VIDEO_USAGE_CHOICES: Final[List[str]] = [
    UNKNOWN,
    "产品展示",
    "使用场景",
    "功能讲解",
    "对比评测",
    "直播切片",
    "海报/静帧",
]


# --- Weather / time ---
WEATHER_CHOICES: Final[List[str]] = [
    UNKNOWN,
    "晴天",
    "阴天",
    "雨天",
    "雪天",
    "雾天",
    "夜雨",
]

TIME_CHOICES: Final[List[str]] = [
    UNKNOWN,
    "白天",
    "夜晚",
    "黄昏",
    "清晨",
    "室内",
]


# --- Color ---
CAR_COLOR_CHOICES: Final[List[str]] = [
    UNKNOWN,
    "黑",
    "白",
    "灰",
    "红",
    "蓝",
    "黄",
    "紫",
    "粉",
    "茶",
    "多种",
    "其他涂装",
]


# --- Movement (keep small & stable; allow a static fallback) ---
MOVEMENT_CHOICES: Final[List[str]] = [
    UNKNOWN,
    "静态展示",
    "行驶",
    "刹车",
    "变道",
    "转弯",
    "掉头",
    "泊车",
    "倒车",
    "充电",
]

