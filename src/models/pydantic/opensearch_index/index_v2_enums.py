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
    "航拍",
    "屏幕录制",
    "展台转盘",
]

SHOT_TYPE_CHOICES: Final[List[str]] = [
    UNKNOWN,
    "大远景",
    "远景",
    "中景",
    "特写",
    "大特写",
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
    "使用场景展示",
    "功能讲解",
    "对比评测",
    "建立空间感",
    "展示细节",
    "烘托氛围",
    "品牌/形象传达",
    "权益/价格说明"
]

FOOTAGE_TYPE_CHOICES: Final[List[str]] = [
    UNKNOWN,
    "CG",
    "原创实拍",
    "KOL拍摄",
    "TVC切片",
    "直播切片",
    "海报/静帧",
]


# --- Weather / time ---
WEATHER_CHOICES: Final[List[str]] = [
    UNKNOWN,
    "晴天",
    "酷暑",
    "阴天",
    "极寒",
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

PRODUCT_STATUS_SCENE_CHOICES: Final[List[str]] = [
    UNKNOWN,
    "静态内饰",
    "静态外观",
    "静态空间",
    "路跑内饰",
    "路跑外观",
    "发布会现场",
]

PERSON_DETAIL_CHOICES: Final[List[str]] = [
    "无人物",
    "老人",
    "小孩",
    "男性",
    "女性",
    "多人",
]


# --- Movement (keep small & stable; allow a static fallback) ---
MOVEMENT_CHOICES: Final[List[str]] = [
    UNKNOWN,
    "静态展示",
    "行驶",
    "刹车",
    "加速",
    "起步",
    "变道",
    "转弯",
    "掉头",
    "泊车",
    "倒车",
    "充电",
    "爆胎",
]

KEY_TRAITS_CHOICES: Final[List[str]] = [
    # --- 能源/补能/续航 ---
    "续航",
    "低油耗",
    "大电池",
    "充电快",
    "路跑",
    "充电",
    # --- 操控/动作/动态事件 ---
    "加速",
    "起步",
    "转弯",
    "掉头",
    "刹车",
    "爆胎",
    "轮胎转弯",
    # --- 座舱/人物/乘坐 ---
    "后排",
    # --- 天气/季节 ---
    "夏天",
    "高温",
    "冬天",
    "雨天",
    # --- 场景/路况 ---
    "山路",
    "街区",
    "狭窄街道",
    "露营",
    "自驾",
    # --- NVH/音频/空调 ---
    "安静",
    "降噪",
    "声道",
    "音箱",
    "空调",
]

