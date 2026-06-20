"""
智己 LS6 / LS9 产品方卖点词表：供视频分析 v2 视觉提示与口播转写 Stage1 系统提示注入。
数据文件：src/data/zhiji_selling_points.json
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "zhiji_selling_points.json"


@lru_cache
def _catalog() -> Dict[str, Any]:
    with open(_DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def normalize_zhiji_car_key(car_model: Optional[str]) -> Optional[str]:
    """从表单/口播里的车型文案归一到 LS6 或 LS9。"""
    raw = (car_model or "").strip().upper()
    if not raw:
        return None
    # 先判 LS9，避免「LS6」误匹配含 6 的串
    if "LS9" in raw:
        return "LS9"
    if "LS6" in raw:
        return "LS6"
    return None


def _format_term_block(name: str, payload: Dict[str, Any]) -> str:
    defin = str(payload.get("definition") or "").strip()
    benefits = payload.get("core_benefits") or []
    scenes = payload.get("related_scenes") or []
    bstr = "、".join(str(x) for x in benefits if str(x).strip())
    sstr = "、".join(str(x) for x in scenes if str(x).strip())
    parts = [f"「{name}」"]
    if defin:
        parts.append(defin)
    if bstr:
        parts.append(f"核心价值：{bstr}")
    if sstr:
        parts.append(f"典型使用场景：{sstr}")
    return "；".join(parts)


def build_v2_vision_selling_appendix(car_model: Optional[str]) -> str:
    """拼接进豆包视觉分析 v2 的用户提示前导说明（短、可检索）。"""
    key = normalize_zhiji_car_key(car_model)
    if not key:
        return ""
    terms = _catalog().get(key)
    if not isinstance(terms, dict) or not terms:
        return ""
    lines = [
        f"【{key} 官方功能/卖点参考（辅助识别画面中的技术与营销表述；一切须以画面客观事实为准，禁止臆造）】",
        "- core_benefits 类信息可映射到 function_selling_points / design_selling_points / function_adjectives / marketing_phrases；",
        "- related_scenes 类信息可映射到 scenario_a / scenario_b，并与 scene_location、product_status_scene 一致；",
        "- 词条的完整中文名称可出现在 text（若画面有字样/UI）或 description 的客观表述中。",
        "",
    ]
    for name in sorted(terms.keys(), key=len, reverse=True):
        payload = terms.get(name)
        if not isinstance(payload, dict):
            continue
        lines.append("- " + _format_term_block(name, payload))
    return "\n".join(lines).strip()


def build_script_stage1_selling_appendix(car_model: Optional[str]) -> str:
    """口播转写 Stage1 系统提示附录：帮助分镜规划对齐官方叫法与典型场景。"""
    key = normalize_zhiji_car_key(car_model)
    if not key:
        return ""
    terms = _catalog().get(key)
    if not isinstance(terms, dict) or not terms:
        return ""
    lines = [
        f"【{key} 官方卖点词表（分镜规划时：术语尽量与下列「标准名称」一致；场景取向与 typical 场景相容）】",
    ]
    for name in sorted(terms.keys(), key=len, reverse=True):
        payload = terms.get(name)
        if not isinstance(payload, dict):
            continue
        lines.append("- " + _format_term_block(name, payload))
    return "\n".join(lines).strip()
