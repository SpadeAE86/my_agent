# core/skills/selector.py — Skill 场景化选择器
from __future__ import annotations
from typing import List, Dict, Any
from core.skills.loader import skill_loader


class SkillSelector:
    def select_skills(self, query: str | list) -> List[Dict[str, Any]]:
        """
        根据用户查询匹配相关的技能。
        目前采用简单的关键词匹配策略。
        """
        if isinstance(query, list):
            text_parts = []
            for part in query:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text") or "")
                elif isinstance(part, str):
                    text_parts.append(part)
            query = " ".join(text_parts)
        elif not isinstance(query, str):
            query = str(query)

        query_lower = query.lower()
        matched = []
        for skill in skill_loader.list_skills():
            # 匹配名字、描述或内容里的关键词
            if (skill["name"].lower() in query_lower or
                skill["description"].lower() in query_lower or
                ("image" in query_lower and "生图" in skill["description"]) or
                ("海报" in query_lower and "banner" in skill["description"].lower()) or
                ("咖啡" in query_lower and "banner" in skill["description"].lower()) or
                ("图" in query_lower and "生图" in skill["description"]) or
                ("画" in query_lower and "生图" in skill["description"]) or
                ("提示词" in query_lower and "提示词" in skill["description"]) or
                ("模板" in query_lower and "模板" in skill["description"])):
                matched.append(skill)
        return matched


skill_selector = SkillSelector()
