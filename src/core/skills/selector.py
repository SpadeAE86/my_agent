# core/skills/selector.py — Skill 场景化选择器
# 职责:
#   1. 根据当前任务描述, 从注册表中匹配最相关的 Skills
#   2. 匹配策略: 关键词匹配 + 标签过滤 + 可选的语义相似度
#   3. 返回需要注入到 prompt 中的 Skill 列表
#   4. 标记为"需要 readSkill"的 Skill, 通知 Agent 先读取再使用
