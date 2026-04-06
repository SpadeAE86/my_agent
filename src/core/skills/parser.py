# core/skills/parser.py — Skill 文档解析器
# 职责:
#   1. 解析 SKILL.md 的 Markdown 内容
#   2. 提取: 使用场景、参数说明、使用示例、注意事项
#   3. 将结构化信息格式化为 prompt 片段供 LLM 理解
#   4. 合并同目录下 Pydantic .py 的 Schema 信息
