# core/skills/loader.py — Skill 文件加载器
# 职责:
#   1. 扫描 skills/ 目录, 发现所有 SKILL.md 文件
#   2. 解析 YAML frontmatter (name, description, tags)
#   3. 加载同目录下的 .py 文件作为可选的 Pydantic Schema 约束
#   4. 构建 Skill 注册表供 selector 使用
