# core/skills/ — 进阶能力定义 (Skills)
# 与 Tools 的区别:
#   Tools = 严格 Pydantic Schema, 高频稳定, 框架直接调用
#   Skills = Markdown 文档, 描述复杂使用场景, Agent 调用前需先 readSkill 理解
# 子模块:
#   loader    — 加载 Skills.md 和同目录下的 Pydantic schema (.py)
#   selector  — 根据任务关键词匹配最相关的 Skill
#   parser    — 解析 Skill 文档, 提取参数说明和使用示例
