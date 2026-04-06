# core/llm/prompt_builder.py — 动态 Prompt 组装器
# 职责:
#   1. 按优先级拼接 prompt 各部分:
#      [系统提示] + [长期记忆] + [技能说明] + [工具列表] + [短期对话历史] + [用户消息]
#   2. Token 预算管理: 在上下文窗口内智能裁剪各部分
#   3. 支持不同角色的 prompt 模板 (主 Agent vs 子 Agent)
#   4. 工具列表 → JSON Schema 的格式化输出
