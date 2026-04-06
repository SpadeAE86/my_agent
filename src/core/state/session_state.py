# core/state/session_state.py — 会话级状态
# 字段:
#   session_id      — 会话唯一标识
#   user_id         — 用户标识
#   main_agent      — 主 Agent 实例引用
#   sub_agents      — 活跃子 Agent 列表
#   created_at      — 会话创建时间
#   last_active_at  — 最后活跃时间
#   is_planning     — 是否处于规划模式 (等待用户确认 plan)
