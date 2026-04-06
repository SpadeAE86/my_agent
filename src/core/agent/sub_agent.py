# core/agent/sub_agent.py — 子任务执行 Agent (Worker)
# 职责:
#   1. 从 task.json 加载被分配的目标、约束条件与上下文
#   2. 使用主 Agent 授予的工具子集执行任务
#   3. 维护自己的短期记忆 (独立 JSONL)
#   4. 任务完成后返回结构化结果给主 Agent
# 限制:
#   - 无法创建子 Agent (防止递归失控)
#   - 无法修改 plan.md / task.json
#   - 工具集由主 Agent 在创建时指定
