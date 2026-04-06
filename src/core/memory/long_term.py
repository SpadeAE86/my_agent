# core/memory/long_term.py — 长期记忆 (MEMORY.md)
# 存储路径: data/memory/{user_id}/MEMORY.md
# 职责:
#   1. 由后台 Cron 任务定期触发, 对中期日志进行深度压缩
#   2. 提取: 用户画像、行为偏好、长期目标、重要知识点
#   3. 增量更新 MEMORY.md (不覆盖, 仅合并新信息)
#   4. 仅主 Agent 拥有长期记忆的读取权限
