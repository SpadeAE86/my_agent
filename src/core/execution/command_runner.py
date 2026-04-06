# core/execution/command_runner.py — Shell 命令执行器
# 职责:
#   1. 在子进程中执行用户或工具发起的 Shell 命令
#   2. 超时保护: 超过限制自动终止进程
#   3. 输出捕获: stdout + stderr 合并返回
#   4. 工作目录隔离: 每次执行在指定的沙箱目录中
