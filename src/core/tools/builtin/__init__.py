# core/tools/builtin/__init__.py — 内置工具包
# 存放框架自带的工具实现。
# 每个 .py 文件暴露一个 `tool_def: ToolDef` 变量,
# ToolManager.auto_discover() 会自动扫描并注册。
#
# 已实现:
#   spawn_agent   — 派生子 Agent 执行子任务
#
# 待实现:
#   file_read     — 读取文件内容
#   file_write    — 写入/创建文件
#   file_search   — 搜索文件 (grep/glob)
#   shell_exec    — 执行 Shell 命令
#   web_search    — 网络搜索
#   memory_search — 搜索 Agent 记忆
