# core/tools/__init__.py — 工具子系统公共接口
# 子模块:
#   schema       — Pydantic 基类: ToolInput / ToolOutput / ToolDef
#   tool_manager — 统一的注册表 + 执行器 (ToolManager)
#   builtin/     — 内置工具 (spawn_agent, file_read, shell_exec 等)

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput
from core.tools.tool_manager import ToolManager

__all__ = ["ToolDef", "ToolInput", "ToolOutput", "ToolManager"]
