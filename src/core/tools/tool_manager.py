# core/tools/tool_manager.py — 工具管理器 (注册 + 执行 合一)
# 职责:
#   1. 注册/发现: 启动时扫描 builtin/ 目录, 支持运行时动态注册
#   2. 查询: 按名称/标签查找工具, 生成 LLM 工具列表
#   3. 执行: 接收 ToolCall 事件, 查找对应工具, 安全执行并返回 ToolResult
#   4. 安全: 超时保护, 异常捕获, 日志记录

from __future__ import annotations

import asyncio
import importlib
import pkgutil
from time import time
from typing import Any

from core.agent.event import ToolCall, ToolResult
import core
from models.pydantic.tool_schema import ToolDef, ToolOutput
from infra.logging.logger import logger


class ToolManager:
    """
    统一的工具注册表 + 执行器。

    用法:
        manager = ToolManager()
        manager.auto_discover()                # 自动扫描 builtin/ 下的工具
        manager.register(my_tool_def)          # 手动注册自定义工具
        schemas = manager.to_llm_schemas()     # 生成给 LLM 的 function calling schemas
        result = await manager.execute(event)  # 执行 ToolCall 事件
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    # ─── 注册 ────────────────────────────────────────────────────

    def register(self, tool_def: ToolDef) -> None:
        """注册一个工具定义。重复注册同名工具会覆盖。"""
        if tool_def.name in self._tools:
            logger.warning(f"工具 {tool_def.name} 已存在，将被覆盖")
        self._tools[tool_def.name] = tool_def
        logger.info(f"已注册工具: {tool_def.name}")

    def unregister(self, name: str) -> bool:
        """移除一个已注册的工具，返回是否成功。"""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def auto_discover(self) -> int:
        """
        自动扫描 core/tools/builtin/ 下的所有模块。
        每个模块需要暴露一个 `tool_def: ToolDef` 变量。
        返回成功注册的工具数量。
        """
        import core.tools.builtin as builtin_pkg

        count = 0
        for module_info in pkgutil.iter_modules(builtin_pkg.__path__):
            if module_info.name.startswith("_"):
                continue
            try:
                module = importlib.import_module(
                    f"core.tools.builtin.{module_info.name}"
                )
                if hasattr(module, "tool_def"):
                    self.register(module.tool_def)
                    count += 1
                else:
                    logger.debug(
                        f"模块 {module_info.name} 没有 tool_def 变量, 跳过"
                    )
            except Exception as e:
                logger.error(f"加载内置工具 {module_info.name} 失败: {e}")
        logger.info(f"自动发现完成, 共注册 {count} 个内置工具")
        return count

    # ─── 查询 ────────────────────────────────────────────────────

    def get(self, name: str) -> ToolDef | None:
        """按名称查找工具。"""
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        """返回所有已注册的工具名称。"""
        return list(self._tools.keys())

    def list_by_tags(self, *tags: str) -> list[ToolDef]:
        """返回包含指定标签的所有工具。"""
        tag_set = set(tags)
        return [t for t in self._tools.values() if tag_set & set(t.tags)]

    def to_llm_schemas(self, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        """
        生成给 LLM 的 function calling 工具列表。
        :param allowed: 可选的白名单, 只导出这些工具; None 表示全部导出
        """
        tools = self._tools.values()
        if allowed is not None:
            allowed_set = set(allowed)
            tools = [t for t in tools if t.name in allowed_set]
        return [t.to_llm_schema() for t in tools]

    # ─── 执行 ────────────────────────────────────────────────────

    async def execute(self, event: ToolCall, **context: Any) -> ToolResult:
        """
        安全执行一个 ToolCall 事件。

        流程:
        1. 查找工具定义
        2. 校验入参 (Pydantic)
        3. 带超时执行 handler
        4. 异常捕获 → 结构化 ToolResult
        """
        start = time()
        tool = self.get(event.tool_name)

        # 工具不存在
        if tool is None:
            return ToolResult(
                call_id=event.call_id,
                tool_name=event.tool_name,
                output="",
                error=f"未知工具: {event.tool_name}",
                success=False,
                agent_id=event.agent_id,
            )

        # 校验入参
        try:
            validated_input = tool.input_schema(**event.arguments)
        except Exception as e:
            error_msg = f"参数校验失败: {e}"
            logger.error(
                f"工具 {event.tool_name} 入参校验失败!\n  收到的参数: {event.arguments}\n  错误: {e}"
            )
            return ToolResult(
                call_id=event.call_id,
                tool_name=event.tool_name,
                output="",
                error=error_msg,
                success=False,
                agent_id=event.agent_id,
            )

        # 带超时执行
        try:
            output: ToolOutput = await asyncio.wait_for(
                tool.handler(validated_input, **context),
                timeout=tool.timeout,
            )
            elapsed = time() - start
            logger.debug(
                f"工具 {event.tool_name} 执行完成, 耗时 {elapsed:.2f}"
            )

            return ToolResult(
                call_id=event.call_id,
                tool_name=event.tool_name,
                output=output.message or output.model_dump_json(),
                error=None if output.success else output.message,
                success=output.success,
                agent_id=event.agent_id,
            )

        except asyncio.TimeoutError:
            return ToolResult(
                call_id=event.call_id,
                tool_name=event.tool_name,
                output="",
                error=f"工具 {event.tool_name} 执行超时 (>{tool.timeout}s)",
                success=False,
                agent_id=event.agent_id,
            )
        except Exception as e:
            logger.exception(f"工具 {event.tool_name} 执行异常")
            return ToolResult(
                call_id=event.call_id,
                tool_name=event.tool_name,
                output="",
                error=f"执行异常: {type(e).__name__}: {e}",
                success=False,
                agent_id=event.agent_id,
            )

tool_manager = ToolManager() #ToolManager全局单例