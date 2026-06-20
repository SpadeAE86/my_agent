# core/tools/builtin/prompt_templates.py — 提示词模板管理工具
from __future__ import annotations
from typing import Optional, Any
from pydantic import Field

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput
from services.media_generate_services.prompt_template_db_service import prompt_template_db_service


class PromptTemplatesInput(ToolInput):
    """manage_prompt_templates 工具的入参。"""
    action: str = Field(
        ...,
        description="操作类型，可选值: 'list' (列出所有模板名称), 'get' (获取特定模板内容), 'create' (保存或更新模板)"
    )
    name: Optional[str] = Field(
        default=None,
        description="模板名称。当 action 为 'get' 或 'create' 时必填"
    )
    content: Optional[str] = Field(
        default=None,
        description="模板的具体提示词内容。当 action 为 'create' 时必填"
    )


class PromptTemplatesOutput(ToolOutput):
    """manage_prompt_templates 工具的出参。"""
    result: Any = Field(None, description="操作执行的结果")


async def handle_prompt_templates(
    params: PromptTemplatesInput,
    *,
    agent: Any = None,
    tool_manager: Any = None,
    **_kwargs: Any,
) -> ToolOutput:
    action = params.action.strip().lower()
    
    if action == "list":
        try:
            names = await prompt_template_db_service.list_names()
            return ToolOutput(
                success=True,
                message=f"已找到的模板列表: {names}",
                data={"names": names}
            )
        except Exception as e:
            return ToolOutput(
                success=False,
                message=f"列出模板失败: {e}"
            )
            
    elif action == "get":
        if not params.name:
            return ToolOutput(
                success=False,
                message="错误：当 action 为 'get' 时，必须提供 name 参数"
            )
        try:
            tmpl = await prompt_template_db_service.get_by_name(params.name)
            if tmpl:
                return ToolOutput(
                    success=True,
                    message=f"成功获取模板 '{params.name}': {tmpl.content}",
                    data={"name": tmpl.name, "content": tmpl.content}
                )
            else:
                return ToolOutput(
                    success=False,
                    message=f"未找到名为 '{params.name}' 的模板"
                )
        except Exception as e:
            return ToolOutput(
                success=False,
                message=f"获取模板失败: {e}"
            )
            
    elif action == "create":
        if not params.name or not params.content:
            return ToolOutput(
                success=False,
                message="错误：当 action 为 'create' 时，必须提供 name 和 content 参数"
            )
        try:
            tmpl = await prompt_template_db_service.upsert(params.name, params.content)
            return ToolOutput(
                success=True,
                message=f"成功创建/更新模板 '{params.name}'",
                data={"name": tmpl.name, "content": tmpl.content}
            )
        except Exception as e:
            return ToolOutput(
                success=False,
                message=f"保存模板失败: {e}"
            )
            
    else:
        return ToolOutput(
            success=False,
            message=f"未知操作类型 '{params.action}'，可选操作为: 'list', 'get', 'create'"
        )


# ─── 工具定义 ─────────────────────────────────────────────────────
tool_def = ToolDef(
    name="manage_prompt_templates",
    description=(
        "用于管理与调用数据库中的提示词模板。支持：\n"
        "1. list: 列出所有可用的模板名称，供用户挑选；\n"
        "2. get: 获取特定模板的提示词内容进行生图；\n"
        "3. create: 新建或修改指定模板，便于复用沉淀知识。"
    ),
    input_schema=PromptTemplatesInput,
    output_schema=PromptTemplatesOutput,
    handler=handle_prompt_templates,
    tags=["prompt", "template", "database"],
    timeout=10.0,
    is_concurrency_safe=True,
)
