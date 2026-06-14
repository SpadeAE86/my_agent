from typing import Any

def get_section_using_tools(tool_manager: Any) -> str | None:
    """工具使用指引 — 根据当前可用工具动态生成，并整合专属工具优先的规范。"""
    if tool_manager is None:
        return None

    tool_names = tool_manager.list_names()
    if not tool_names:
        return None

    lines = ["# Using Your Tools\n"]

    # 专属工具替代通用 Bash 的明确指引
    lines.append(
        "Do NOT use the generic bash/command execution tool to run commands when a relevant dedicated tool is provided. "
        "Using dedicated tools allows the user to better understand, review, and approve your work. This is CRITICAL:\n"
        "- **To read files**: Use `read_file` instead of `cat`, `head`, `tail`, or `sed`.\n"
        "- **To edit files**: Use `replace_file_content` or `multi_replace_file_content` instead of `sed`, `awk`, or writing custom python edit scripts.\n"
        "- **To create files**: Use `write_to_file` instead of `cat <<EOF` redirection or `echo` shell redirects.\n"
        "- **To search for files**: Use dedicated search/list tools instead of running `find` or `ls` in terminal shells.\n"
        "- **To manage/edit force-directed graphs**: Use `list_graphs` to list saved graphs, `read_graph` to read their relations, and `update_graph` (or `make_graph`) to write changes. Do NOT manually edit graph JSON files using file writing tools unless dedicated tools are insufficient.\n"
    )

    # 列出所有可用工具
    lines.append("## Available Tool Schemas")
    for name in tool_names:
        tool = tool_manager.get(name)
        if tool:
            lines.append(f"- **{tool.name}**: {tool.description}")

    # 调用优化指引
    lines.append("")
    lines.append(
        "You can call multiple tools in a single response. "
        "If you intend to call multiple tools and there are no dependencies between them, make all independent tool calls in parallel to increase efficiency. "
        "However, if some tool calls depend on previous calls to inform dependent values, do NOT call these tools in parallel and instead call them sequentially.\n\n"
        "**CRITICAL FRAMEWORK CONSTRAINT / 框架核心约束**:\n"
        "- If you decide to perform any action (such as reading the canvas, updating nodes, or generating images), you MUST output the corresponding tool calls *immediately* in the same response. Do NOT output a conversational response saying you 'will do it' or 'plan to do it' without attaching the tool call, because the framework will automatically terminate the loop and close the session if no tool calls are emitted.\n"
        "- 如果你决定执行任何操作（例如读取画布、更新节点、读取/分析图片、生成图像等），你必须在当前回复中**立即**输出对应的工具调用（tool call）。绝对不能只在回复中说“我将去读取图片”、“我准备对比一下”等陈述性计划文本而不附带任何工具调用！因为如果没有输出工具调用，框架会自动认为任务已结束并终止运行会话。"
    )

    # 子 Agent 说明 (如存在 spawn_agent)
    if "spawn_agent" in tool_names:
        lines.append("")
        lines.append(
            "## Spawning Sub Agents\n\n"
            "Use the spawn_agent tool for complex sub-tasks that require multiple steps. "
            "The sub-agent starts with zero context — brief it like a colleague who just walked in. "
            "Explain what you're trying to accomplish, what you've already learned, "
            "and give enough context for it to make judgment calls.\n\n"
            "Do NOT delegate understanding. Don't write 'based on your findings, fix the bug'. "
            "Write prompts that prove you understood: include file paths, line numbers, what to change."
        )

    return "\n".join(lines)
