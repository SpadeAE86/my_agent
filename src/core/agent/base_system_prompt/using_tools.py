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
        "However, if some tool calls depend on previous calls to inform dependent values, do NOT call these tools in parallel and instead call them sequentially."
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
