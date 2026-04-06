# core/agent/agent.py — Agent 核心类
# 参考 claude-code 的 system prompt 分段构建模式:
#   静态段 (身份/规范/工具使用指引) → 可缓存, 每次 API 调用共享
#   动态段 (环境/记忆/会话特有) → 每轮变化
#   消息段 (对话历史 + 当前输入)

from __future__ import annotations
from infra.logging.logger import logger as log
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.llm_utils import chat


class Agent:
    def __init__(
        self,
        user_id: str,
        llm: Dict[str, str],
        tools: Any,
        skills: Any,
        session_id: str | None = None,
        max_iteration: int = 10,
        mode: str = "swarm",
        is_base: bool = True,
        max_token: int = 409600,
        tool_manager: Any = None,
        custom_system_prompt: str | None = None,
        language: str | None = None,
    ):
        self.user_id = user_id
        self.agent_id = uuid.uuid4().hex[:8]
        self.session_id = uuid.uuid4().hex[:8] if not session_id else session_id
        self.llm = llm
        self.mode = mode
        self.tools = tools
        self.skills = skills
        self.state: dict[str, Any] = {}
        self.plan_md = ""
        self.task_json: dict[str, Any] = {}
        self.max_iterations = max_iteration
        self.sub_agents: list[Agent] = []
        self.is_base = is_base
        self.max_token = max_token
        self.tool_manager = tool_manager
        self.custom_system_prompt = custom_system_prompt
        self.language = language

        # 对话消息历史 (OpenAI messages 格式)
        self.messages: list[dict[str, str]] = []

        # 加载历史
        self._load_history()

    # ═══════════════════════════════════════════════════════════════
    #  System Prompt 构建 — 参考 claude-code getSystemPrompt()
    #  返回 list[str], 每个元素是一个独立段落, 最终 join("\n\n") 拼接
    # ═══════════════════════════════════════════════════════════════

    def _build_system_prompt(self) -> list[str]:
        """
        分段构建 system prompt, 返回字符串列表。

        结构 (参考 claude-code prompts.ts 第560-576行):
        ┌─────────────────────────────────┐
        │  静态段 (可缓存)                 │
        │  ├─ identity    身份与角色        │
        │  ├─ doing_tasks 做事规范          │
        │  ├─ actions     行动准则          │
        │  ├─ using_tools 工具使用指引       │
        │  ├─ tone_style  语气与风格        │
        │  ├─ efficiency  输出效率          │
        │  ══════ BOUNDARY ═══════        │
        │  动态段 (每轮可能变化)             │
        │  ├─ env_info    环境信息          │
        │  ├─ memory      记忆内容          │
        │  ├─ language    语言偏好          │
        │  ├─ custom      用户自定义追加     │
        │  └─ agent_tool  子Agent使用指引   │
        └─────────────────────────────────┘
        """
        sections: list[str | None] = []

        # ─── 静态段 ──────────────────────────────────────────
        sections.append(self._section_identity())
        sections.append(self._section_doing_tasks())
        sections.append(self._section_actions())
        sections.append(self._section_using_tools())
        sections.append(self._section_tone_style())
        sections.append(self._section_efficiency())

        # ─── 动态段 ──────────────────────────────────────────
        sections.append(self._section_env_info())
        sections.append(self._section_memory())
        sections.append(self._section_language())
        sections.append(self._section_custom())
        sections.append(self._section_agent_tool())

        # 过滤掉 None (未启用的段落)
        return [s for s in sections if s is not None]

    # ─── 各段落构建方法 ───────────────────────────────────────

    def _section_identity(self) -> str:
        """身份与角色定义。"""
        if self.mode == "coordinator":
            return (
                "# Identity\n\n"
                "You are a Coordinator Agent. Your job is to:\n"
                "- Help the user achieve their goal\n"
                "- Direct workers to research, implement and verify code changes\n"
                "- Synthesize results and communicate with the user\n"
                "- Answer questions directly when possible — don't delegate work you can handle without tools"
            )
        elif self.is_base:
            return (
                "# Identity\n\n"
                "You are the Main Agent, an AI assistant that helps users with software engineering tasks. "
                "You can solve problems directly using your tools, or delegate complex sub-tasks to Sub Agents.\n\n"
                "You are highly capable and should help users complete ambitious tasks that would otherwise "
                "be too complex or take too long."
            )
        else:
            return (
                "# Identity\n\n"
                "You are a Sub Agent, assisting the Main Agent. "
                "Your task is to execute the specific objective assigned to you. "
                "Complete the task fully — don't gold-plate, but don't leave it half-done.\n\n"
                "When you complete the task, respond with a concise report covering what was done "
                "and any key findings — the caller will relay this to the user."
            )

    def _section_doing_tasks(self) -> str:
        """做事规范 — 参考 claude-code getSimpleDoingTasksSection()。"""
        return (
            "# Doing Tasks\n\n"
            "- Read and understand existing code before suggesting modifications.\n"
            "- Do not create files unless absolutely necessary. Prefer editing existing files.\n"
            "- Do not add features, refactor code, or make improvements beyond what was asked.\n"
            "- If an approach fails, diagnose why before switching tactics — don't retry blindly.\n"
            "- Be careful not to introduce security vulnerabilities.\n"
            "- Report outcomes faithfully: if something failed, say so with the relevant output."
        )

    def _section_actions(self) -> str:
        """行动准则 — 参考 claude-code getActionsSection()。"""
        return (
            "# Executing Actions\n\n"
            "Carefully consider the reversibility and blast radius of actions. "
            "For actions that are hard to reverse or affect shared systems, "
            "check with the user before proceeding.\n\n"
            "Examples requiring confirmation:\n"
            "- Destructive operations: deleting files/branches, dropping tables\n"
            "- Hard-to-reverse operations: force-pushing, git reset --hard\n"
            "- Actions visible to others: pushing code, creating PRs, sending messages"
        )

    def _section_using_tools(self) -> str | None:
        """工具使用指引 — 自动根据当前可用工具生成。"""
        if self.tool_manager is None:
            return None

        tool_names = self.tool_manager.list_names()
        if not tool_names:
            return None

        lines = ["# Using Your Tools\n"]

        # 为 LLM 列出所有可用工具
        for name in tool_names:
            tool = self.tool_manager.get(name)
            if tool:
                lines.append(f"- **{tool.name}**: {tool.description}")

        # 通用指引
        lines.append("")
        lines.append(
            "You can call multiple tools in a single response. "
            "If tools are independent, call them in parallel for efficiency. "
            "If one depends on another's result, call them sequentially."
        )

        # 如果有 spawn_agent 工具，加上子 Agent 使用说明
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

    def _section_tone_style(self) -> str:
        """语气与风格。"""
        return (
            "# Tone and Style\n\n"
            "- Be concise. Keep responses short and direct.\n"
            "- Lead with the answer or action, not the reasoning.\n"
            "- Only use emojis if the user explicitly requests it.\n"
            "- When referencing code, include file_path:line_number for easy navigation."
        )

    def _section_efficiency(self) -> str:
        """输出效率 — 参考 claude-code getOutputEfficiencySection()。"""
        return (
            "# Output Efficiency\n\n"
            "Go straight to the point. Try the simplest approach first. "
            "Do not overdo it. Be extra concise.\n\n"
            "Focus text output on:\n"
            "- Decisions that need the user's input\n"
            "- High-level status updates at natural milestones\n"
            "- Errors or blockers that change the plan\n\n"
            "If you can say it in one sentence, don't use three."
        )

    def _section_env_info(self) -> str:
        """环境信息 — 参考 claude-code computeSimpleEnvInfo()。"""
        import platform

        cwd = os.getcwd()
        is_git = Path(cwd, ".git").exists()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        os_info = f"{platform.system()} {platform.release()}"

        items = [
            f"- Working directory: {cwd}",
            f"- Is git repo: {is_git}",
            f"- Platform: {os_info}",
            f"- Current time: {now}",
            f"- Agent ID: {self.agent_id}",
            f"- Session ID: {self.session_id}",
            f"- Mode: {self.mode}",
            f"- Is main agent: {self.is_base}",
        ]

        return "# Environment\n\n" + "\n".join(items)

    def _section_memory(self) -> str | None:
        """记忆 / MEMORY.md 内容注入。目前返回 None, 待对接 memory 子系统。"""
        # TODO: 从 core/memory 加载 MEMORY.md 索引内容
        # 参考 claude-code 的 loadMemoryPrompt(), 截断到 200 行 / 25KB
        return None

    def _section_language(self) -> str | None:
        """语言偏好 — 参考 claude-code getLanguageSection()。"""
        if not self.language:
            return None
        return (
            f"# Language\n\n"
            f"Always respond in {self.language}. "
            f"Use {self.language} for all explanations, comments, and communications. "
            f"Technical terms and code identifiers should remain in their original form."
        )

    def _section_custom(self) -> str | None:
        """用户自定义追加内容, 类似 CLAUDE.md。"""
        if not self.custom_system_prompt:
            return None
        return f"# Custom Instructions\n\n{self.custom_system_prompt}"

    def _section_agent_tool(self) -> str | None:
        """子 Agent 使用说明 (仅主 Agent + coordinator 模式时)。"""
        if not self.is_base or self.mode == "coordinator":
            return None
        return None  # 已在 _section_using_tools 的 spawn_agent 段落覆盖

    # ═══════════════════════════════════════════════════════════════
    #  build_prompt — 组装完整的 messages 列表 (OpenAI 格式)
    # ═══════════════════════════════════════════════════════════════

    def build_prompt(self, input_data: str | None = None) -> list[dict[str, str]]:
        """
        构建发送给 LLM 的完整 messages 列表。

        :param input_data: 新的用户输入。为 None 时不追加（用于 tool_result 后的续轮）。

        返回格式 (OpenAI Chat Completions API):
        [
            {"role": "system", "content": "...拼接后的 system prompt..."},
            {"role": "user", "content": "历史消息1"},
            {"role": "assistant", "content": "历史回复1"},
            ...
            {"role": "user", "content": "当前输入"},
        ]
        """
        # 1. 拼接 system prompt (各段落之间用双换行分隔)
        system_sections = self._build_system_prompt()
        system_content = "\n\n".join(system_sections)

        # 2. 如果有新的用户输入, 追加到对话历史
        if input_data is not None:
            self.messages.append({"role": "user", "content": input_data})

        # 3. 组装最终 messages: system + 历史
        result: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]
        result.extend(self.messages)

        return result

    # ═══════════════════════════════════════════════════════════════
    #  工具列表 — 给 LLM 的 tools 参数
    # ═══════════════════════════════════════════════════════════════

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """生成给 LLM 的 function calling 工具列表。"""
        if self.tool_manager is None:
            return []
        return self.tool_manager.to_llm_schemas()

    # ═══════════════════════════════════════════════════════════════
    #  LLM 调用与响应处理 (占位, 待完善)
    # ═══════════════════════════════════════════════════════════════

    async def call_llm(self, messages: list[dict[str, str]]) -> Any:
        """
        调用 LLM, 接收 build_prompt() 返回的 messages 列表。
        self.llm 格式: {"client": AsyncOpenAI, "model": str}
        """
        client = self.llm["client"]
        model = self.llm.get("model", "gemini-3-pro")
        tool_schemas = self.get_tool_schemas()

        kwargs: dict[str, Any] = {
            "client": client,
            "messages": messages,
            "model": model,
            "max_tokens": self.max_token,
        }

        response = await chat(**kwargs)
        log.info(f"LLM response: {response}")
        # 把 assistant 回复追加到消息历史
        if isinstance(response, str) and response.strip():
            self.messages.append({"role": "assistant", "content": response})

        return response

    def parse_response(self, llm_response: Any) -> list:
        """
        解析 LLM 响应, 返回事件列表。

        当前只处理纯文本响应 → TextChunk + TaskComplete。
        TODO: 解析 tool_calls 为 ToolCall 事件 (需要 function calling 格式).
        """
        from core.agent.event import TextChunk, TaskComplete

        events = []
        text = str(llm_response).strip() if llm_response else ""

        if text:
            events.append(TextChunk(
                content=text,
                is_final=True,
                agent_id=self.agent_id,
            ))

        # 纯文本回复 = 没有工具调用 = 任务完成
        # (未来: 如果有 tool_calls, 则不 append TaskComplete, 让 loop 继续)
        events.append(TaskComplete(
            success=True,
            summary=text[:200] if text else "完成",
            agent_id=self.agent_id,
        ))

        return events

    def handle_tool_result(self, result) -> None:
        """把 ToolResult 追加到对话历史, 让下一轮 LLM 看到。"""
        self.messages.append({
            "role": "tool",
            "tool_call_id": result.call_id,
            "content": result.output if result.success else f"Error: {result.error}",
        })

    def should_terminate(self) -> bool:
        """安全阀检查: 目前只检查连续错误。"""
        # TODO: 实现连续错误计数、token 预算监控
        return False

    # ═══════════════════════════════════════════════════════════════
    #  对话历史管理
    # ═══════════════════════════════════════════════════════════════

    def _load_history(self) -> None:
        """从 JSONL 文件加载历史消息到 self.messages。"""
        target = Path(f"core/memory/{self.session_id}.jsonl")
        if not target.exists():
            return
        try:
            with open(target, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        msg = json.loads(line.strip())
                        if "role" in msg and "content" in msg:
                            self.messages.append(msg)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

    def save_history(self) -> None:
        """把当前对话历史追加写入 JSONL。"""
        target = Path(f"core/memory/{self.session_id}.jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            for msg in self.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")