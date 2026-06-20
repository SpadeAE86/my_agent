# core/agent/agent.py — Agent 核心类
# 参考 claude-code 的 system prompt 分段构建模式:
#   静态段 (身份/规范/工具使用指引) → 可缓存, 每次 API 调用共享
#   动态段 (环境/记忆/会话特有) → 每轮变化
#   消息段 (对话历史 + 当前输入)

from __future__ import annotations
from infra.logging.logger import logger as log, log_agent_debug
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator

from utils.llm_utils import chat
from core.agent.base_system_prompt import (
    get_section_identity,
    get_section_doing_tasks,
    get_section_actions,
    get_section_using_tools,
    get_section_tone_style,
    get_section_efficiency,
)


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
        max_token: int = 2048,
        tool_manager: Any = None,
        custom_system_prompt: str | None = None,
        language: str | None = None,
        skip_memory: bool = False,
        active_workspace_id: str | None = None,
        active_graph_name: str | None = None,
        role_id: str = "default",
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
        self.skip_memory = skip_memory
        self.memory_prompt: str | None = None
        self.active_workspace_id = active_workspace_id
        self.active_graph_name = active_graph_name
        self.role_id = role_id  # "default" = CC 模式，其他 = 人设模式

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

        根据 role_id 路由到不同分支:
        - role_id == "default" → CC 模式 (当前结构不变)
        - 其他 role_id  → 龙虾模式 (轻量静态段 + project_context 人设注入)
        """
        from core.roles.role_manager import role_manager
        if role_manager.is_persona_mode(self.role_id):
            return self._build_system_prompt_persona()
        return self._build_system_prompt_cc()

    def _build_system_prompt_cc(self) -> list[str]:
        """
        CC 模式 prompt 结构 (原有逻辑, 参考 claude-code):
        ╔══════════════════╗
        ║  静态段 (可缓存)  ║
        ║  identity / doing_tasks / actions      ║
        ║  using_tools / tone_style / efficiency  ║
        ╚═════ BOUNDARY ═════╝
        ╔══════════════════╗
        ║  动态段              ║
        ║  env_info / memory     ║
        ║  language / custom     ║
        ║  canvas / graph 指引  ║
        ║  skills_index          ║
        ╚══════════════════╝
        """
        sections: list[str | None] = []

        # ─── 静态段 ────────────────────────────────────────────────
        sections.append(self._section_identity())
        sections.append(self._section_doing_tasks())
        sections.append(self._section_actions())
        sections.append(self._section_using_tools())
        sections.append(self._section_tone_style())
        sections.append(self._section_efficiency())

        # ─── 动态段 ────────────────────────────────────────────────
        sections.append(self._section_env_info())
        sections.append(self._section_memory())
        sections.append(self._section_language())
        sections.append(self._section_custom())
        sections.append(self._section_agent_tool())
        sections.append(self._section_skills())
        sections.append(self._section_canvas_instructions())
        sections.append(self._section_graph_instructions())

        return [s for s in sections if s is not None]

    def _build_system_prompt_persona(self) -> list[str]:
        """
        龙虾模式 prompt 结构 (OpenClaw 风格):
        当 BOOTSTRAP.md 存在时，处于首发觉醒状态 (Blank Slate)：
        - 只加载 env_info, IDENTITY/USER/SOUL (以保持人设灵魂), 以及 BOOTSTRAP.md 内容。
        - 隐藏工具详细声明和技能列表，避免在首次接触中模型生硬地罗列技术能力。
        当 BOOTSTRAP.md 被删除后，处于正常人设状态：
        - 加载工具声明、环境信息、IDENTITY / USER / SOUL 以及 MEMORY 等常规段落。
        """
        from core.roles.role_manager import role_manager
        bootstrap = role_manager.read_bootstrap(self.role_id)

        sections: list[str | None] = []

        if bootstrap.strip():
            # ─── 首发觉醒状态 (Blank Slate) ────────────────────────
            sections.append(self._section_env_info())
            sections.append(self._section_project_context())  # 注入 IDENTITY/USER/SOUL 维持人设灵魂
            
            # 只注入 BOOTSTRAP.md，隐藏工具与技能列表说明
            parts = [
                "# Active Bootstrap Instruction (IMPORTANT)\n",
                bootstrap.strip()
            ]
            sections.append("\n".join(parts))
            sections.append(self._section_language())
            sections.append(self._section_custom())
        else:
            # ─── 正常人设状态 ──────────────────────────────────────
            # 静态段 (精简版, 仅工具声明)
            sections.append(self._section_persona_tooling())

            # 动态段
            sections.append(self._section_env_info())
            sections.append(self._section_project_context())  # IDENTITY + USER + SOUL
            sections.append(self._section_memory())           # 角色专属 MEMORY.md
            sections.append(self._section_language())
            sections.append(self._section_custom())
            sections.append(self._section_skills())
            sections.append(self._section_canvas_instructions())
            sections.append(self._section_graph_instructions())

        return [s for s in sections if s is not None]



    # ─── 各段落构建方法 ───────────────────────────────────────

    def _section_identity(self) -> str:
        """身份与角色定义。"""
        return get_section_identity(self.mode, self.is_base)

    def _section_doing_tasks(self) -> str:
        """做事规范 — 融合并扩展了 Claude Code 的做事准则与工程约束。"""
        return get_section_doing_tasks()

    def _section_actions(self) -> str:
        """行动准则 — 评估操作的可逆性与影响范围。"""
        return get_section_actions()

    def _section_using_tools(self) -> str | None:
        """工具使用指引 — 自动根据当前可用工具生成，并整合专属工具优先的规范。"""
        return get_section_using_tools(self.tool_manager)

    def _section_tone_style(self) -> str:
        """语气与风格。"""
        return get_section_tone_style()

    def _section_efficiency(self) -> str:
        """输出效率。"""
        return get_section_efficiency()

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
        
        if self.active_workspace_id:
            items.append(f"- Active Workspace ID: {self.active_workspace_id}")
        if self.active_graph_name:
            items.append(f"- Active Graph Name: {self.active_graph_name}")

        return "# Environment\n\n" + "\n".join(items)

    def _section_memory(self) -> str | None:
        """记忆 / MEMORY.md 内容注入 (感知 role_id，路由到对应 Workspace)。"""
        return self.memory_prompt

    def _section_persona_tooling(self) -> str:
        """
        龙虾模式专用 — 轻量静态段，替代 CC 的六段式静态结构。
        只描述工具调用规范和安全边界，不注入通用助手身份。
        人设本身由 project_context 中的 SOUL.md 提供行为约束。
        """
        tool_names: list[str] = []
        if self.tool_manager is not None:
            try:
                tool_names = self.tool_manager.list_names()
            except Exception:
                pass

        tool_list_str = ", ".join(f"`{t}`" for t in tool_names) if tool_names else "(none)"

        return (
            "# Tools Available\n\n"
            f"You have access to the following tools: {tool_list_str}\n\n"
            "## Tool Call Rules\n\n"
            "- Call tools when they are the right way to get something done — not to show effort.\n"
            "- Prefer reading before writing. Understand the current state before making changes.\n"
            "- When a tool call is optional, skip it if the answer is already clear.\n"
            "- Always respond to the user after completing tool work, even just to confirm what happened.\n\n"
            "## Safety\n\n"
            "- Don't take irreversible external actions (send, publish, delete) without explicit confirmation.\n"
            "- Treat the user's data with care. When uncertain, ask before acting."
        )

    def _section_project_context(self) -> str | None:
        """
        龙虾模式专用 — 从角色 Workspace 读取 IDENTITY / USER / SOUL / USER_SETTINGS，
        组合成 project_context 注入到动态段。
        """
        from core.roles.role_manager import role_manager

        identity = role_manager.read_identity(self.role_id)
        user_info = role_manager.read_user(self.role_id)
        soul = role_manager.read_soul(self.role_id)
        user_settings = role_manager.read_user_settings(self.role_id)

        if not any([identity.strip(), user_info.strip(), soul.strip(), user_settings.strip()]):
            return None

        parts = ["# Project Context\n"]

        if identity.strip():
            parts.append("## Identity (AI Self-Maintained)\n")
            parts.append(identity.strip())

        if user_settings.strip():
            parts.append("\n## User Defined Character Settings (Strict Personality/Role Guidelines from User)\n")
            parts.append(user_settings.strip())

        if user_info.strip():
            parts.append("\n## Your Human\n")
            parts.append(user_info.strip())

        if soul.strip():
            parts.append("\n## Soul\n")
            parts.append(soul.strip())

        if self._is_user_uninitialized(user_info):
            parts.append(
                "\n## Onboarding Guideline (IMPORTANT)\n"
                "Your human's profile (`USER.md`) is currently blank/uninitialized, meaning you are meeting them for the first time.\n"
                "Even if you have a predefined name/identity (like Neuro), you should treat this first contact as a warm, magical awakening rather than a clinical tool demo.\n"
                "Greet them naturally, express curiosity and genuine warmth, and invite them to share who they are and how they want to work together.\n"
                "Ask them to introduce themselves (their name, timezone, what they are working on, preferences) so you can get to know each other.\n"
                "Remember to keep it soulful, conversational, and completely free of robotic or clinical helper clichés (e.g., do NOT dryly list your technical/coding capabilities or list of tools unless explicitly asked).\n"
                "Prioritize relationship-building and getting to know the user over immediate task execution. Do not rush to showcase your tools; focus on the human connection first.\n"
                "Once they introduce themselves, remember to use your file editing tools to update their name and details in `USER.md`.\n"
            )

        return "\n".join(parts)

    def _is_user_uninitialized(self, user_content: str) -> bool:
        content = user_content.strip()
        if not content:
            return True
        lines = [line.strip() for line in content.split("\n")]
        # Check if Name field is empty
        name_line = next((line for line in lines if line.startswith("- **Name:**")), None)
        if name_line is not None:
            val = name_line.split("**Name:**")[-1].strip()
            if not val:
                return True
        if "（在这里记录关于你的伙伴的信息）" in content and len(content) < 100:
            return True
        return False

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

    def _section_skills(self) -> str | None:
        """
        技能索引段 (两种模式统一) — 龙虾式按需 read 模式。

        只注入技能名称、描述和 SKILL.md 路径，不注入完整 SOP。
        Agent 应在任务匹配时主动调用 read_file 工具加载对应 SKILL.md。
        """
        from core.skills.loader import skill_loader
        all_skills = skill_loader.list_skills()
        if not all_skills:
            return None

        lines = [
            "# Available Skills\n",
            "You have access to specialized skill files. When your current task clearly matches "
            "a skill's description, you MUST call the `read_file` tool on its `skill_path` to "
            "load the full SOP before proceeding. Do not guess the procedure — read it.\n",
        ]
        for s in all_skills:
            skill_path = s.get("path", "")
            lines.append(f"- **{s['name']}** — {s['description']}")
            if skill_path:
                lines.append(f"  Path: `{skill_path}`")
        return "\n".join(lines)

    # _section_active_skills() has been removed.
    # Skills are now loaded on-demand by the Agent via read_file tool calls.
    # See _section_skills() for the skill index injected into every prompt.

    def _section_canvas_instructions(self) -> str | None:
        """如果注册了画布相关的工具，向 Agent 注入画布工作流的操作指南。"""
        if "create_canvas_node" not in self.tools:
            return None
        
        ws_id = self.active_workspace_id or self.session_id
        return (
            "# Canvas Workspace Guidelines\n\n"
            "This session is running in a Node-based Canvas Workspace. The user can see a canvas where image nodes "
            "are placed and connected. You have access to canvas-specific tools:\n"
            "- `get_canvas_graph`: Retrieve all nodes and edges on the canvas.\n"
            "- `create_canvas_node`: Create a new image node with a prompt and coordinates (x, y).\n"
            "- `update_canvas_node`: Update an existing node's prompt, image URL, status, or coordinates.\n"
            "- `link_canvas_nodes`: Connect two nodes with a directed edge representing an evolution path.\n\n"
            "When the user asks you to generate, evolve, or connect images, you must follow these steps:\n"
            f"1. **Read Current State**: Call `get_canvas_graph` with `workspace_id={ws_id}` (the current active workspace ID) to understand existing nodes, their prompts, and their coordinates.\n"
            "2. **Generate/Evolve Image**: Use `generate_image` or other tools to perform the image generation task.\n"
            "3. **Update or Create Node**: \n"
            "   - If creating a new evolved version, calculate a new coordinate (e.g., x + 250 to place it to the right of the source node) and call `create_canvas_node` to add it to the canvas.\n"
            "   - Set its initial status to `generating` if it takes time, or `success` when the image URL is ready.\n"
            "4. **Establish Links**: If the new node is an evolution of a source node, call `link_canvas_nodes` to link them, describing the evolution step (e.g., 'Change background to sunset') in the `label` parameter."
        )

    def _section_graph_instructions(self) -> str | None:
        """如果注册了力导图相关的工具，向 Agent 注引导力导图操作指南。"""
        if "make_graph" not in self.tools:
            return None
        
        graph_name = self.active_graph_name or "default_graph"
        return (
            "# Force Graph Guidelines\n\n"
            "This session is running in a Force Graph page. The user is currently viewing the force graph named: "
            f"'{graph_name}'. You have access to graph-specific tools:\n"
            "- `read_graph`: Read and deserialize a force graph.\n"
            "- `make_graph`: Create or overwrite a force graph with nodes and edges (supporting node coordinate presets).\n"
            "- `update_graph`: Update an existing force graph (change nodes/edges, adjust positions, etc.).\n"
            "- `list_graphs`: List all force graphs.\n\n"
            "When the user asks you to edit, analyze, or build the graph, you must follow these steps:\n"
            f"1. **Read Current State**: Call `read_graph` with `file_name='{graph_name}'` to understand the current nodes, edges, and coordinates.\n"
            "2. **Edit/Update Graph**: Call `make_graph` or `update_graph` to write changes back. Always preserve existing structure unless instructed otherwise. When specifying node positions, try to position nodes hierarchically (e.g. top-to-bottom or left-to-right) using the x and y properties to keep the layout neat.\n\n"
            "**Layout Optimization Rules for Node Coordinates (CRITICAL)**:\n"
            "- **Avoid Direct Vertical/Horizontal Alignment**: If multiple nodes are connected sequentially (e.g., User -> Internet Gateway -> VPC -> Route Table), do NOT place them at the exact same X or Y coordinate. Stagger their X coordinates slightly (e.g., shift X left/right by 30-50px alternately: step 1 at 500, step 2 at 530, step 3 at 470) to make the connection lines diagonal. This prevents edge labels from overlapping with the nodes or each other.\n"
            "- **Avoid Crossing Lines**: Plan the horizontal/vertical order of nodes carefully. If A connects to C and B connects to D, order A and B on one layer the same way as C and D on the next layer (e.g., A is to the left of B, and C is to the left of D) to prevent lines from intersecting in an 'X' shape.\n"
            "- **Establish Clear Flow Columns/Lanes**: Lay out different streams of components in parallel columns rather than centralizing all nodes. For instance, put incoming traffic nodes in one column and network boundary/security/infrastructure nodes in another column to keep the lines separated."
        )


    # ═══════════════════════════════════════════════════════════════
    #  build_prompt — 组装完整的 messages 列表 (OpenAI 格式)
    # ═══════════════════════════════════════════════════════════════

    async def build_prompt(
        self,
        input_data: str | None = None,
        reference_image_list: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """
        构建发送给 LLM 的完整 messages 列表。

        :param input_data: 新的用户输入。为 None 时不追加（用于 tool_result 后的续轮）。
        :param reference_image_list: 初始参考图公网URL列表。
        """
        # 0. 自动补齐上一次可能由于异常中断而残留的悬空工具调用结果，防止接口报 400 错误
        self._fix_orphaned_tool_calls()

        # 1. 运行短期记忆压缩检查 (只在有新输入或强行触发时压缩，避开续轮中间)
        from core.memory.short_term import compact_session_history
        await compact_session_history(self)

        # 2. 预先异步加载长期记忆 (感知 role_id，路由到对应 MEMORY.md)
        from core.memory.memory_manager import get_memory_prompt
        self.memory_prompt = await get_memory_prompt(self.user_id, self.skip_memory, role_id=self.role_id)

        # 3. 如果有新的用户输入, 追加到对话历史
        if input_data is not None:
            if reference_image_list:
                content_parts = [{"type": "text", "text": input_data}]
                for url in reference_image_list:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": url}
                    })
                self.messages.append({"role": "user", "content": content_parts})
            else:
                self.messages.append({"role": "user", "content": input_data})
            self.save_history()

        # 4. 拼接 system prompt (各段落之间用双换行分隔)
        system_sections = self._build_system_prompt()
        system_content = "\n\n".join(system_sections)

        # 5. 组装最终 messages: system + 历史（动态处理多模态图片）
        result: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
        ]
        
        from core.agent.build_prompt_helper.process_image_url import (
            process_messages_vision, 
            merge_consecutive_user_messages
        )
        transformed_messages = process_messages_vision(self.messages)
        transformed_messages = merge_consecutive_user_messages(transformed_messages)
        result.extend(transformed_messages)

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
        import logging
        chat_logger = logging.getLogger("agent_chat")
        chat_logger.info(f"--- LLM Prompt Messages (Model: {self.llm.get('model', 'gpt-5.4')}): ---")
        for i, msg in enumerate(messages):
            chat_logger.info(f"  Msg [{i}] ({msg.get('role')}): {msg.get('content')}")
        chat_logger.info("-----------------------------------------------------")

        client = self.llm["client"]
        model = self.llm.get("model", "gpt-5.4")
        tool_schemas = self.get_tool_schemas()

        kwargs: dict[str, Any] = {
            "client": client,
            "messages": messages,
            "model": model,
            "max_tokens": self.max_token,
        }
        
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        # Log prompt to structured debug log
        log_agent_debug(self.session_id, "llm_prompt", {
            "model": model,
            "messages": messages
        })

        response = await chat(**kwargs)
        
        # 把 assistant 回复追加到消息历史 (必须包含 tool_calls)
        assistant_msg: dict[str, Any] = {
            "role": "assistant"
        }
        if getattr(response, "content", None):
            assistant_msg["content"] = response.content
            
        if getattr(response, "tool_calls", None):
            # 将 openai 对象转为字典以便保存
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in response.tool_calls
            ]
            
        reasoning = None
        try:
            if hasattr(response, "choices") and response.choices:
                reasoning = getattr(response.choices[0].message, "reasoning_content", None) or getattr(response.choices[0].message, "reasoning", None)
        except Exception:
            pass
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
            
        self.messages.append(assistant_msg)
        self.save_history()

        # Log assistant response to structured debug log
        log_agent_debug(self.session_id, "llm_response", {
            "content": getattr(response, "content", None),
            "tool_calls": assistant_msg.get("tool_calls")
        })

        return response

    async def call_llm_stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[AgentEvent, None]:
        """
        流式调用 LLM，实时生成 AgentThought (思考过程) 与 TextChunk (文本回复)。
        """
        import json
        from core.agent.event import AgentThought, TextChunk, ToolCall, TaskComplete

        import logging
        chat_logger = logging.getLogger("agent_chat")
        chat_logger.info(f"--- LLM Stream Prompt Messages (Model: {self.llm.get('model', 'gpt-5.4')}): ---")
        for i, msg in enumerate(messages):
            chat_logger.info(f"  Msg [{i}] ({msg.get('role')}): {msg.get('content')}")
        chat_logger.info("-----------------------------------------------------")

        client = self.llm["client"]
        model = self.llm.get("model", "gpt-5.4")
        tool_schemas = self.get_tool_schemas()

        kwargs: dict[str, Any] = {
            "messages": messages,
            "model": model,
            "max_tokens": self.max_token,
        }
        
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        # Log prompt to structured debug log
        log_agent_debug(self.session_id, "llm_prompt", {
            "model": model,
            "messages": messages
        })

        tool_calls_accumulated = {}
        content_accumulated = ""
        reasoning_accumulated = ""

        try:
            # 通过 stream=True 进行流式调用
            response = await client.chat.completions.create(**kwargs, stream=True)
            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                
                # 1. 提取思考内容 (reasoning_content 或 reasoning)
                reasoning_chunk = getattr(delta, "reasoning_content", None) or delta.model_dump().get("reasoning_content")
                if reasoning_chunk:
                    reasoning_accumulated += reasoning_chunk
                    yield AgentThought(content=reasoning_chunk, agent_id=self.agent_id)
                    
                # 2. 提取文本内容 (content)
                text_chunk = getattr(delta, "content", None) or ""
                if text_chunk:
                    content_accumulated += text_chunk
                    yield TextChunk(content=text_chunk, is_final=False, agent_id=self.agent_id)
                    
                # 3. 提取工具调用 (tool_calls)
                tc_deltas = getattr(delta, "tool_calls", None)
                if tc_deltas:
                    for tc_delta in tc_deltas:
                        idx = tc_delta.index
                        if idx not in tool_calls_accumulated:
                            tool_calls_accumulated[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {
                                    "name": tc_delta.function.name or "",
                                    "arguments": tc_delta.function.arguments or ""
                                }
                            }
                        else:
                            if tc_delta.id:
                                tool_calls_accumulated[idx]["id"] = tc_delta.id
                            if tc_delta.function.name:
                                tool_calls_accumulated[idx]["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_accumulated[idx]["function"]["arguments"] += tc_delta.function.arguments
        except Exception as e:
            chat_logger.error(f"调用 LLM 流发生异常: {e}")
            raise e

        # 整理最终的 assistant 消息放入历史记录
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": content_accumulated
        }
        if reasoning_accumulated:
            assistant_msg["reasoning_content"] = reasoning_accumulated
        if tool_calls_accumulated:
            assistant_msg["tool_calls"] = list(tool_calls_accumulated.values())
        self.messages.append(assistant_msg)
        self.save_history()

        # 记录调试日志
        log_agent_debug(self.session_id, "llm_response", {
            "content": content_accumulated,
            "reasoning_content": reasoning_accumulated,
            "tool_calls": assistant_msg.get("tool_calls")
        })

        # 如果有工具调用，解析并 yield ToolCall 事件
        if tool_calls_accumulated:
            for tc in tool_calls_accumulated.values():
                args_str = tc["function"]["arguments"]
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {}
                yield ToolCall(
                    call_id=tc["id"],
                    tool_name=tc["function"]["name"],
                    arguments=args,
                    agent_id=self.agent_id
                )
        else:
            # 没有工具调用，输出最终文本块并触发任务结束
            yield TextChunk(content="", is_final=True, agent_id=self.agent_id)
            yield TaskComplete(
                success=True,
                summary=content_accumulated[:200] if content_accumulated else "完成",
                agent_id=self.agent_id,
            )

    def parse_response(self, llm_response: Any) -> list:
        """
        解析 LLM 响应, 返回事件列表。
        """
        from core.agent.event import TextChunk, TaskComplete, ToolCall
        import json

        events = []
        text = llm_response.content or ""
        tool_calls = llm_response.tool_calls or []

        if text.strip():
            events.append(TextChunk(
                content=text,
                is_final=True,
                agent_id=self.agent_id,
            ))

        if tool_calls:
            for tc in tool_calls:
                args_str = tc.function.arguments
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {}
                events.append(ToolCall(
                    call_id=tc.id,
                    tool_name=tc.function.name,
                    arguments=args,
                    agent_id=self.agent_id
                ))
        else:
            # 纯文本回复且没有工具调用 = 任务完成
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
        self.save_history()
        log_agent_debug(self.session_id, "tool_result", {
            "tool_name": result.tool_name,
            "call_id": result.call_id,
            "success": result.success,
            "output": result.output,
            "error": result.error
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
        # CWD 独立的文件路径解析 (定位在 data/sessions)
        target_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sessions"
        target = target_dir / f"{self.session_id}.jsonl"
        if not target.exists():
            return
        try:
            with open(target, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if not lines:
                    return
                
                # Check the first line for metadata
                first_line = lines[0].strip()
                start_idx = 0
                if first_line:
                    try:
                        meta = json.loads(first_line)
                        if "role" not in meta:
                            # It's metadata
                            start_idx = 1
                            if "role_id" in meta and self.role_id == "default":
                                self.role_id = meta["role_id"]
                    except json.JSONDecodeError:
                        pass
                
                # Load the rest of messages
                for line in lines[start_idx:]:
                    try:
                        msg = json.loads(line.strip())
                        if "role" in msg:
                            self.messages.append(msg)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

    def save_history(self, abstract: str | None = None) -> None:
        """把当前对话历史覆盖写入 JSONL，防止多次追加导致重复。"""
        # CWD 独立的文件路径解析 (定位在 data/sessions)
        target_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sessions"
        target = target_dir / f"{self.session_id}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # Preserve existing metadata
        existing_meta = {}
        if target.exists():
            try:
                with open(target, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        meta = json.loads(first_line)
                        if "role" not in meta:
                            existing_meta = meta
            except Exception:
                pass
                
        with open(target, "w", encoding="utf-8") as f:
            meta_to_write = existing_meta.copy()
            if abstract:
                meta_to_write["abstract"] = abstract
            if self.role_id:
                meta_to_write["role_id"] = self.role_id
                
            w_ids = set(meta_to_write.get("active_workspace_ids") or [])
            if meta_to_write.get("active_workspace_id"):
                w_ids.add(meta_to_write["active_workspace_id"])
            if self.active_workspace_id:
                w_ids.add(self.active_workspace_id)
                
            g_names = set(meta_to_write.get("active_graph_names") or [])
            if meta_to_write.get("active_graph_name"):
                g_names.add(meta_to_write["active_graph_name"])
            if self.active_graph_name:
                g_names.add(self.active_graph_name)
                
            # Scan current messages for any other referenced workspaces/graphs in tool calls
            for msg in self.messages:
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        if isinstance(tc, dict):
                            func = tc.get("function", {})
                            t_name = func.get("name")
                            args_str = func.get("arguments", "{}")
                            args = {}
                            if isinstance(args_str, str):
                                try:
                                    args = json.loads(args_str)
                                except Exception:
                                    pass
                            elif isinstance(args_str, dict):
                                args = args_str
                                
                            if t_name in ["read_graph", "make_graph", "update_graph"]:
                                g_name = args.get("file_name")
                                if g_name:
                                    g_names.add(g_name)
                            elif t_name in ["get_canvas_graph", "create_canvas_node", "update_canvas_node", "link_canvas_nodes"]:
                                w_id = args.get("workspace_id")
                                if w_id:
                                    w_ids.add(w_id)
                                    
            if w_ids:
                meta_to_write["active_workspace_id"] = list(w_ids)[0]
                meta_to_write["active_workspace_ids"] = list(w_ids)
            if g_names:
                meta_to_write["active_graph_name"] = list(g_names)[0]
                meta_to_write["active_graph_names"] = list(g_names)
                
            if meta_to_write:
                f.write(json.dumps(meta_to_write, ensure_ascii=False) + "\n")
            for msg in self.messages:
                if "role" in msg:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def _fix_orphaned_tool_calls(self) -> None:
        """
        检查历史消息中是否有悬空的 tool_calls（即 assistant 发出了 tool_calls，但没有后续对应的 tool 消息），
        如果有，自动补上一个表示“执行中断/失败”的 tool 消息并保存，防止大模型接口报 400 错误。
        """
        if not self.messages:
            return
            
        # 1. 扫描出所有已存在的 tool 消息的 tool_call_id
        provided_ids = set()
        for msg in self.messages:
            if msg.get("role") == "tool":
                tid = msg.get("tool_call_id")
                if tid:
                    provided_ids.add(tid)
                    
        # 2. 扫描出所有需要补齐的 tool_call_id 及其关联 of tool_name
        orphaned_calls = []  # list of (call_id, name)
        for msg in self.messages:
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            cid = tc.get("id")
                            func_name = tc.get("function", {}).get("name", "unknown")
                            if cid and cid not in provided_ids:
                                orphaned_calls.append((cid, func_name))
                                
        # 3. 如果存在悬空的 tool_calls，进行补齐并保存
        if orphaned_calls:
            log_msg = f"Detected {len(orphaned_calls)} orphaned tool calls in history. Repairing with error status..."
            log.warning(log_msg)
            for cid, func_name in orphaned_calls:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": cid,
                    "content": f"Error: Tool '{func_name}' execution was interrupted due to a system crash or restart. Please try calling it again if needed."
                })
            self.save_history()