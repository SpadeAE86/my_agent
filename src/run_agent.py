# run_agent.py — main_loop 入口, 用于跑一轮看看效果
#
# 用法:
#   cd src
#   python run_agent.py
#   python run_agent.py "帮我看看当前目录下有什么文件"
#
# 这个脚本会:
#   1. 创建 AsyncOpenAI client
#   2. 初始化 ToolManager, 自动发现 builtin 工具
#   3. 创建 Agent 实例
#   4. 调用 main_loop, 打印所有事件

import asyncio
import sys
import logging
from pathlib import Path

# 确保 src 在 Python 路径里
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from openai import AsyncOpenAI
from core.agent.agent import Agent
from core.agent.agent_loop import main_loop
from core.tools.tool_manager import ToolManager


# ─── 配置 ────────────────────────────────────────────────────────
LLM_CONFIG = {
    "client": AsyncOpenAI(
        base_url="https://z.apiyihe.org/v1",
        api_key="sk-TMd7SbPPbVw1JMx0GYKflkWkv8Mzi1tb0B64Y9HqBQ53TaqW",
    ),
    "model": "gemini-3-pro",
}

DEFAULT_QUERY = "你好，请介绍一下你自己，你有什么能力？"

# ─── 事件打印 ─────────────────────────────────────────────────────
def print_event(event):
    """格式化打印事件。"""
    t = event.event_type
    aid = getattr(event, "agent_id", "")

    if t == "status_update":
        print(f"\n⏳ [{aid}] {event.message}")

    elif t == "agent_thought":
        print(f"💭 [{aid}] {event.content[:200]}")

    elif t == "text_chunk":
        if event.is_final:
            print(f"\n📝 [{aid}] Agent 回复:")
            print("─" * 60)
            print(event.content)
            print("─" * 60)
        else:
            print(event.content, end="", flush=True)

    elif t == "tool_call":
        print(f"\n🔧 [{aid}] 调用工具: {event.tool_name}({event.arguments})")

    elif t == "tool_result":
        status = "✅" if event.success else "❌"
        print(f"{status} [{aid}] 工具结果: {event.output[:200] if event.output else event.error}")

    elif t == "task_complete":
        status = "✅ 完成" if event.success else f"⛔ 终止: {event.reason}"
        print(f"\n🏁 [{aid}] {status}")

    elif t == "plan_update":
        print(f"📋 [{aid}] Plan {event.action}: {event.content[:100]}...")

    else:
        print(f"❓ [{aid}] 未知事件: {t}")


# ─── 主函数 ────────────────────────────────────────────────────────
async def run(query: str):
    print("=" * 60)
    print("🚀 DIYProject Agent Runner")
    print("=" * 60)

    # 1. 初始化 ToolManager
    tm = ToolManager()
    discovered = tm.auto_discover()
    print(f"📦 已发现 {discovered} 个内置工具: {tm.list_names()}")

    # 2. 创建 Agent
    agent = Agent(
        user_id="test_user",
        llm=LLM_CONFIG,
        tools=tm.list_names(),
        skills={},
        max_iteration=3,     # 限制 3 轮, 方便测试
        mode="swarm",
        is_base=True,
        max_token=4096,      # 测试用小一点
        tool_manager=tm,
        language="中文",
    )
    print(f"🤖 Agent 已创建: id={agent.agent_id}, session={agent.session_id}")

    # 3. 打印 system prompt (调试用)
    system_sections = agent._build_system_prompt()
    print(f"\n📜 System Prompt 共 {len(system_sections)} 段, "
          f"总长 {sum(len(s) for s in system_sections)} 字符:")
    for i, section in enumerate(system_sections):
        first_line = section.split("\n")[0]
        print(f"   [{i}] {first_line} ({len(section)} chars)")

    # 4. 运行 main_loop
    print(f"\n💬 用户输入: {query}")
    print("=" * 60)

    try:
        async for event in main_loop(agent, query, tool_manager=tm):
            print_event(event)
    except Exception as e:
        print(f"\n💥 运行出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # 5. 打印最终消息历史 (调试用)
    print(f"\n📊 最终消息历史: {len(agent.messages)} 条")
    for i, msg in enumerate(agent.messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")[:80]
        print(f"   [{i}] {role}: {content}...")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    asyncio.run(run(query))
