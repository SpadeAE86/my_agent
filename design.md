# DIYProject 架构设计方案 (v1.0)

## 0. 核心愿景
构建一个高内聚、易扩展、具备多层记忆感知能力且支持复杂任务并行拆解的智能体框架。

---

## 1. 目录结构设计 (Project Layout)
```text
project_root/
│
├─ src/
│   ├─ FastAPI_server.py                # 🚀 统一服务入口
│   ├─ routers/                         # 🌐 API 路由层
│   │   ├─ chat.py                      # 基础对话
│   │   ├─ agent.py                     # 管理子 Agent 声明周期
│   │   ├─ memory.py                    # 记忆增删改查
│   │   └─ task.py                      # 任务状态查看
│   │  
│   ├─ services/                        # 🧩 业务逻辑编排
│   │   ├─ chat_service.py              # 对话流控
│   │   ├─ agent_service.py             # Agent 实例化逻辑
│   │   ├─ memory_service.py            # 跨层记忆同步
│   │   ├─ task_service.py              # 任务下发与聚合
│   │   └─ stream_service.py            # SSE 封装
│   │ 
│   ├─ core/                            # 🔥 Agent 内核 (核心算法与逻辑)
│   │   ├─ agent/
│   │   │   ├─ main_agent.py            # 主控决策层 (Master)
│   │   │   ├─ sub_agent.py             # 任务执行层 (Worker)
│   │   │   ├─ agent_loop.py            # Generator 核心驱动
│   │   │   └─ event.py                 # 事件模型定义
│   │   ├─ planning/
│   │   │   ├─ planner.py               # 任务拆解器
│   │   │   ├─ task_dispatcher.py       # 任务调度
│   │   │   ├─ execution_graph.py       # DAG/并行执行图
│   │   │   └─ plan_validator.py        # 闭环校验
│   │   │
│   │   ├─ memory/                      # 🧠 记忆系统
│   │   │   ├─ memory_manager.py        # 统一调度接口
│   │   │   ├─ short_term.py            # JSONL 实时读写
│   │   │   ├─ mid_term.py              # MD 日志化存储
│   │   │   ├─ long_term.py             # 压缩后的 MD/向量
│   │   │   ├─ summarizer.py            # 背景 Cron 总结逻辑
│   │   │   └─ retriever.py             # 检索增强 (RAG)
│   │   │
│   │   ├─ tools/                       # 🛠️ 严格工具集
│   │   │   ├─ registry.py              # 注册表
│   │   │   ├─ executor.py              # 安全执行容器
│   │   │   ├─ schema.py                # Pydantic 定义
│   │   │   └─ builtin/                 # 内置基础工具
│   │   │
│   │   ├─ skills/                      # 📚 进阶能力 (Skills)
│   │   │   ├─ loader.py                # MD/PY 加载
│   │   │   ├─ selector.py              # 场景化注入
│   │   │   └─ parser.py                # 提示词解析
│   │   │
│   │   ├─ llm/                         # 🤖 LLM 适配
│   │   │   ├─ client.py                # 不同 Provider 适配
│   │   │   ├─ streaming.py             # 流式处理
│   │   │   └─ prompt_builder.py        # 动态模板
│   │   │ 
│   │   ├─ execution/                   # ⚡ 执行环境
│   │   │   ├─ tool_runner.py
│   │   │   ├─ command_runner.py
│   │   │   └─ sandbox.py               # 沙箱隔离
│   │   │ 
│   │   └─ state/                       # 💾 状态管理
│   │       ├─ agent_state.py
│   │       ├─ session_state.py
│   │       └─ task_state.py
│   │
│   ├─ infra/                           # 🏗️ 基础设施 (持久化与三方对接)
│   │   ├─ logging/                     # 日志与回溯
│   │   ├─ storage/                     # Diskcache/FileStore
│   │   ├─ cache/                       # Redis/LocalCache
│   │   ├─ mq/                          # 消息队列
│   │   ├─ scheduler/                   # APScheduler/Worker
│   │   ├─ stream/                      # SSE 格式化
│   │   └─ config/                      # 环境变量与配置
│   │ 
│   ├─ models/                          # 📦 数据模型 (Pydantic/ORM)
│   ├─ exceptions/                      # 异常体系
│   └─ utils/                           # 纯工具函数
│
├─ tests/                               # 单元/集成测试
└─ data/                                # 本地存储目录 (memory/logs/cache)
```

---

## 2. 核心系统详述

### 2.1 记忆架构 (Memory Layer)
设计原则：由快到慢，由散到简。
1.  **短期记忆 (Short-term)**:
    *   **存储**: `data/memory/{user_id}/{session_id}.jsonl`。
    *   **子 Agent**: `subagents/{agent_id}.jsonl`。
    *   **特性**: 记录每一轮的完整请求与响应。
2.  **中期记忆 (Mid-term)**:
    *   **存储**: `data/memory/{user_id}/logs/YYYY/MM/YYYY-MM-DD.md`。
    *   **特性**: 会话结束后，将关键对话提炼为 Markdown 日记，记录重要事件与结论。
3.  **长期记忆 (Long-term)**:
    *   **存储**: `data/memory/{user_id}/MEMORY.md`。
    *   **特性**: 由后台任务 (Cron) 对中期记忆进行深度压缩，归纳人物画像、偏好、长期目标。
4.  **技术实现**:
    *   使用 **Diskcache (SQLite backend)** 处理并发读写文件时的竞争问题。
    *   支持**云端同步**: 预留后端扩展接口（如 Supabase/PostgreSQL）同步 Markdown 片段。

### 2.2 任务规划与子智能体 (Planning & Sub-Agents)
1.  **主 Agent 能力**: 拥有写 `plan.md` 和 `task.json` 的权限。
2.  **工作流**:
    *   复杂请求 → 主 Agent 识别 → 编写 `plan.md` → (用户确认) → 拆分为多个异步任务。
    *   创建子 Agent → 仅下发 `task.json` 规定的目标、约束及必要 Context。
    *   子 Agent 完成任务 → 返回结构化结果 → 主 Agent 汇总，更新 `plan.md` 进度。
3.  **并行逻辑**: 利用 `execution_graph.py` 构建任务依赖树，无依赖任务自动并行。

### 2.3 工具 (Tools) 与 技能 (Skills)
*   **Tools**: 强类型定义。基于 Pydantic 提供严格的 JSON Schema，追求高调用成功率。
*   **Skills**: 软能力注入。基于 Markdown 文档描述使用场景与操作范式，辅助 Agent 理解如何组合使用 Tools。
*   **动态加载**: Agent 在接到某类任务时，框架根据关键词自动挂载相关的 `Skills.md`。

### 2.4 后台异步体系 (Background Tasks)
1.  **优先队列**: 异步任务（如记忆同步、长文本总结、清理）进入队列。
2.  **心跳机制**: 独立线程每 30 分钟触发，控制 Token 预算，按优先级消化队列。
3.  **定时调度**: 使用 APScheduler 处理固定频率任务（如凌晨的长期记忆压缩）。

---

## 3. 待办与后续演进
- [ ] 初始化 `infra/storage` 支持 Diskcache 基础封装。
- [ ] 构建 `core/agent/agent_loop.py` 支持 Generator 模式。
- [ ] 实现第一版 `core/planning/planner.py` 的 MD 导出。
- [ ] 接入 OpenAI/Claude 模型流式接口。

> [!NOTE]
> 该设计优先考虑本地化运行的可靠性，通过文件系统保证数据的可读性，通过 Diskcache 保证并发的鲁棒性。
