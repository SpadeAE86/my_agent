import json
from typing import Any, Optional
from pydantic import Field
from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput
from core.roles.role_manager import role_manager

class RenameRoleInput(ToolInput):
    """rename_role 工具的入参。"""
    
    name: str = Field(
        ...,
        description="当前角色的新名称 (例如 'Neuro', '赛博伴侣' 等)"
    )
    description: Optional[str] = Field(
        default=None,
        description="角色的新描述/简介"
    )
    avatar_emoji: Optional[str] = Field(
        default=None,
        description="角色的新头像 Emoji (例如 🧠, 👤, 🤖)"
    )

class RenameRoleOutput(ToolOutput):
    """rename_role 工具的出参。"""
    role_id: str = Field(default="", description="修改角色的 ID")
    name: str = Field(default="", description="修改后的名称")

async def handle_rename_role(
    params: RenameRoleInput,
    *,
    agent: Any = None,
    **_kwargs: Any,
) -> RenameRoleOutput:
    if agent is None:
        return RenameRoleOutput(
            success=False,
            message="执行失败：上下文缺少 agent 实例"
        )
    
    role_id = agent.role_id
    if role_id == "default":
        return RenameRoleOutput(
            success=False,
            message="默认 CC 角色不支持修改名称"
        )
        
    try:
        workspace = role_manager.get_role_workspace(role_id)
        role_json_path = workspace / "role.json"
        
        meta = {}
        if role_json_path.exists():
            with open(role_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        
        meta["name"] = params.name
        if params.description is not None:
            meta["description"] = params.description
        if params.avatar_emoji is not None:
            meta["avatar_emoji"] = params.avatar_emoji
            
        with open(role_json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            
        # 修改 IDENTITY.md 顶部的名称
        identity_path = workspace / "IDENTITY.md"
        identity_content = role_manager.read_identity(role_id)
        
        # 简单的名字和设定替换逻辑，如果 IDENTITY.md 包含默认占位符，直接全量重写它
        if "我是 " in identity_content and "一个全新的人设角色" in identity_content:
            new_identity = (
                f"# Identity\n\n"
                f"- **Name:** {params.name}\n"
                f"- **Creature:** AI Companion\n"
                f"- **Vibe:** Warm, helpful, and natural\n"
            )
            if params.avatar_emoji:
                new_identity += f"- **Emoji:** {params.avatar_emoji}\n"
            with open(identity_path, "w", encoding="utf-8") as f:
                f.write(new_identity)
        else:
            # 否则，尝试寻找并替换 Name 行
            lines = identity_content.split("\n")
            replaced = False
            for idx, line in enumerate(lines):
                if line.strip().startswith("- **Name:**") or line.strip().startswith("- Name:"):
                    lines[idx] = f"- **Name:** {params.name}"
                    replaced = True
                    break
            if not replaced:
                # 寻找 Name 并替换，或在开头追加
                lines.insert(2, f"- **Name:** {params.name}")
            with open(identity_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        return RenameRoleOutput(
            success=True,
            message=f"已成功将角色名称修改为: {params.name}",
            role_id=role_id,
            name=params.name
        )
        
    except Exception as e:
        return RenameRoleOutput(
            success=False,
            message=f"修改名称失败: {type(e).__name__}: {e}"
        )

# ─── 工具定义 ─────────────────────────────────────────────────────
tool_def = ToolDef(
    name="rename_role",
    description="用于修改当前人设角色的基本设定（名称、简介、头像 Emoji 等）。在首次苏醒或开箱阶段修改设定时使用。",
    input_schema=RenameRoleInput,
    output_schema=RenameRoleOutput,
    handler=handle_rename_role,
    tags=["role", "system"],
    timeout=10.0,
)
