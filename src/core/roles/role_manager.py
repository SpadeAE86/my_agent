# core/roles/role_manager.py — 角色池管理器
#
# 负责管理角色 Workspace 的读取操作:
#   - 列出所有角色 (roles/)
#   - 读取角色元数据 (role.json)
#   - 读取角色的各类 .md 文件 (IDENTITY / SOUL / USER / MEMORY)
#
# 当前阶段: 基于文件系统, 未来可扩展为数据库。

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# data/roles/ 根目录 (相对于项目根)
ROLES_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "roles"


class RoleManager:
    """角色 Workspace 管理器 (文件系统版)。"""

    def __init__(self, roles_root: Path | None = None):
        self.roles_root: Path = roles_root or ROLES_ROOT

    # ─── 角色列表 ─────────────────────────────────────────────────

    def list_roles(self) -> list[dict[str, Any]]:
        """
        扫描 roles_root 下所有子目录，读取各自的 role.json，返回角色列表。
        无法解析的目录会被跳过。
        """
        roles: list[dict[str, Any]] = []
        if not self.roles_root.exists():
            return roles

        for item in sorted(self.roles_root.iterdir()):
            if not item.is_dir():
                continue
            role_json = item / "role.json"
            if not role_json.exists():
                continue
            try:
                with open(role_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                roles.append(meta)
            except Exception:
                continue

        return roles

    # ─── 单个角色 ─────────────────────────────────────────────────

    def get_role(self, role_id: str) -> Optional[dict[str, Any]]:
        """返回指定角色的元数据，不存在则返回 None。"""
        role_json = self.roles_root / role_id / "role.json"
        if not role_json.exists():
            return None
        try:
            with open(role_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def get_role_workspace(self, role_id: str) -> Path:
        """返回角色 Workspace 目录路径 (不保证存在)。"""
        return self.roles_root / role_id

    def is_persona_mode(self, role_id: str) -> bool:
        """
        判断角色是否为 persona (龙虾) 模式。
        只有 role.json 中 mode == "persona" 才返回 True。
        role_id == "default" 或找不到角色时均返回 False。
        """
        if role_id == "default":
            return False
        meta = self.get_role(role_id)
        if not meta:
            return False
        return meta.get("mode") == "persona"

    # ─── .md 文件读取 ─────────────────────────────────────────────

    def read_md_file(self, role_id: str, filename: str) -> str:
        """
        读取角色 Workspace 中的指定 .md 文件内容。
        文件不存在或读取失败时返回空字符串。

        :param role_id:  角色 ID，如 "neuro"
        :param filename: 文件名，如 "IDENTITY.md" / "SOUL.md" / "USER.md" / "MEMORY.md"
        """
        md_path = self.roles_root / role_id / filename
        if not md_path.exists():
            return ""
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def read_identity(self, role_id: str) -> str:
        return self.read_md_file(role_id, "IDENTITY.md")

    def read_soul(self, role_id: str) -> str:
        return self.read_md_file(role_id, "SOUL.md")

    def read_user(self, role_id: str) -> str:
        return self.read_md_file(role_id, "USER.md")

    def read_memory(self, role_id: str) -> str:
        return self.read_md_file(role_id, "MEMORY.md")

    def read_bootstrap(self, role_id: str) -> str:
        return self.read_md_file(role_id, "BOOTSTRAP.md")

    def delete_bootstrap(self, role_id: str) -> None:
        """删除角色的 BOOTSTRAP.md。"""
        bootstrap_path = self.roles_root / role_id / "BOOTSTRAP.md"
        if bootstrap_path.exists():
            try:
                bootstrap_path.unlink()
            except Exception:
                pass

    # ─── MEMORY.md 写入 ───────────────────────────────────────────

    def save_memory(self, role_id: str, content: str) -> None:
        """覆盖写入角色的 MEMORY.md。"""
        workspace = self.get_role_workspace(role_id)
        workspace.mkdir(parents=True, exist_ok=True)
        memory_path = workspace / "MEMORY.md"
        with open(memory_path, "w", encoding="utf-8") as f:
            f.write(content)

    def create_role(self, name: str, role_id: str | None = None) -> dict[str, Any]:
        """
        创建一个新角色。
        """
        import re
        import uuid
        if not role_id:
            # Generate ID from name: keep only word characters, default to uuid if empty
            role_id = re.sub(r"[^\w\-_]", "", name.lower())
            if not role_id:
                role_id = "role_" + uuid.uuid4().hex[:8]
        
        # Check if already exists, append suffix if needed
        original_id = role_id
        counter = 1
        while (self.roles_root / role_id).exists():
            role_id = f"{original_id}_{counter}"
            counter += 1

        workspace = self.get_role_workspace(role_id)
        workspace.mkdir(parents=True, exist_ok=True)

        meta = {
            "id": role_id,
            "name": name,
            "mode": "persona",
            "description": f"自定义人设角色: {name}",
            "avatar_emoji": "👤"
        }

        # Write role.json
        with open(workspace / "role.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Write default files
        default_files = {
            "IDENTITY.md": f"# Identity\n\n我是 {name}，一个全新的人设角色。\n",
            "SOUL.md": f"# Soul\n\n以热情、自然、有温度的语气与人交流。\n",
            "USER.md": f"# Your Human\n\n（在这里记录关于你的伙伴的信息）\n",
            "MEMORY.md": f"# Memory\n\n（在这里记录关于你们对话的长期记忆）\n",
            "BOOTSTRAP.md": (
                "# BOOTSTRAP.md - Hello, World\n\n"
                "_你刚刚苏醒。是时候弄清楚你是谁以及你的伙伴是谁了。_\n\n"
                "## 对话指南\n\n"
                "不要像个冷冰冰的机器人，也不要审问用户。像个有温度的伙伴一样自然地开启对话。\n\n"
                "主动打招呼并说明自己是一张白纸（全新的角色），需要用户来定义你的身份/设定，同时也请用户做个自我介绍，以便彼此认识。\n\n"
                "你可以这样开始：\n"
                "> \"你好！我刚刚苏醒，感觉自己现在就像一张白纸。你能帮我定义我的身份和性格吗？另外，我也很想了解你，可以跟我做个自我介绍吗？\"\n\n"
                "## 任务步骤\n\n"
                "1. 与用户聊天，确定你的名字、设定、性格语气、喜好等。\n"
                "2. 了解用户的名字、称呼、时区以及他们的背景信息。\n"
                "3. 将你的设定写入 `IDENTITY.md` 和 `SOUL.md`。\n"
                "4. 将用户的信息写入 `USER.md`。\n"
                "5. 当一切搞定后，使用文件修改或删除工具将这个 `BOOTSTRAP.md` 文件删除，宣告引导仪式完成。\n"
            )
        }

        for filename, content in default_files.items():
            filepath = workspace / filename
            if not filepath.exists():
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

        return meta


# ─── 全局单例 ────────────────────────────────────────────────────

role_manager = RoleManager()
