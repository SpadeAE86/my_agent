import os
import shlex
import asyncio
from typing import Any, List, Tuple
import re

from pydantic import Field

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput

class RunCommandInput(ToolInput):
    """run_command 工具的入参。"""
    command: str = Field(
        ...,
        description="要执行的 Shell 命令。只允许运行安全的开发工具和命令（如 pytest, python, npm, pnpm, git 等），不允许危险操作。"
    )

class RunCommandOutput(ToolOutput):
    """run_command 工具的出参。"""
    stdout: str = Field(default="", description="标准输出")
    stderr: str = Field(default="", description="标准错误")
    exit_code: int = Field(default=0, description="退出码")

# 永久命令白名单
PERMANENT_WHITELIST = {
    "git", "pytest", "npm", "pnpm", "python", "poetry", "pip", "uv", "ruff", "black"
}

# 禁用某些带有强破坏性参数的命令
BLACK_FLAGS = [
    (r"\bgit\b", r"\bpush\b.*\b--force\b"),
    (r"\bgit\b", r"\breset\b.*\b--hard\b"),
    (r"\brm\b", r"-rf|-r"),
]

def validate_command(cmd_str: str) -> Tuple[bool, str, List[str]]:
    """
    对命令执行安全校验。
    返回: (是否安全, 错误描述, 参数列表)
    """
    try:
        args = shlex.split(cmd_str)
        if not args:
            return False, "命令内容不能为空", []
        
        executable = args[0]
        # 去掉路径和 Windows 扩展名
        executable_name = os.path.basename(executable).lower()
        if executable_name.endswith(".exe"):
            executable_name = executable_name[:-4]
        elif executable_name.endswith(".bat"):
            executable_name = executable_name[:-4]
        elif executable_name.endswith(".cmd"):
            executable_name = executable_name[:-4]

        # 检查可执行命令是否在白名单中
        if executable_name not in PERMANENT_WHITELIST:
            return False, f"可执行程序 '{executable_name}' 不在安全命令白名单中 ({', '.join(PERMANENT_WHITELIST)})", args
            
        # 检查是否包含禁用的危险参数
        for exec_pat, flag_pat in BLACK_FLAGS:
            if re.search(exec_pat, executable_name):
                if re.search(flag_pat, cmd_str):
                    return False, "命令中包含禁用的高风险/破坏性参数 (例如 --force, --hard, -rf 等)", args
                    
        return True, "Safe", args
    except Exception as e:
        return False, f"命令解析失败: {type(e).__name__}: {e}", []

async def handle_run_command(
    params: RunCommandInput,
    *,
    agent: Any = None,
    tool_manager: Any = None,
    **_kwargs: Any,
) -> RunCommandOutput:
    cmd_str = params.command.strip()
    is_safe, err_msg, args = validate_command(cmd_str)
    
    if not is_safe:
        return RunCommandOutput(
            success=False,
            message=f"命令校验未通过: {err_msg}",
            stdout="",
            stderr=err_msg,
            exit_code=-1,
            data={"stdout": "", "stderr": err_msg, "exit_code": -1}
        )
    
    try:
        # 使用 create_subprocess_shell 执行命令并捕获标准输出和错误
        proc = await asyncio.create_subprocess_shell(
            cmd_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout_str = stdout_bytes.decode("utf-8", errors="ignore")
        stderr_str = stderr_bytes.decode("utf-8", errors="ignore")
        exit_code = proc.returncode
        
        success = (exit_code == 0)
        msg = f"命令执行完成，退出码: {exit_code}" if success else f"命令执行失败，退出码: {exit_code}"
        
        return RunCommandOutput(
            success=success,
            message=msg,
            stdout=stdout_str,
            stderr=stderr_str,
            exit_code=exit_code,
            data={
                "stdout": stdout_str,
                "stderr": stderr_str,
                "exit_code": exit_code
            }
        )
        
    except Exception as e:
        return RunCommandOutput(
            success=False,
            message=f"命令执行中发生异常: {type(e).__name__}: {e}",
            stdout="",
            stderr=str(e),
            exit_code=-2,
            data={"stdout": "", "stderr": str(e), "exit_code": -2}
        )

# ─── 工具定义 ─────────────────────────────────────────────────────
tool_def = ToolDef(
    name="run_command",
    description=(
        "在当前工作目录下执行安全的 Shell 命令（仅限开发/构建/测试任务）。"
        "只允许运行白名单内的可执行命令（如 pytest, python, npm, pnpm, git 等），"
        "禁止任何包含高风险/破坏性参数的命令（如 --hard, --force, rm -rf 等）。"
    ),
    input_schema=RunCommandInput,
    output_schema=RunCommandOutput,
    handler=handle_run_command,
    tags=["system", "execute"],
    timeout=60.0,
    is_concurrency_safe=False,
)
