import os
from typing import Any

from pydantic import Field

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput
from infra.storage.file_registry import file_registry

class ReadFileInput(ToolInput):
    """read_file 工具的入参。"""
    
    target_path: str = Field(
        ..., 
        description="要读取的目标文件的绝对路径或相对工作区的相对路径"
    )
    start_line: int = Field(
        default=1,
        description="开始读取的行号，默认从第 1 行开始",
        ge=1
    )
    lines_amount: int = Field(
        default=-1,
        description="指定要读取的行数。默认 -1 表示从 start_line 一直读到文件末尾",
        ge=-1
    )

class ReadFileOutput(ToolOutput):
    """read_file 工具的出参。"""
    
    target_path: str = Field(default="", description="读取的文件路径")
    content: str = Field(default="", description="读取到的文件内容片段")
    start_line: int = Field(default=1, description="实际开始读的行落")
    read_lines: int = Field(default=0, description="实际读取的行数")

async def handle_read_file(
    params: ReadFileInput,
    *,
    agent: Any = None,
    tool_manager: Any = None,
    **_kwargs: Any,
) -> ReadFileOutput:
    
    path = os.path.abspath(params.target_path)
    
    try:
        content_str = await file_registry.read_lines(
            local_path=path,
            start_line=params.start_line,
            lines_amount=params.lines_amount
        )
        # 计算实际读了多少行
        read_lines_count = len(content_str.splitlines())
        
        return ReadFileOutput(
            success=True,
            message=f"成功从 {path} 读取了 {read_lines_count} 行",
            data={
                "target_path": path,
                "content": content_str,
                "start_line": params.start_line,
                "read_lines": read_lines_count
            },
            target_path=path,
            content=content_str,
            start_line=params.start_line,
            read_lines=read_lines_count
        )
        
    except Exception as e:
        return ReadFileOutput(
            success=False,
            message=f"读取文件失败: {type(e).__name__}: {e}",
            target_path=path,
            content="",
        )

# ─── 工具定义 ─────────────────────────────────────────────────────
tool_def = ToolDef(
    name="read_file",
    description=(
        "用于安全读取本地文件内容的工具。支持局部读取（通过 start_line 和 lines_amount 进行分页或只读特定区域），"
        "读取时会受到文件读写锁的保护，保证你在读到一致的内容。"
    ),
    input_schema=ReadFileInput,
    output_schema=ReadFileOutput,
    handler=handle_read_file,
    tags=["file", "read", "system"],
    timeout=10.0,
    is_concurrency_safe=True,
)
