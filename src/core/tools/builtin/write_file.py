import os
from typing import Any, Literal

from pydantic import Field

from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput
from core.tools.file_manager import file_manager

class WriteFileInput(ToolInput):
    """write_file 工具的入参。"""
    
    target_path: str = Field(
        ..., 
        description="要写入的目标文件的绝对路径或相对工作区的相对路径"
    )
    content: str = Field(
        ...,
        description="打算写入的文件内容文本"
    )
    mode: Literal['overwrite', 'append', 'insert'] = Field(
        default='overwrite',
        description="写入模式：'overwrite'为覆盖，'append'为末尾追加，'insert'为按行插入"
    )
    offset_line: int = Field(
        default=-1,
        description="当 mode 为 'insert' 时必填，代表内容应该插入在该行号的位置前(从 1 开始计)"
    )

class WriteFileOutput(ToolOutput):
    """write_file 工具的出参。"""
    
    target_path: str = Field(default="", description="被写入的目标路径")

async def handle_write_file(
    params: WriteFileInput,
    *,
    agent: Any = None,
    tool_manager: Any = None,
    **_kwargs: Any,
) -> WriteFileOutput:
    
    path = os.path.abspath(params.target_path)
    
    try:
        await file_manager.write_to_file(
            local_path=path,
            content=params.content,
            mode=params.mode,
            offset_line=params.offset_line
        )
        
        mode_desc = {
            "overwrite": "覆盖写入",
            "append": "追加写入",
            "insert": f"插入写入(行{params.offset_line})"
        }.get(params.mode, "未知模式写入")
        
        return WriteFileOutput(
            success=True,
            message=f"成功完成对 {path} 的 {mode_desc}",
            data={
                "target_path": path,
                "mode": params.mode
            },
            target_path=path
        )
        
    except Exception as e:
        return WriteFileOutput(
            success=False,
            message=f"写入文件失败: {type(e).__name__}: {e}",
            target_path=path,
        )

# ─── 工具定义 ─────────────────────────────────────────────────────
tool_def = ToolDef(
    name="write_file",
    description=(
        "用于安全写入、修改本地文件的工具。你在修改或写文件前会申请排它锁定(UUID lock)，避免并发写冲突。"
        "支持全量覆盖、内容追加以及特定行号前的精准插入。需要多批次向文件不同位置注入内容时推荐使用 insert 模式。"
    ),
    input_schema=WriteFileInput,
    output_schema=WriteFileOutput,
    handler=handle_write_file,
    tags=["file", "write", "system"],
    timeout=30.0,
    is_concurrency_safe=True, # The FileManager handles the locks, so the tool itself allows concurrency safe orchestration
)
