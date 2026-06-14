import os
from typing import Any
from pydantic import Field
from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput

class ListDirInput(ToolInput):
    """list_dir 工具的入参。"""
    target_path: str = Field(
        default=".", 
        description="要列出的目标目录路径，可以是绝对路径，或者是相对于工作目录的相对路径"
    )

class ListDirOutput(ToolOutput):
    """list_dir 工具的出参。"""
    target_path: str = Field(default="", description="列出的目录路径")
    items: list[dict] = Field(default_factory=list, description="目录下的文件和子目录列表")

async def handle_list_dir(
    params: ListDirInput,
    *,
    agent: Any = None,
    tool_manager: Any = None,
    **_kwargs: Any,
) -> ListDirOutput:
    
    path = os.path.abspath(params.target_path)
    
    if not os.path.exists(path):
        return ListDirOutput(
            success=False,
            message=f"路径不存在: {path}",
            target_path=path,
            items=[]
        )
        
    if not os.path.is_dir(path):
        return ListDirOutput(
            success=False,
            message=f"该路径不是一个目录: {path}",
            target_path=path,
            items=[]
        )
        
    try:
        items = []
        for name in os.listdir(path):
            item_path = os.path.join(path, name)
            is_dir = os.path.isdir(item_path)
            size = os.path.getsize(item_path) if not is_dir else 0
            mtime = os.path.getmtime(item_path)
            items.append({
                "name": name,
                "is_dir": is_dir,
                "size": size,
                "last_modified": mtime
            })
            
        return ListDirOutput(
            success=True,
            message=f"成功列出目录 {path} 中的 {len(items)} 个项",
            data={
                "target_path": path,
                "items": items
            },
            target_path=path,
            items=items
        )
        
    except Exception as e:
        return ListDirOutput(
            success=False,
            message=f"列出目录内容失败: {type(e).__name__}: {e}",
            target_path=path,
            items=[]
        )

tool_def = ToolDef(
    name="list_dir",
    description="用于读取本地目录的子文件与子文件夹列表，可用来查找需要操作的力导图文件、代码文件等，维护目录卫生。",
    input_schema=ListDirInput,
    output_schema=ListDirOutput,
    handler=handle_list_dir,
    tags=["file", "read", "system"],
    timeout=10.0,
    is_concurrency_safe=True,
)
