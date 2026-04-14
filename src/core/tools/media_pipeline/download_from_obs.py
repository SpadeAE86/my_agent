import os
from pydantic import Field
from models.pydantic.tool_schema import ToolDef, ToolInput, ToolOutput
from utils.obs_utils import download_from_obs
from infra.storage.file_registry import file_registry

class DownloadFromObsInput(ToolInput):
    """从华为云 OBS 下载资源的入参。"""
    obs_path: str = Field(..., description="必定包含在 OBS 上的对象路径，例如 aigc/test/123.mp4。")

async def download_from_obs_tool(params: DownloadFromObsInput) -> ToolOutput:
    try:
        # 调用底层 utils，该 utils 本身集成了 diskcache 处理与 TTL
        real_local_path = await download_from_obs(params.obs_path)
        
        # 将生成的类似 /4f/1af32.val 加入到虚拟 File ID 管理器
        file_id = file_registry.register_file(real_local_path)
        
        content = (
            f"✅ 从 OBS ({params.obs_path}) 拉取已完成。\n"
            f"内部缓存路径已隐去，该文件分配的本次会话通信短指针为: `{file_id}`\n"
            f"【警告】大模型后续流程若需要该视频文件用于抽帧、上传或裁剪等任务，请统一使用此 {file_id} 作为传参变量！"
        )
        return ToolOutput(success=True, content=content)
    except Exception as e:
        return ToolOutput(success=False, content=f"❌ 下载失败，请检查路径是否拼写错误。报错日志: {e}")

# Tool definition for external registry 
download_from_obs_def = ToolDef(
    name="download_from_obs",
    description="专门从外部对象存储拉取视频或图片资源进本地缓存中。这通常是基于云端素材进行创作的第一步。",
    parameters=DownloadFromObsInput,
    func=download_from_obs_tool
)
