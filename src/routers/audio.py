import uuid
from typing import Any, Dict, Literal, Optional
from fastapi import APIRouter, Query, HTTPException, File, UploadFile, Form
from sqlmodel import select

from models.sqlmodel.voice_timbre import VoiceTimbre
from models.sqlmodel.voice_cloned import VoiceCloned
from models.sqlmodel.voice_designed import VoiceDesigned
from services.tts_services.voice_clone_service import voice_clone_service
from services.tts_services.voice_design_service import voice_design_service
from infra.storage.mysql_connector import mysql_connector
from infra.logging.logger import logger as log


audio_router = APIRouter(prefix="/audio", tags=["audio"])

@audio_router.get("/models")
async def list_voice_models(
    model_type: Literal["all", "big", "small"] = Query(default="big", description="Filter model family"),
    provider: str = Query(default="all", description="Filter by provider (volcano, qwen, all)"),
    is_online: Optional[bool] = Query(default=None, description="Filter by online/offline status"),
) -> Dict[str, Any]:
    """
    获取所有的音色列表，并按优先级与模型类型排序、去重
    """
    try:
        async with mysql_connector.session_scope() as session:
            # 查询所有启用的音色，并按 priority DESC, voice_model_type ASC, voice_character ASC 排序
            stmt = select(VoiceTimbre).where(VoiceTimbre.is_enabled == 1)
            
            if provider != "all":
                stmt = stmt.where(VoiceTimbre.provider == provider)
                
            if is_online is not None:
                stmt = stmt.where(VoiceTimbre.is_online == (1 if is_online else 0))
                
            stmt = stmt.order_by(
                VoiceTimbre.priority.desc(),
                VoiceTimbre.voice_model_type.asc(),
                VoiceTimbre.voice_character.asc()
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        seen = set()
        unique_rows = []
        for row in rows:
            # 兼容空类型处理
            actual_model_type = row.voice_model_type
            if not actual_model_type or actual_model_type == "default":
                if row.provider == "volcano":
                    actual_model_type = "small" if row.voice_character == "天才少女" else "big"
                else:
                    actual_model_type = "big"

            # 过滤小模型音色
            if actual_model_type == "small":
                continue

            # 过滤不支持 instruct 的火山大模型音色
            if row.provider == "volcano" and actual_model_type == "big":
                code_lower = row.voice_code.lower()
                if "uranus" not in code_lower and "saturn" not in code_lower:
                    continue

            if model_type != "all" and actual_model_type != model_type:
                continue

            pair = (row.voice_character, actual_model_type)
            if pair not in seen:
                seen.add(pair)
                unique_rows.append((row, actual_model_type))

        def _to_item(row: VoiceTimbre, model_type_val: str) -> dict[str, Any]:
            # 统一性别映射，前端使用 "0" 代表女，"1" 代表男
            sex_val = row.sex
            if sex_val in ("女", "0"):
                sex_val = "0"
            elif sex_val in ("男", "1"):
                sex_val = "1"

            # 根据性别动态转换“少年/少女”的显示
            age_type = row.age_type
            if age_type == "少年/少女":
                if sex_val == "1":
                    age_type = "少年"
                elif sex_val == "0":
                    age_type = "少女"

            return {
                "id": row.id,
                "voice_character": row.voice_character,
                "voice_code": row.voice_code,
                "voice_model_type": model_type_val,
                "note": row.note,
                "is_enabled": bool(row.is_enabled),
                "priority": row.priority,
                "age_type": age_type,
                "sex": sex_val,
                "full_voice": row.full_voice,
                "provider": row.provider,
                "is_online": bool(row.is_online),
            }

        if model_type == "all":
            grouped: dict[str, list[dict[str, Any]]] = {"big": [], "small": []}
            for row, model_type_val in unique_rows:
                grouped.setdefault(model_type_val, []).append(_to_item(row, model_type_val))
            data: Any = grouped
        else:
            data = [_to_item(row, model_type_val) for row, model_type_val in unique_rows]

        return {"code": 200, "message": "ok", "data": data}
    except Exception as e:
        log.error("查询音色列表失败: {}", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询音色列表失败: {str(e)}")


@audio_router.post("/voices/clone")
async def clone_custom_voice(
    file: UploadFile = File(...),
    voice_character: str = Form(...),
    text: Optional[str] = Form(default=None),
    language: int = Form(default=0),
    provider: str = Form(default="volcano")
):
    """
    上传语音样本并提交声音复刻训练任务（支持火山引擎和阿里云Qwen）
    """
    audio_bytes = await file.read()
    filename = file.filename or ""
    audio_format = filename.split(".")[-1].lower() if "." in filename else "mp3"
    if audio_format not in ("wav", "mp3", "ogg", "m4a", "aac", "pcm"):
        audio_format = "mp3"

    if provider == "qwen":
        voice = await voice_clone_service.clone_voice_qwen(
            voice_character=voice_character,
            audio_bytes=audio_bytes,
            audio_format=audio_format
        )
        return {"code": 200, "message": "声音复刻成功", "data": voice}
    else:
        if not text:
            raise HTTPException(status_code=400, detail="火山声音复刻必须提供 prompt 文本")
        voice = await voice_clone_service.clone_voice(
            voice_character=voice_character,
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            prompt_text=text,
            language=language
        )
        return {"code": 200, "message": "声音复刻任务已提交", "data": voice}


@audio_router.get("/voices/cloned")
async def list_cloned_voices(refresh: bool = Query(default=False)):
    """
    获取当前用户所有的自定义复刻音色列表
    """
    try:
        import asyncio
        await asyncio.gather(
            voice_clone_service.sync_volcano_voices(force_refresh=refresh),
            voice_clone_service.sync_qwen_voices(force_refresh=refresh),
            return_exceptions=True
        )
        async with mysql_connector.session_scope() as session:
            stmt = select(VoiceCloned).order_by(VoiceCloned.created_at.desc())
            res = await session.execute(stmt)
            rows = res.scalars().all()
            
        # Resolve update_to chains
        id_map = {r.id: r for r in rows}
        pointed_to_ids = {r.update_to for r in rows if r.update_to is not None}
        
        resolved_rows = []
        for r in rows:
            if r.id in pointed_to_ids:
                continue
            
            curr = r
            visited = {curr.id}
            while curr.update_to is not None and curr.update_to in id_map:
                next_id = curr.update_to
                if next_id in visited:
                    break
                visited.add(next_id)
                curr = id_map[next_id]
            resolved_rows.append(curr)
            
        import re
        filtered_rows = []
        for r in resolved_rows:
            if r.provider == "volcano" and r.custom_speaker_id.startswith("S_") and r.status == 1:
                continue
            if r.provider == "qwen":
                match = re.search(r"\d{8}", r.custom_speaker_id)
                if match and match.group(0) < "20260624":
                    continue
            if not r.demo_audio_url and r.status != 1:
                continue
            filtered_rows.append(r)
            
        return {"code": 200, "message": "ok", "data": filtered_rows}
    except Exception as e:
        log.error("查询自定义复刻音色失败: {}", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询自定义音色失败: {str(e)}")


@audio_router.get("/volcano/voices")
async def list_volcano_voices(
    page_number: int = Query(default=1),
    page_size: int = Query(default=100)
):
    """
    通过火山 Open API 获取当前 AppID 下全部的已购/已训练音色状态
    """
    from utils.volcano_utils import query_volcano_train_statuses
    try:
        res = await query_volcano_train_statuses(page_number, page_size)
        return {"code": 200, "message": "ok", "data": res}
    except Exception as e:
        log.error("查询火山音色状态失败: {}", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询火山音色状态失败: {str(e)}")



@audio_router.delete("/voices/cloned/{voice_id}")
async def delete_cloned_voice(voice_id: int):
    """
    删除特定的自定义复刻音色记录
    """
    await voice_clone_service.delete_cloned_voice(voice_id)
    return {"code": 200, "message": "删除成功"}


@audio_router.get("/voices/cloned/{voice_id}/status")
async def get_cloned_voice_status(voice_id: int):
    """
    手动向火山引擎查询并刷新特定的自定义复刻音色训练状态
    """
    voice = await voice_clone_service.query_and_update_status(voice_id)
    return {"code": 200, "message": "查询刷新成功", "data": voice}


@audio_router.post("/voices/cloned/{voice_id}/upgrade")
async def upgrade_cloned_voice(voice_id: int):
    """
    提交从 V1 升级至 V3 版本的音色克隆升级操作（火山引擎v3）
    """
    voice = await voice_clone_service.upgrade_cloned_voice(voice_id)
    return {"code": 200, "message": "升级任务已提交", "data": voice}


@audio_router.get("/voices/designed/{voice_id}/status")
async def get_designed_voice_status(voice_id: int):
    """
    手动向火山引擎查询并刷新特定的自定义设计音色训练状态
    """
    voice = await voice_clone_service.query_and_update_designed_status(voice_id)
    return {"code": 200, "message": "查询刷新成功", "data": voice}


@audio_router.post("/voices/designed/{voice_id}/upgrade")
async def upgrade_designed_voice(voice_id: int):
    """
    提交从 V1 升级至 V3 版本的音色设计升级操作（火山引擎v3）
    """
    voice = await voice_clone_service.upgrade_designed_voice(voice_id)
    return {"code": 200, "message": "升级任务已提交", "data": voice}




@audio_router.post("/voices/design")
async def design_custom_voice(
    voice_character: str = Form(...),
    custom_speaker_id: str = Form(...),
    text: str = Form(...),
    text_prompt: Optional[str] = Form(default=None),
    image_url: Optional[str] = Form(default=None),
    language: int = Form(default=0)
):
    """
    提交声音设计/声音定制任务（火山引擎v3）
    """
    voice = await voice_design_service.design_voice(
        voice_character=voice_character,
        custom_speaker_id=custom_speaker_id,
        text=text,
        text_prompt=text_prompt,
        image_url=image_url,
        language=language
    )
    return {"code": 200, "message": "声音设计成功", "data": voice}


@audio_router.get("/voices/designed")
async def list_designed_voices():
    """
    获取所有的自定义设计音色列表
    """
    try:
        async with mysql_connector.session_scope() as session:
            stmt = select(VoiceDesigned).order_by(VoiceDesigned.created_at.desc())
            res = await session.execute(stmt)
            rows = res.scalars().all()
            
        # Resolve update_to chains
        id_map = {r.id: r for r in rows}
        pointed_to_ids = {r.update_to for r in rows if r.update_to is not None}
        
        resolved_rows = []
        for r in rows:
            if r.id in pointed_to_ids:
                continue
            
            curr = r
            visited = {curr.id}
            while curr.update_to is not None and curr.update_to in id_map:
                next_id = curr.update_to
                if next_id in visited:
                    break
                visited.add(next_id)
                curr = id_map[next_id]
            resolved_rows.append(curr)
            
        import re
        filtered_rows = []
        for r in resolved_rows:
            if r.provider == "volcano" and r.custom_speaker_id.startswith("S_") and r.status == 1:
                continue
            if r.provider == "qwen":
                match = re.search(r"\d{8}", r.custom_speaker_id)
                if match and match.group(0) < "20260624":
                    continue
            if not r.demo_audio_url and r.status != 1:
                continue
            filtered_rows.append(r)
            
        return {"code": 200, "message": "ok", "data": filtered_rows}
    except Exception as e:
        log.error("查询自定义设计音色失败: {}", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询自定义设计音色失败: {str(e)}")


@audio_router.delete("/voices/designed/{voice_id}")
async def delete_designed_voice(voice_id: int):
    """
    删除特定的自定义设计音色记录
    """
    await voice_design_service.delete_designed_voice(voice_id)
    return {"code": 200, "message": "删除成功"}


@audio_router.post("/timbre")
async def create_unified_timbre(
    is_online: bool = Form(...),
    provider: str = Form(default="volcano"),
    voice_character: str = Form(...),
    file: Optional[UploadFile] = File(default=None),
    ref_audio_url: Optional[str] = Form(default=None),
    text: Optional[str] = Form(default=None),
    tag: Optional[str] = Form(default=None),
    description: Optional[str] = Form(default=None),
    age_type: Optional[str] = Form(default=None),
    sex: Optional[str] = Form(default=None),
    language: int = Form(default=0),
    demo_text: Optional[str] = Form(default=None)
):
    """
    统一音色创建接口：
    - 在线版 (is_online=True)：只支持音频 (克隆) 或 文本 (设计) 之一。
    - 离线版 (is_online=False)：支持音频+文本（克隆），或单文本（设计）。
    """
    # 验证规则
    if is_online:
        if provider == "volcano":
            # 火山在线版：声音克隆（file 或 ref_audio_url）选填文本（text）；声音设计必须提供 text。
            if not file and not ref_audio_url and not text:
                raise HTTPException(status_code=400, detail="在线版必须提供音频文件、参考音频URL或设计文本之一")
        else:
            # 阿里在线版：克隆提供 file 或 ref_audio_url，设计只提供 text，不能都有
            if (file or ref_audio_url) and text:
                raise HTTPException(status_code=400, detail="在线版只支持单音频(克隆)或单文本(设计)创建，不能同时提供音频 and 文本")
            if not file and not ref_audio_url and not text:
                raise HTTPException(status_code=400, detail="在线版必须提供音频文件、参考音频URL或设计文本之一")
    else:
        if not file and not text:
            raise HTTPException(status_code=400, detail="离线版必须提供音频(克隆)、文本(设计)或两者都提供")

    try:
        # 执行创建逻辑
        if is_online:
            if file or ref_audio_url:
                # 在线声音克隆
                if file:
                    audio_bytes = await file.read()
                    filename = file.filename or ""
                    audio_format = filename.split(".")[-1].lower() if "." in filename else "mp3"
                else:
                    import httpx
                    log.info(f"Downloading original reference audio from: {ref_audio_url}")
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        resp = await client.get(ref_audio_url)
                        if resp.status_code != 200:
                            raise HTTPException(status_code=400, detail=f"无法从网络读取参考音频: {ref_audio_url}")
                        audio_bytes = resp.content
                    filename = ref_audio_url.split("/")[-1].split("?")[0]
                    audio_format = filename.split(".")[-1].lower() if "." in filename else "mp3"

                if audio_format not in ("wav", "mp3", "ogg", "m4a", "aac", "pcm"):
                    audio_format = "mp3"

                if provider == "qwen":
                    voice = await voice_clone_service.clone_voice_qwen(
                        voice_character=voice_character,
                        audio_bytes=audio_bytes,
                        audio_format=audio_format,
                        tag=tag,
                        note=description,
                        age_type=age_type,
                        sex=sex,
                        prompt_text=text,
                        demo_text=demo_text
                    )
                    return {"code": 200, "message": "在线声音复刻成功", "data": voice}
                elif provider == "volcano":
                    voice = await voice_clone_service.clone_voice(
                        voice_character=voice_character,
                        audio_bytes=audio_bytes,
                        audio_format=audio_format,
                        prompt_text=text,
                        tag=tag,
                        note=description,
                        age_type=age_type,
                        sex=sex,
                        demo_text=demo_text
                    )
                    return {"code": 200, "message": "在线声音复刻成功", "data": voice}
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"不支持的在线克隆服务商: {provider}"
                    )
            else:
                # 在线声音设计
                custom_speaker_id = f"vdesign_{uuid.uuid4().hex}"
                voice = await voice_design_service.design_voice(
                    voice_character=voice_character,
                    custom_speaker_id=custom_speaker_id,
                    text=demo_text or "你好，我是你的专属AI克隆声音，希望未来可以好好相处哦",
                    text_prompt=text,
                    tag=tag,
                    note=description,
                    age_type=age_type,
                    sex=sex,
                    provider=provider,
                    is_online=True
                )
                return {"code": 200, "message": "在线声音设计成功", "data": voice}
        else:
            # 离线声音克隆/设计
            if file:
                # 离线声音克隆
                audio_bytes = await file.read()
                filename = file.filename or ""
                audio_format = filename.split(".")[-1].lower() if "." in filename else "mp3"
                if audio_format not in ("wav", "mp3", "ogg", "m4a", "aac", "pcm"):
                    audio_format = "mp3"
                
                voice = await voice_clone_service.clone_voice_offline(
                    voice_character=voice_character,
                    audio_bytes=audio_bytes,
                    audio_format=audio_format,
                    prompt_text=text,
                    tag=tag,
                    note=description,
                    age_type=age_type,
                    sex=sex
                )
                return {"code": 200, "message": "离线声音复刻成功", "data": voice}
            else:
                # 离线声音设计
                voice = await voice_design_service.design_voice_offline(
                    voice_character=voice_character,
                    text=text,
                    tag=tag,
                    note=description,
                    age_type=age_type,
                    sex=sex
                )
                return {"code": 200, "message": "离线声音设计成功", "data": voice}
    except Exception as e:
        log.error("创建音色失败: {}", repr(e), exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"创建音色失败: {str(e)}")


from pydantic import BaseModel

class TimbreConfirmRequest(BaseModel):
    selected_id: int
    selected_type: Literal["cloned", "designed"]
    target_name: str
    candidate_ids: list[int]
    tag: Optional[str] = None
    description: Optional[str] = None
    age_type: Optional[str] = None
    sex: Optional[str] = None
    update_to_link_id: Optional[int] = None
    update_to_link_type: Optional[Literal["cloned", "designed"]] = None

@audio_router.post("/timbre/confirm")
async def confirm_unified_timbre(req: TimbreConfirmRequest):
    """
    确认并保存最终选中的音色，并清理未选中的候选音色记录
    """
    try:
        async with mysql_connector.session_scope() as session:
            # 1. 如果有 update_to_link_id，找到原音色记录并建立指向，同时修改其名称释放原名称
            if req.update_to_link_id:
                orig_type = req.update_to_link_type or req.selected_type
                if orig_type == "cloned":
                    orig_voice = await session.get(VoiceCloned, req.update_to_link_id)
                else:
                    orig_voice = await session.get(VoiceDesigned, req.update_to_link_id)
                
                if orig_voice:
                    log.info(f"Linking original voice id={orig_voice.id} to new voice id={req.selected_id}")
                    orig_voice.update_to = req.selected_id
                    # 重新命名以避免唯一键约束冲突
                    orig_voice.voice_character = f"{orig_voice.voice_character}_upgraded_{orig_voice.id}"
                    session.add(orig_voice)

            # 2. 查找并删除除原音色外的其他同名音色记录，避免唯一键约束冲突
            stmt_cloned = select(VoiceCloned).where(VoiceCloned.voice_character == req.target_name)
            orig_type = req.update_to_link_type or req.selected_type
            if req.update_to_link_id and orig_type == "cloned":
                stmt_cloned = stmt_cloned.where(VoiceCloned.id != req.update_to_link_id)
            res_cloned = await session.execute(stmt_cloned)
            existing_cloned = res_cloned.scalars().all()
            for v in existing_cloned:
                log.info(f"Deleting existing voice with name={req.target_name} from voice_cloned, id={v.id}")
                await session.delete(v)

            stmt_designed = select(VoiceDesigned).where(VoiceDesigned.voice_character == req.target_name)
            if req.update_to_link_id and orig_type == "designed":
                stmt_designed = stmt_designed.where(VoiceDesigned.id != req.update_to_link_id)
            res_designed = await session.execute(stmt_designed)
            existing_designed = res_designed.scalars().all()
            for v in existing_designed:
                log.info(f"Deleting existing voice with name={req.target_name} from voice_designed, id={v.id}")
                await session.delete(v)

            # 3. 将选中的候选音色重命名为最终名称，并更新元数据
            if req.selected_type == "cloned":
                selected_voice = await session.get(VoiceCloned, req.selected_id)
                if selected_voice:
                    selected_voice.voice_character = req.target_name
                    if req.tag is not None: selected_voice.tag = req.tag
                    if req.description is not None: selected_voice.note = req.description
                    if req.age_type is not None: selected_voice.age_type = req.age_type
                    if req.sex is not None: selected_voice.sex = req.sex
                    session.add(selected_voice)
                    log.info(f"Renamed cloned candidate id={req.selected_id} to {req.target_name} and updated metadata")
                else:
                    raise HTTPException(status_code=404, detail="找不到选中的克隆候选音色记录")
            else:
                selected_voice = await session.get(VoiceDesigned, req.selected_id)
                if selected_voice:
                    selected_voice.voice_character = req.target_name
                    if req.tag is not None: selected_voice.tag = req.tag
                    if req.description is not None: selected_voice.note = req.description
                    if req.age_type is not None: selected_voice.age_type = req.age_type
                    if req.sex is not None: selected_voice.sex = req.sex
                    session.add(selected_voice)
                    log.info(f"Renamed designed candidate id={req.selected_id} to {req.target_name} and updated metadata")
                else:
                    raise HTTPException(status_code=404, detail="找不到选中的设计候选音色记录")

            # 4. 删除其他未选中的候选音色记录
            for cid in req.candidate_ids:
                if cid == req.selected_id:
                    continue
                if req.selected_type == "cloned":
                    cand = await session.get(VoiceCloned, cid)
                    if cand:
                        log.info(f"Deleting unused cloned candidate id={cid}")
                        await session.delete(cand)
                else:
                    cand = await session.get(VoiceDesigned, cid)
                    if cand:
                        log.info(f"Deleting unused designed candidate id={cid}")
                        await session.delete(cand)

            await session.commit()
        return {"code": 200, "message": "保存成功"}
    except Exception as e:
        log.error("确认并保存自定义音色失败: {}", e, exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"保存自定义音色失败: {str(e)}")


