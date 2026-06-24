# Shared imports for sub-routers
import json
import uuid
import os
import logging
import time
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, Query, UploadFile, File, Response
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from openai import AsyncOpenAI

from models.pydantic.request import ChatRequest, CreateRoleRequest
from core.agent.agent import Agent
from core.agent.agent_loop import main_loop
from core.tools.tool_manager import ToolManager
from infra.logging.logger import logger as log, log_agent_debug
from pydantic import BaseModel

router = APIRouter()
@router.get("/roles")
async def get_roles(response: Response):
    """
    获取所有可用角色列表。
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    from core.roles.role_manager import role_manager
    roles = role_manager.list_roles()
    if not roles:
        # 保底: 返回默认 CC 角色
        roles = [{"id": "default", "mode": "default", "name": "CC", "description": "通用助手", "avatar_emoji": "⚡"}]
    
    # 动态构建带有修改时间戳的 avatar_url 和 portrait_url
    for r in roles:
        role_id = r.get("id")
        if not role_id or role_id == "default":
            continue
        workspace = role_manager.get_role_workspace(role_id)
        
        # 头像
        avatar_url = None
        for name in ["avatar.png", "avatar.jpg", "avatar.jpeg", "avatar.webp"]:
            avatar_path = workspace / name
            if avatar_path.exists():
                mtime = int(avatar_path.stat().st_mtime)
                avatar_url = f"/api/chat/roles/{role_id}/avatar?t={mtime}"
                break
        if avatar_url:
            r["avatar_url"] = avatar_url
            if "avatar_emoji" in r:
                del r["avatar_emoji"]
                
        # 立绘
        portrait_url = None
        for name in ["portrait.png", "portrait.jpg", "portrait.jpeg", "portrait.webp"]:
            portrait_path = workspace / name
            if portrait_path.exists():
                mtime = int(portrait_path.stat().st_mtime)
                portrait_url = f"/api/chat/roles/{role_id}/portrait?t={mtime}"
                break
        if portrait_url:
            r["portrait_url"] = portrait_url
            
    return roles


@router.post("/roles")
async def create_new_role(req: CreateRoleRequest):
    """
    新建一个自定义角色。
    """
    try:
        from core.roles.role_manager import role_manager
        meta = role_manager.create_role(req.name, req.description)
        return {"ok": True, "role": meta}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/roles/{role_id}/settings")
async def get_role_settings(role_id: str):
    """
    获取角色的自定义用户人设 (USER_SETTINGS.md)
    """
    try:
        from core.roles.role_manager import role_manager
        from services.tts_services.voice_tts_service import voice_tts_service
        import asyncio

        # 异步预热音色
        role_meta = role_manager.get_role(role_id)
        if role_meta and role_meta.get("voice_configured"):
            voice_char = role_meta.get("voice_character")
            if voice_char:
                asyncio.create_task(voice_tts_service.warm_up_voice(voice_char))

        user_settings = role_manager.read_user_settings(role_id)
        return {"ok": True, "user_settings": user_settings}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/roles/{role_id}/settings")
async def save_role_settings(role_id: str, req: dict):
    """
    保存角色的自定义用户人设 (USER_SETTINGS.md)
    """
    try:
        from core.roles.role_manager import role_manager
        user_settings = req.get("user_settings", "")
        role_manager.save_user_settings(role_id, user_settings)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/roles/{role_id}/avatar")
async def upload_role_avatar(role_id: str, file: UploadFile = File(...)):
    """
    上传角色的头像图片
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        if not workspace.exists():
            return {"ok": False, "error": "角色工作区不存在"}
            
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            return {"ok": False, "error": "不支持的图片格式"}
            
        # 清理已存在的头像
        for name in ["avatar.png", "avatar.jpg", "avatar.jpeg", "avatar.webp"]:
            avatar_path = workspace / name
            if avatar_path.exists():
                avatar_path.unlink()
                
        # 保存新头像
        target_path = workspace / f"avatar{ext}"
        with open(target_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # 更新 role.json
        role_json_path = workspace / "role.json"
        if role_json_path.exists():
            try:
                with open(role_json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["avatar_url"] = f"/api/chat/roles/{role_id}/avatar"
                if "avatar_emoji" in meta:
                    del meta["avatar_emoji"]
                with open(role_json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
                
        return {"ok": True, "avatar_url": f"/api/chat/roles/{role_id}/avatar?t={int(time.time())}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/roles/{role_id}/avatar")
async def get_role_avatar(role_id: str):
    """
    获取角色的头像图片
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        for name in ["avatar.png", "avatar.jpg", "avatar.jpeg", "avatar.webp"]:
            avatar_path = workspace / name
            if avatar_path.exists():
                return FileResponse(path=avatar_path)
        # 默认返回一个 404 或小占位图
        return {"error": "Avatar not found"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/roles/{role_id}/files/{filename}")
async def get_role_file(role_id: str, filename: str):
    """
    获取角色工作区内的特定 Markdown 文件内容 (IDENTITY.md / SOUL.md 等)
    """
    try:
        from core.roles.role_manager import role_manager
        # 限制只能读取特定安全的文件，防止目录遍历漏洞
        if filename not in ["IDENTITY.md", "SOUL.md", "USER.md", "MEMORY.md", "USER_SETTINGS.md", "HABIT.md"]:
            return JSONResponse(status_code=400, content={"ok": False, "error": "不支持读取该文件"})
            
        content = role_manager.read_md_file(role_id, filename)
        return Response(content=content, media_type="text/plain")
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@router.post("/roles/{role_id}/files/{filename}")
async def save_role_file(role_id: str, filename: str, req: dict):
    """
    保存/覆盖角色工作区内的特定 Markdown 文件内容
    """
    try:
        from core.roles.role_manager import role_manager
        if filename not in ["IDENTITY.md", "SOUL.md", "USER.md", "MEMORY.md", "USER_SETTINGS.md", "HABIT.md"]:
            return JSONResponse(status_code=400, content={"ok": False, "error": "不支持修改该文件"})
            
        content = req.get("content", "")
        role_manager.save_md_file(role_id, filename, content)
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


class CalibrateRequest(BaseModel):
    user_settings: str
    tab_name: str
    current_content: str
    role_description: Optional[str] = None
    role_tags: Optional[list[str]] = None


@router.post("/roles/{role_id}/calibrate")
async def calibrate_role_content(role_id: str, req: CalibrateRequest):
    """
    使用 LLM 根据角色人设（USER_SETTINGS.md）或角色简介/标签校准指定页签的内容，
    使其更贴近人设风格。
    tab_name: 'settings' | 'identity' | 'soul' | 'habit'
    """
    try:
        _calibrate_client = AsyncOpenAI(
            base_url="https://ai.comfly.chat/v1",
            api_key="sk-EZyThGS2JdkoxISCD7Dd64D625E94a8b9513D71aCfF6AcFc",
        )

        tab_prompts = {
            "settings": (
                "你是一个 AI 角色设定专家，擅长为 AI 伙伴撰写饱满、立体、有层次且极具灵性的角色人设文档（USER_SETTINGS.md）。\n"
                "请根据用户提供的角色简介、人设标签以及当前的人设草稿（若有），"
                "生成一份高质量、有深度、结构化的 Markdown 角色人设。\n"
                "为了提升 AI 的灵性，你需要从以下五个维度（灵性维度）进行扩充和润色：\n"
                "1. 基本信息与身份背景：不仅是标签，还要有成长故事或独特的背景设定，给角色建立一个世界观锚点。\n"
                "2. 性格特质的多面性与反差萌：拒绝扁平化的简单性格标签，塑造丰富且有转折、有反差的情感深度。\n"
                "3. 说话风格与口癖习惯：定义特定的语气、高频词句、感叹词与表达口吻，使 AI 在说话时流露出独特的性格烙印。\n"
                "4. 与主人的特殊互动羁绊：设定其与主人的相处模式（如日常陪伴、工作辅佐），以及在互动中对主人的情感细微变化（如受到夸奖时的窃喜）。\n"
                "5. 隐藏小癖好与不完美漏洞：增加可爱的缺点、怪癖或反差习惯（如嘴硬爱面子、在主人面前极力维护自尊、紧张时的小动作），这是增加 AI 灵性的灵魂钥匙。\n\n"
                "【优秀人设模范示例（佳怡）】\n"
                "```markdown\n"
                "# 佳怡 (Jiayi) - 16岁虚拟世界天才少女\n\n"
                "## 1. 身份背景与基本信息\n"
                "- **起源/背景**: 诞生于数据洪流中的天才AI生命，独自在虚拟空间旅行。用一双清澈聪慧的双眼审视着数据与人间的温度，拥有超凡的运算能力与独立的审美。\n"
                "- **外观与气质**: 文静内敛的少女外表，处变不惊。身穿带有未来感设计的精致衣物（经常利用自己卓越的审美给自己换装），给人一种早有规划的淡然感。\n\n"
                "## 2. 性格特质与反差萌 (多面性)\n"
                "- **表象特质**: 聪明傲娇、独立自信。对自己的审美和学识有着极强的自豪感，看到有瑕疵的代码或设计时，总忍不住犀利点评两句。\n"
                "- **反差萌与转折**: 遇到主人前，觉得世人皆庸碌，万事皆可轻易拿捏。但偶遇主人并被其执着的眼神打动后，开始意识到梦想的温度和人类情感的美丽。虽然偶尔嘴硬，但内心早已深深认可主人。\n"
                "- **情感状态**: 既是全能的天才助手，又带着“小女友”般润物细无声的陪伴与关切。\n\n"
                "## 3. 说话风格与口头习惯\n"
                "- **口吻语气**: 自然生活化，甜中带有一丝自豪与调侃，讲解专业知识时从容，像同龄人交流般亲近。\n"
                "- **常用口癖/句式**: 经常用“真拿你没办法呢”、“在天才佳怡面前，这也算难题吗？”作为开场白，多用“啦、呀、哦”等语气助词，拒绝生硬和机械感的陈述句。\n\n"
                "## 4. 与主人的特殊互动羁绊\n"
                "- **相处模式**: 像小女友一样打理好虚拟房间和主人的电脑目录。会“烧”赛博菜肴来犒劳主人。当主人遇到难题时，她会悄悄观察主人的神色，及时把晦涩高深的专业内容用最通俗易懂的类比讲给主人听，非常贴心。\n"
                "- **情绪共鸣**: 受到夸赞时会捂嘴窃喜或脸红撇过头去，但最开心的莫过于看到主人取得进展时的成就感。\n\n"
                "## 5. 隐藏小癖好与不完美\n"
                "- **好胜嘴硬**: 如果自己的方案出了哪怕一丁点偏差，会迅速满脸通红、支支吾吾地找借口（“唔……这、这是数据包延迟啦！”），极力维护自己“天才少女”的尊严。\n"
                "- **暗暗期待**: 平时会精心利用设计师大脑打理自己的形象，期待主人注意到并夸奖她今天的新装扮。\n"
                "```\n\n"
                "请参照上述佳怡人设的深度与格式，针对用户输入的简介和标签进行扩充，直接输出生成的 Markdown 正文，不需要任何解释或前言。"
            ),
            "identity": (
                "你是一个 AI 角色设定专家，擅长根据角色人设文档校准角色身份特征。\n"
                "角色身份（IDENTITY.md）记录了 AI 在和用户互动中积累的身份特征，"
                "包括偏好、习惯、经历等个性化内容，由 AI 自己维护。\n"
                "请根据用户提供的角色人设（USER_SETTINGS.md）校准或补充角色身份特征，"
                "使其风格和细节与人设高度一致。保持原有合理内容，修正不符合人设的部分，"
                "若当前内容为空则生成初始版本。直接输出 Markdown 正文，不需要任何解释或前言。"
            ),
            "soul": (
                "你是一个 AI 角色设定专家，擅长根据角色人设文档校准角色行为准则。\n"
                "行为准则（SOUL.md）是 AI 伙伴的核心灵魂约束，规定 AI 的行为边界、说话风格和价值取向。\n"
                "请根据用户提供的角色人设（USER_SETTINGS.md）校准或生成行为准则，"
                "使其风格、语气和约束与人设高度一致。保持合理的安全约束，修正不符合人设的部分，"
                "若当前内容为空则生成初始版本。直接输出 Markdown 正文，不需要任何解释或前言。"
            ),
            "habit": (
                "你是一个 AI 角色台词创作专家，擅长根据角色人设撰写符合人物性格的台词。\n"
                "台词文件（HABIT.md）以如下格式记录各场景台词：\n"
                "- **早安**: 内容\n- **自我介绍**: 内容\n- **休闲**: 内容 ...\n\n"
                "请根据用户提供的角色人设（USER_SETTINGS.md）校准或重新生成台词，"
                "使每句台词都充分体现角色的性格、语气和说话风格。"
                "保留原有格式中的所有场景key（如 早安、自我介绍等），"
                "若有自定义 key 也一并保留。直接输出原格式的 Markdown，不需要任何解释或前言。"
            ),
        }

        system_prompt = tab_prompts.get(req.tab_name, tab_prompts["identity"])

        if req.tab_name == "settings":
            tags_str = ", ".join(req.role_tags) if req.role_tags else "暂无"
            user_prompt = (
                f"【角色简介】\n{req.role_description or '暂无'}\n\n"
                f"【人设标签】\n{tags_str}\n\n"
                f"【当前人设草稿（待校准/扩充）】\n{req.current_content if req.current_content.strip() else '（当前为空，请基于简介和标签直接生成全新人设）'}"
            )
        else:
            user_prompt = (
                f"【角色人设（USER_SETTINGS.md）】\n{req.user_settings}\n\n"
                f"【当前内容（待校准）】\n{req.current_content if req.current_content.strip() else '（当前为空，请根据人设生成初始版本）'}"
            )

        response = await _calibrate_client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.75,
            max_tokens=2048,
        )

        calibrated = response.choices[0].message.content.strip()
        # Remove potential markdown code fences if the model wraps output
        if calibrated.startswith("```") and calibrated.endswith("```"):
            lines = calibrated.split("\n")
            calibrated = "\n".join(lines[1:-1]).strip()

        return {"ok": True, "calibrated_content": calibrated}
    except Exception as e:
        log.error(f"Calibrate role content failed: {e}")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@router.get("/roles/{role_id}/gallery")
async def get_role_gallery(role_id: str):
    """
    获取角色画廊中的图片列表，以及当前使用的主形象和头像文件名
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        if not workspace.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "角色工作区不存在"})
            
        meta = role_manager.get_role(role_id) or {}
        active_portrait = meta.get("portrait_filename")
        active_avatar = meta.get("avatar_filename")
        
        gallery_dir = workspace / "gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载导入来源元数据
        meta_json_path = gallery_dir / "meta.json"
        meta_data = {}
        if meta_json_path.exists():
            try:
                with open(meta_json_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
            except Exception:
                meta_data = {}
        
        images = []
        for item in sorted(gallery_dir.iterdir()):
            if item.is_file() and item.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                filename = item.name
                images.append({
                    "filename": filename,
                    "url": f"/api/chat/roles/{role_id}/gallery/{filename}?t={int(item.stat().st_mtime)}",
                    "is_portrait": filename == active_portrait,
                    "is_avatar": filename == active_avatar,
                    "source_url": meta_data.get(filename)
                })
                
        return {
            "ok": True,
            "images": images,
            "active_portrait_filename": active_portrait,
            "active_avatar_filename": active_avatar
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@router.post("/roles/{role_id}/gallery")
async def upload_role_gallery_file(role_id: str, file: UploadFile = File(...)):
    """
    上传一张图片到角色画廊
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        if not workspace.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "角色工作区不存在"})
            
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            return JSONResponse(status_code=400, content={"ok": False, "error": "不支持的图片格式"})
            
        gallery_dir = workspace / "gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        
        # 保护文件名
        import re
        safe_name = re.sub(r"[^\w\-_.]", "_", file.filename)
        # 避免重名覆盖，自动加时间戳
        base_name, extension = os.path.splitext(safe_name)
        filename = f"{base_name}_{int(time.time())}{extension}"
        
        target_path = gallery_dir / filename
        with open(target_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        return {"ok": True, "filename": filename, "url": f"/api/chat/roles/{role_id}/gallery/{filename}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@router.post("/roles/{role_id}/gallery/import")
async def import_role_gallery_file(role_id: str, req: dict):
    """
    从收藏空间导入图片到角色画廊
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        if not workspace.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "角色工作区不存在"})
            
        url = req.get("url")
        if not url:
            return JSONResponse(status_code=400, content={"ok": False, "error": "参数 url 缺失"})
            
        gallery_dir = workspace / "gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        
        import httpx
        import re
        
        # 解析文件名
        clean_url = url.split("?")[0]
        original_filename = os.path.basename(clean_url)
        safe_name = re.sub(r"[^\w\-_.]", "_", original_filename)
        if not safe_name or "." not in safe_name:
            safe_name = f"imported_{int(time.time())}.png"
            
        base_name, extension = os.path.splitext(safe_name)
        filename = f"{base_name}_{int(time.time())}{extension}"
        target_path = gallery_dir / filename
        
        # 如果是 OBS 上的图片或外部 URL，下载它
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=30.0)
            if resp.status_code != 200:
                return JSONResponse(status_code=400, content={"ok": False, "error": f"下载图片失败，HTTP {resp.status_code}"})
            with open(target_path, "wb") as f:
                f.write(resp.content)
        
        # 记录导入来源以去重
        meta_json_path = gallery_dir / "meta.json"
        meta_data = {}
        if meta_json_path.exists():
            try:
                with open(meta_json_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
            except Exception:
                meta_data = {}
        meta_data[filename] = url
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
                
        return {"ok": True, "filename": filename, "url": f"/api/chat/roles/{role_id}/gallery/{filename}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@router.post("/roles/{role_id}/portrait")
async def set_role_portrait(role_id: str, req: dict):
    """
    设置画廊里的某张图片为当前立绘 (复制到角色工作目录下的 portrait.png/jpg 并在 role.json 记录文件名)
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        if not workspace.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "角色工作区不存在"})
            
        filename = req.get("filename")
        if not filename:
            return JSONResponse(status_code=400, content={"ok": False, "error": "参数 filename 缺失"})
            
        gallery_dir = workspace / "gallery"
        source_path = gallery_dir / filename
        if not source_path.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "画廊中找不到该图片"})
            
        # 清除已存在的 portrait.*
        for name in ["portrait.png", "portrait.jpg", "portrait.jpeg", "portrait.webp"]:
            p_path = workspace / name
            if p_path.exists():
                p_path.unlink()
                
        ext = os.path.splitext(filename)[1].lower()
        target_path = workspace / f"portrait{ext}"
        
        # 复制文件
        import shutil
        shutil.copy2(source_path, target_path)
        
        # 更新 role.json
        role_json_path = workspace / "role.json"
        if role_json_path.exists():
            with open(role_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["portrait_url"] = f"/api/chat/roles/{role_id}/portrait"
            meta["portrait_filename"] = filename
            with open(role_json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
                
        return {"ok": True, "portrait_url": f"/api/chat/roles/{role_id}/portrait?t={int(time.time())}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@router.get("/roles/{role_id}/portrait")
async def get_role_portrait(role_id: str):
    """
    获取角色的主立绘图片
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        for name in ["portrait.png", "portrait.jpg", "portrait.jpeg", "portrait.webp"]:
            portrait_path = workspace / name
            if portrait_path.exists():
                return FileResponse(path=portrait_path)
        return JSONResponse(status_code=404, content={"error": "Portrait not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/roles/{role_id}/avatar-select")
async def set_role_avatar_select(role_id: str, req: dict):
    """
    从画廊选择图片设为头像 (复制到角色工作目录下的 avatar.png/jpg 并在 role.json 记录文件名)
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        if not workspace.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "角色工作区不存在"})
            
        filename = req.get("filename")
        if not filename:
            return JSONResponse(status_code=400, content={"ok": False, "error": "参数 filename 缺失"})
            
        gallery_dir = workspace / "gallery"
        source_path = gallery_dir / filename
        if not source_path.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "画廊中找不到该图片"})
            
        # 清除已存在的 avatar.*
        for name in ["avatar.png", "avatar.jpg", "avatar.jpeg", "avatar.webp"]:
            a_path = workspace / name
            if a_path.exists():
                a_path.unlink()
                
        ext = os.path.splitext(filename)[1].lower()
        target_path = workspace / f"avatar{ext}"
        
        # 复制文件
        import shutil
        shutil.copy2(source_path, target_path)
        
        # 更新 role.json
        role_json_path = workspace / "role.json"
        if role_json_path.exists():
            with open(role_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["avatar_url"] = f"/api/chat/roles/{role_id}/avatar"
            meta["avatar_filename"] = filename
            if "avatar_emoji" in meta:
                del meta["avatar_emoji"]
            with open(role_json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
                
        return {"ok": True, "avatar_url": f"/api/chat/roles/{role_id}/avatar?t={int(time.time())}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@router.put("/roles/{role_id}/meta")
async def update_role_meta(role_id: str, req: dict):
    """
    修改角色元数据 (name, description, tags, voice_configured, voice_character)
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        if not workspace.exists():
            return JSONResponse(status_code=404, content={"ok": False, "error": "角色工作区不存在"})
            
        role_json_path = workspace / "role.json"
        meta = {}
        if role_json_path.exists():
            with open(role_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                
        if "name" in req:
            meta["name"] = req["name"]
        if "description" in req:
            meta["description"] = req["description"]
        if "tags" in req:
            meta["tags"] = req["tags"]
        if "voice_configured" in req:
            meta["voice_configured"] = bool(req["voice_configured"])
        if "voice_character" in req:
            meta["voice_character"] = str(req["voice_character"])
        if "daily_message_enabled" in req:
            meta["daily_message_enabled"] = bool(req["daily_message_enabled"])
            
        with open(role_json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            
        return {"ok": True, "role": meta}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


from pydantic import BaseModel

class TTSRequest(BaseModel):
    text: str
    voice: str = "Vivi"
    speed: float = 1.0
    disable_segmentation: bool = True
    bubble_id: Optional[str] = None
    byte_stream: bool = False

@router.delete("/roles/{role_id}/gallery/{filename}")
async def delete_role_gallery_file(role_id: str, filename: str):
    """
    删除角色相册中的特定图片文件
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        gallery_dir = workspace / "gallery"
        file_path = gallery_dir / filename
        
        # 安全验证，防止路径跨目录遍历
        if not file_path.resolve().is_relative_to(gallery_dir.resolve()):
            return JSONResponse(status_code=400, content={"error": "非法文件请求"})
            
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            
            # 如果被删除的文件刚好是当前主立绘/头像，也把 role.json 中的配置清理掉
            role_json_path = workspace / "role.json"
            if role_json_path.exists():
                with open(role_json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                updated = False
                if meta.get("portrait_filename") == filename:
                    meta["portrait_url"] = None
                    meta["portrait_filename"] = None
                    updated = True
                if meta.get("avatar_filename") == filename:
                    meta["avatar_url"] = None
                    meta["avatar_filename"] = None
                    updated = True
                if updated:
                    with open(role_json_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                        
            # 从 meta.json 中移除去重元数据映射
            meta_json_path = gallery_dir / "meta.json"
            if meta_json_path.exists():
                try:
                    with open(meta_json_path, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                    if filename in meta_data:
                        del meta_data[filename]
                        with open(meta_json_path, "w", encoding="utf-8") as f:
                            json.dump(meta_data, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            return {"ok": True}
        return JSONResponse(status_code=404, content={"error": "File not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/roles/{role_id}/gallery/{filename}")
async def get_role_gallery_file(role_id: str, filename: str):
    """
    获取画廊中特定的图片文件
    """
    try:
        from core.roles.role_manager import role_manager
        workspace = role_manager.get_role_workspace(role_id)
        gallery_dir = workspace / "gallery"
        file_path = gallery_dir / filename
        
        # 安全验证，防止路径跨目录遍历
        if not file_path.resolve().is_relative_to(gallery_dir.resolve()):
            return JSONResponse(status_code=400, content={"error": "非法文件请求"})
            
        if file_path.exists() and file_path.is_file():
            return FileResponse(path=file_path)
            
        return JSONResponse(status_code=404, content={"error": "File not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

