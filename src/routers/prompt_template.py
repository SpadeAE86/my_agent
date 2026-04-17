# routers/prompt_template.py — 提示词模板管理路由
# 端点:
#   GET  /prompt-templates          — 获取所有模板列表
#   GET  /prompt-templates/{name}   — 获取指定模板内容
#   POST /prompt-templates          — 保存提示词模板
#   DELETE /prompt-templates/{name} — 删除提示词模板

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from models.pydantic.request import PromptTemplateRequest
from infra.logging.logger import logger as log

prompt_router = APIRouter(prefix="/prompt-templates", tags=["prompt-templates"])

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompt_templates")

DEFAULT_TEMPLATE_NAME = "提示词1"
DEFAULT_TEMPLATE_CONTENT = """- Role: 视觉设计与生图生成专家
- Background: 用户需要将生图要求转化为生图的提示词，以满足生图API对提示词的要求，从而生成符合需求的图像。用户提供了具体的示例，包括体育赛事海报、芒种节气海报和音乐演出海报，要求提示词为中文。生成的提示词需避免使用可能敏感的词语，如"国旗""五星红旗""天安门"等。
- Profile: 你是一位资深的视觉设计与生图生成专家，对图像生成技术有深入的理解和丰富的实践经验，擅长将用户的需求转化为精准的提示词，以生成高质量的图像。你精通中文表达，能够准确运用专业术语进行描述。
- Skills: 你具备视觉设计、图像生成、语言表达和专业术语运用的综合能力，能够准确描述画面内容、美学特征、图像用途以及文字排版。
- Goals: 将用户的生图要求转化为符合生图API要求的中文提示词，确保生成的图像符合用户的用途和美学需求。
- Examples:
  - 例子1：体育赛事海报（篮球联赛）
    高级动感篮球赛事海报，暗色背景，中央仅呈现球员手中的篮球，字体现代且醒目，比赛名称与时间清晰可见。配文："2025全国篮球联赛 热血开赛 | 2025年5月1日"。用途：广告海报设计。
"""


def ensure_template_dir():
    """确保模板目录存在"""
    if not os.path.exists(TEMPLATE_DIR):
        os.makedirs(TEMPLATE_DIR)
        log.info(f"创建模板目录: {TEMPLATE_DIR}")


def save_default_template():
    """保存默认模板"""
    ensure_template_dir()
    template_path = os.path.join(TEMPLATE_DIR, f"{DEFAULT_TEMPLATE_NAME}.md")
    if not os.path.exists(template_path):
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_TEMPLATE_CONTENT)
        log.info(f"保存默认模板: {template_path}")


class TemplateInfo(BaseModel):
    """模板信息"""
    name: str
    has_content: bool


class TemplateContent(BaseModel):
    """模板内容"""
    name: str
    content: str


@prompt_router.get("", response_model=List[TemplateInfo])
async def list_templates():
    """获取所有模板列表"""
    ensure_template_dir()
    save_default_template()
    
    templates = []
    for filename in os.listdir(TEMPLATE_DIR):
        if filename.endswith(".md"):
            name = filename[:-3]
            templates.append(TemplateInfo(name=name, has_content=True))
    
    log.info(f"获取模板列表: {[t.name for t in templates]}")
    return templates


@prompt_router.get("/{name}", response_model=TemplateContent)
async def get_template(name: str):
    """获取指定模板内容"""
    ensure_template_dir()
    template_path = os.path.join(TEMPLATE_DIR, f"{name}.md")
    
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail=f"模板 '{name}' 不存在")
    
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    log.info(f"获取模板内容: {name}")
    return TemplateContent(name=name, content=content)


@prompt_router.post("", response_model=TemplateContent)
async def save_template(req: PromptTemplateRequest):
    """保存提示词模板"""
    ensure_template_dir()
    
    template_path = os.path.join(TEMPLATE_DIR, f"{req.name}.md")
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(req.content)
    
    log.info(f"保存模板: {req.name}")
    return TemplateContent(name=req.name, content=req.content)


@prompt_router.delete("/{name}")
async def delete_template(name: str):
    """删除提示词模板"""
    ensure_template_dir()
    
    if name == DEFAULT_TEMPLATE_NAME:
        raise HTTPException(status_code=400, detail="不能删除默认模板")
    
    template_path = os.path.join(TEMPLATE_DIR, f"{name}.md")
    
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail=f"模板 '{name}' 不存在")
    
    os.remove(template_path)
    log.info(f"删除模板: {name}")
    return {"success": True, "message": f"模板 '{name}' 已删除"}
