from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from sqlmodel import select, desc
from datetime import datetime

from infra.logging.logger import logger as log
from models.sqlmodel.agent_daily_message import AgentDailyMessage
from infra.storage.mysql_connector import mysql_connector

router = APIRouter(prefix="/agent", tags=["agent"])

@router.get("/daily-messages")
async def list_daily_messages(
    user_id: str = Query(default="default_user", description="用户ID"),
    agent_id: Optional[str] = Query(default=None, description="过滤特定Agent ID"),
    limit: int = Query(default=15, ge=1, le=100, description="返回的最大留言卡片数量")
):
    """
    获取用户的晚间留言卡片列表，用于留言看板展示。
    按照日期倒序排列，最新的留言排在最前面。
    """
    try:
        async with mysql_connector.session_scope() as session:
            stmt = select(AgentDailyMessage).where(AgentDailyMessage.user_id == user_id)
            if agent_id:
                stmt = stmt.where(AgentDailyMessage.agent_id == agent_id)
            
            stmt = stmt.order_by(desc(AgentDailyMessage.date_str)).limit(limit)
            results = await session.execute(stmt)
            messages = list(results.scalars().all())
            
            payload = []
            for msg in messages:
                payload.append({
                    "id": msg.id,
                    "user_id": msg.user_id,
                    "agent_id": msg.agent_id,
                    "date_str": msg.date_str,
                    "content": msg.content,
                    "created_at": msg.created_at.strftime("%Y-%m-%d %H:%M:%S") if msg.created_at else None
                })
            return {"success": True, "data": payload}
    except Exception as e:
        log.error(f"Failed to fetch daily messages for user '{user_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
