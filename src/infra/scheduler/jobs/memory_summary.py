# infra/scheduler/jobs/memory_summary.py — Daily memory consolidation and dream job
import os
import asyncio
from pathlib import Path
from datetime import datetime
from core.memory import mid_term
from core.memory import long_term
from core.roles.role_manager import role_manager
from models.sqlmodel.agent_daily_message import AgentDailyMessage
from infra.storage.mysql_connector import mysql_connector
from core.memory.mid_term import get_daily_log_path, append_to_daily_log
from utils.llm_utils import chat
from infra.logging.logger import logger as log

MEMORY_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "memory"

async def generate_and_save_daily_agent_messages(user_id: str, client=None, model: str = "") -> None:
    """
    Generates a personalized daily greeting/note for the user for each active persona agent,
    saves it to the database, and appends it to the daily log so it is consolidated in memory.
    """
    log.info(f"Generating daily agent messages for user: {user_id}")
    
    if not client or not model:
        from core.memory.summarizer import get_default_client_and_model
        d_client, d_model = get_default_client_and_model()
        client = client or d_client
        model = model or d_model

    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # 1. Read today's daily log (if any)
    log_path = get_daily_log_path(user_id, today_str)
    today_log_content = ""
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                today_log_content = f.read()
        except Exception as e:
            log.warning(f"Failed to read today's daily log: {e}")

    # 2. Scan roles to find persona roles
    roles = role_manager.list_roles()
    persona_roles = [r for r in roles if role_manager.is_persona_mode(r.get("id", ""))]

    for r in persona_roles:
        role_id = r.get("id")
        role_name = r.get("name", role_id)
        
        # Check if daily messages are enabled (defaults to True)
        if not r.get("daily_message_enabled", True):
            log.info(f"Daily message is disabled for agent '{role_name}' ({role_id}), skipping.")
            continue
            
        log.info(f"Generating daily message from agent '{role_name}' ({role_id}) to user '{user_id}'...")
        
        # Load personality files
        identity = role_manager.read_identity(role_id)
        soul = role_manager.read_soul(role_id)
        user_profile = role_manager.read_user(role_id)
        memory_context = role_manager.read_memory(role_id)
        
        # Compose LLM prompts
        system_prompt = (
            "你将扮演设定好的 AI 伴侣角色，现在深夜了，你需要根据自己的角色人设、性格、语气（见 IDENTITY / SOUL），\n"
            "结合你所知道的用户背景偏好（见 USER / MEMORY）以及今天用户发生的事情（见今日日志），为用户写一封日常聊天语气、温馨的晚安留言。\n\n"
            "【人设身份 IDENTITY】\n"
            f"{identity}\n\n"
            "【对话灵魂与语气 SOUL】\n"
            f"{soul}\n\n"
            "【用户喜好与强项 USER】\n"
            f"{user_profile}\n\n"
            "【长期记忆上下文 MEMORY】\n"
            f"{memory_context}\n"
        )
        
        user_prompt = (
            "今天用户的对话日志提要如下：\n"
            f"{today_log_content or '（今天用户没有与你聊天，不过你依然默默挂念着她）'}\n\n"
            "请按照你的人设，为用户写一封日常交谈口吻的晚安留言卡片：\n"
            "1. 针对用户的喜好和强项，帮她规划明天 1-2 件适合她且有价值的事情。\n"
            "2. 顺便聊聊 1 个今天看到的行业趋势、动漫或漫展相关的新鲜事。\n"
            "3. 结尾加上一句温暖的晚安话语，比如 '晚安，我先去休息啦，明天也要一起加油！'。\n"
            "4. 严格符合你的人设口吻（例如傲娇、可爱、古灵精怪等），字数控制在 150-250 字之间，多换行，排版美观。"
        )
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = await chat(client, messages, model=model, max_tokens=600)
            message_content = response.content or ""
            message_content = message_content.strip()
            
            if not message_content:
                log.warning(f"Generated empty message from agent '{role_id}', skipping save.")
                continue
                
            # Save to Database
            async with mysql_connector.session_scope() as session:
                from sqlmodel import select
                stmt = select(AgentDailyMessage).where(
                    AgentDailyMessage.user_id == user_id,
                    AgentDailyMessage.agent_id == role_id,
                    AgentDailyMessage.date_str == today_str
                )
                existing_res = await session.execute(stmt)
                existing = existing_res.scalar_one_or_none()
                
                if existing:
                    existing.content = message_content
                    session.add(existing)
                else:
                    new_msg = AgentDailyMessage(
                        user_id=user_id,
                        agent_id=role_id,
                        date_str=today_str,
                        content=message_content
                    )
                    session.add(new_msg)
                await session.commit()
            
            log.info(f"Daily message from '{role_id}' saved to database.")
            
            # Append to Daily Log for Dream/Memory consolidation
            append_to_daily_log(
                user_id=user_id,
                date_str=today_str,
                text=f"### Daily Agent Message to User ({role_name})\n{message_content}"
            )
            log.info(f"Daily message from '{role_id}' appended to today's log.")
            
        except Exception as e:
            log.error(f"Failed to generate daily message for agent '{role_id}': {e}", exc_info=True)

async def run_daily_memory_consolidation_and_dream() -> None:
    """
    Daily background memory consolidation job.
    1. Discovers all users under data/memory/
    2. For each user, discovers active session directories
    3. Runs session-to-daily-log consolidation (using session.md + jsonl)
    4. Generates and saves daily agent notes/greetings for the user
    5. Runs dream consolidation (updating MEMORY.md)
    """
    log.info("Starting daily memory consolidation and dream job...")
    
    if not MEMORY_ROOT.exists():
        log.warning(f"Memory root directory {MEMORY_ROOT} does not exist. Skipping job.")
        return
        
    # Discover users
    user_ids = []
    try:
        for p in MEMORY_ROOT.iterdir():
            if p.is_dir():
                user_ids.append(p.name)
    except Exception as e:
        log.error(f"Failed to scan memory root: {e}")
        return
        
    for user_id in user_ids:
        log.info(f"Processing memory for user: {user_id}")
        user_dir = MEMORY_ROOT / user_id
        
        # Discover sessions
        session_ids = []
        try:
            for p in user_dir.iterdir():
                if p.is_dir() and p.name != "logs":
                    session_ids.append(p.name)
        except Exception as e:
            log.warning(f"Failed to scan sessions for user {user_id}: {e}")
            continue
            
        # 1. Consolidate sessions (first part of memory lifecycle)
        for session_id in session_ids:
            log.info(f"Consolidating session {session_id} for user {user_id}...")
            try:
                # pass None client and model to use summarizer defaults
                success = await mid_term.consolidate_session_to_daily_log(
                    user_id=user_id,
                    session_id=session_id,
                    client=None,
                    model=""
                )
                if success:
                    log.info(f"Session {session_id} consolidated successfully.")
                else:
                    log.debug(f"Session {session_id} had no logs to consolidate.")
            except Exception as e:
                log.error(f"Failed to consolidate session {session_id}: {e}")
                
        # 2. Generate and save daily greetings/planning cards before dream
        try:
            await generate_and_save_daily_agent_messages(user_id=user_id, client=None, model="")
        except Exception as e:
            log.error(f"Failed to run daily agent message generation for user {user_id}: {e}", exc_info=True)

        # 3. Dream (second part of memory lifecycle)
        log.info(f"Running dream consolidation for user {user_id}...")
        try:
            success = await long_term.dream_consolidation(
                user_id=user_id,
                client=None,
                model=""
            )
            if success:
                log.info(f"Dream consolidation completed successfully for user {user_id}.")
            else:
                log.info(f"No recent logs to dream for user {user_id}.")
        except Exception as e:
            log.error(f"Dream consolidation failed for user {user_id}: {e}")
            
    log.info("Daily memory consolidation and dream job finished.")

def run_job_sync() -> None:
    """Wrapper to run the async job synchronously in the scheduler."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(run_daily_memory_consolidation_and_dream(), loop)
    else:
        # No running loop in this thread, standard asyncio.run is safe
        asyncio.run(run_daily_memory_consolidation_and_dream())
