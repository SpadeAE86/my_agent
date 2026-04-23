from __future__ import annotations

from typing import List, Optional

from sqlmodel import select

from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.prompt_template import PromptTemplate


class PromptTemplateDBService:
    async def list_names(self) -> List[str]:
        async with mysql_connector.session_scope() as session:
            res = await session.execute(select(PromptTemplate.name).order_by(PromptTemplate.name.asc()))
            return [r[0] for r in res.all()]

    async def get_by_name(self, name: str) -> Optional[PromptTemplate]:
        async with mysql_connector.session_scope() as session:
            res = await session.execute(select(PromptTemplate).where(PromptTemplate.name == name))
            return res.scalar_one_or_none()

    async def upsert(self, name: str, content: str) -> PromptTemplate:
        async with mysql_connector.session_scope() as session:
            res = await session.execute(select(PromptTemplate).where(PromptTemplate.name == name))
            obj = res.scalar_one_or_none()
            if obj is None:
                obj = PromptTemplate(name=name, content=content)
                session.add(obj)
            else:
                obj.content = content
            await session.commit()
            await session.refresh(obj)
            return obj

    async def delete(self, name: str) -> bool:
        async with mysql_connector.session_scope() as session:
            res = await session.execute(select(PromptTemplate).where(PromptTemplate.name == name))
            obj = res.scalar_one_or_none()
            if obj is None:
                return False
            await session.delete(obj)
            await session.commit()
            return True


prompt_template_db_service = PromptTemplateDBService()

