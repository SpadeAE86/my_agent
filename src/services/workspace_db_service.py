from __future__ import annotations

import uuid
from typing import List, Optional, Dict, Any

from sqlmodel import select

from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.workspace import Workspace, WorkspaceNode, WorkspaceEdge


class WorkspaceDBService:
    async def list_workspaces(self) -> List[Workspace]:
        async with mysql_connector.session_scope() as session:
            res = await session.execute(select(Workspace).order_by(Workspace.updated_at.desc()))
            return list(res.scalars().all())

    async def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        async with mysql_connector.session_scope() as session:
            res = await session.execute(select(Workspace).where(Workspace.id == workspace_id))
            return res.scalar_one_or_none()

    async def create_workspace(self, name: str) -> Workspace:
        async with mysql_connector.session_scope() as session:
            ws = Workspace(id=uuid.uuid4().hex[:12], name=name)
            session.add(ws)
            await session.commit()
            await session.refresh(ws)
            return ws

    async def delete_workspace(self, workspace_id: str) -> bool:
        async with mysql_connector.session_scope() as session:
            # 1. 查找 workspace
            res = await session.execute(select(Workspace).where(Workspace.id == workspace_id))
            ws = res.scalar_one_or_none()
            if ws is None:
                return False
            
            # 2. 删除关联的 nodes 和 edges
            nodes_res = await session.execute(select(WorkspaceNode).where(WorkspaceNode.workspace_id == workspace_id))
            for node in nodes_res.scalars().all():
                await session.delete(node)
                
            edges_res = await session.execute(select(WorkspaceEdge).where(WorkspaceEdge.workspace_id == workspace_id))
            for edge in edges_res.scalars().all():
                await session.delete(edge)
                
            await session.delete(ws)
            await session.commit()
            return True

    # ─── Nodes Operations ──────────────────────────────────────────
    async def list_nodes(self, workspace_id: str) -> List[WorkspaceNode]:
        async with mysql_connector.session_scope() as session:
            res = await session.execute(select(WorkspaceNode).where(WorkspaceNode.workspace_id == workspace_id))
            return list(res.scalars().all())

    async def upsert_node(
        self, workspace_id: str, node_id: str, node_type: str, x: float, y: float, data: Dict[str, Any]
    ) -> WorkspaceNode:
        async with mysql_connector.session_scope() as session:
            res = await session.execute(
                select(WorkspaceNode)
                .where(WorkspaceNode.workspace_id == workspace_id)
                .where(WorkspaceNode.id == node_id)
            )
            node = res.scalar_one_or_none()
            if node is None:
                node = WorkspaceNode(
                    id=node_id,
                    workspace_id=workspace_id,
                    type=node_type,
                    x=x,
                    y=y,
                    data=data,
                )
                session.add(node)
            else:
                node.type = node_type
                node.x = x
                node.y = y
                if data is not None:
                    node.data = data
            await session.commit()
            await session.refresh(node)
            return node

    async def delete_node(self, workspace_id: str, node_id: str) -> bool:
        async with mysql_connector.session_scope() as session:
            # 1. 找到并删除 node
            res = await session.execute(
                select(WorkspaceNode)
                .where(WorkspaceNode.workspace_id == workspace_id)
                .where(WorkspaceNode.id == node_id)
            )
            node = res.scalar_one_or_none()
            if node is None:
                return False
            await session.delete(node)

            # 2. 找到并删除与 node 关联的所有 edges (作为 source 或 target)
            edges_res = await session.execute(
                select(WorkspaceEdge)
                .where(WorkspaceEdge.workspace_id == workspace_id)
                .where((WorkspaceEdge.source_node_id == node_id) | (WorkspaceEdge.target_node_id == node_id))
            )
            for edge in edges_res.scalars().all():
                await session.delete(edge)

            await session.commit()
            return True

    # ─── Edges Operations ──────────────────────────────────────────
    async def list_edges(self, workspace_id: str) -> List[WorkspaceEdge]:
        async with mysql_connector.session_scope() as session:
            res = await session.execute(select(WorkspaceEdge).where(WorkspaceEdge.workspace_id == workspace_id))
            return list(res.scalars().all())

    async def upsert_edge(
        self, workspace_id: str, edge_id: str, source: str, target: str, label: Optional[str] = None
    ) -> WorkspaceEdge:
        async with mysql_connector.session_scope() as session:
            res = await session.execute(
                select(WorkspaceEdge)
                .where(WorkspaceEdge.workspace_id == workspace_id)
                .where(WorkspaceEdge.id == edge_id)
            )
            edge = res.scalar_one_or_none()
            if edge is None:
                edge = WorkspaceEdge(
                    id=edge_id,
                    workspace_id=workspace_id,
                    source_node_id=source,
                    target_node_id=target,
                    label=label,
                )
                session.add(edge)
            else:
                edge.source_node_id = source
                edge.target_node_id = target
                if label is not None:
                    edge.label = label
            await session.commit()
            await session.refresh(edge)
            return edge

    async def delete_edge(self, workspace_id: str, edge_id: str) -> bool:
        async with mysql_connector.session_scope() as session:
            res = await session.execute(
                select(WorkspaceEdge)
                .where(WorkspaceEdge.workspace_id == workspace_id)
                .where(WorkspaceEdge.id == edge_id)
            )
            edge = res.scalar_one_or_none()
            if edge is None:
                return False
            await session.delete(edge)
            await session.commit()
            return True


workspace_db_service = WorkspaceDBService()
