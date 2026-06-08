from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from services.workspace_db_service import workspace_db_service
from models.sqlmodel.workspace import Workspace, WorkspaceNode, WorkspaceEdge

workspace_router = APIRouter(prefix="/workspace", tags=["workspace"])


# ─── Pydantic Request Models ──────────────────────────────────────
class WorkspaceCreateRequest(BaseModel):
    name: str


class NodeUpsertRequest(BaseModel):
    id: str
    type: str
    x: float
    y: float
    data: Optional[Dict[str, Any]] = None


class EdgeUpsertRequest(BaseModel):
    id: str
    source: str
    target: str
    label: Optional[str] = None


# ─── Workspace Router Endpoints ───────────────────────────────────
@workspace_router.get("")
async def list_workspaces():
    """
    列出所有画布工程
    """
    items = await workspace_db_service.list_workspaces()
    return {"success": True, "workspaces": items}


@workspace_router.post("")
async def create_workspace(req: WorkspaceCreateRequest):
    """
    创建一个新画布工程
    """
    ws = await workspace_db_service.create_workspace(name=req.name)
    return {"success": True, "workspace": ws}


@workspace_router.delete("/{workspace_id}")
async def delete_workspace(workspace_id: str):
    """
    删除工程及其下所有关联节点和连线
    """
    success = await workspace_db_service.delete_workspace(workspace_id)
    if not success:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"success": True}


@workspace_router.get("/{workspace_id}/graph")
async def get_workspace_graph(workspace_id: str):
    """
    获取指定工程下的所有节点和连线，用于前端画布还原
    """
    ws = await workspace_db_service.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    nodes = await workspace_db_service.list_nodes(workspace_id)
    edges = await workspace_db_service.list_edges(workspace_id)
    
    return {
        "success": True,
        "workspace": ws,
        "nodes": nodes,
        "edges": edges
    }


# ─── Nodes Endpoints ──────────────────────────────────────────────
@workspace_router.post("/{workspace_id}/nodes")
async def upsert_workspace_node(workspace_id: str, req: NodeUpsertRequest):
    """
    创建或更新单个节点（如拖动更新位置坐标、更新图片生成状态等）
    """
    ws = await workspace_db_service.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    node = await workspace_db_service.upsert_node(
        workspace_id=workspace_id,
        node_id=req.id,
        node_type=req.type,
        x=req.x,
        y=req.y,
        data=req.data
    )
    return {"success": True, "node": node}


@workspace_router.delete("/{workspace_id}/nodes/{node_id}")
async def delete_workspace_node(workspace_id: str, node_id: str):
    """
    删除指定节点及其相连的连线
    """
    success = await workspace_db_service.delete_node(workspace_id, node_id)
    if not success:
        raise HTTPException(status_code=404, detail="Node not found")
    return {"success": True}


# ─── Edges Endpoints ──────────────────────────────────────────────
@workspace_router.post("/{workspace_id}/edges")
async def upsert_workspace_edge(workspace_id: str, req: EdgeUpsertRequest):
    """
    创建或更新单个连线（定义节点演进关系）
    """
    ws = await workspace_db_service.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    edge = await workspace_db_service.upsert_edge(
        workspace_id=workspace_id,
        edge_id=req.id,
        source=req.source,
        target=req.target,
        label=req.label
    )
    return {"success": True, "edge": edge}


@workspace_router.delete("/{workspace_id}/edges/{edge_id}")
async def delete_workspace_edge(workspace_id: str, edge_id: str):
    """
    删除特定连线
    """
    success = await workspace_db_service.delete_edge(workspace_id, edge_id)
    if not success:
        raise HTTPException(status_code=404, detail="Edge not found")
    return {"success": True}
