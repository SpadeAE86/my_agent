from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class TaskItem(BaseModel):
    id: str
    description: str
    status: str  # "pending", "in_progress", "completed", "failed"
    result: Optional[str] = None

class Plan(BaseModel):
    goal: str
    tasks: List[TaskItem]

class AgentState(BaseModel):
    session_id: str
    plan: Optional[Plan] = None
    global_rules: str = "Always write clean, standard Python code."
    history: List[Dict[str, Any]] = []
