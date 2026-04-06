from core.state import AgentState, Plan, TaskItem
from agent.sub_agent import SubAgent
import logging

class MainAgent:
    def __init__(self, state: AgentState):
        self.state = state

    def create_or_update_plan(self, user_request: str) -> Plan:
        """
        Calls the LLM to understand the global goal and split it into sub-tasks.
        """
        logging.info("[MainAgent] Analysing goal and creating plan...")
        # Simulated LLM output
        return Plan(
            goal=user_request,
            tasks=[
                TaskItem(id="1", description="Setup database schema", status="pending"),
                TaskItem(id="2", description="Write API endpoints", status="pending")
            ]
        )

    def spawn_subagent(self, task: TaskItem) -> SubAgent:
        """
        Spawns a subagent with a restricted, highly-focused context.
        """
        logging.info(f"[MainAgent] Spawning SubAgent for task: {task.id}")
        # Only inject the global rules and the specific sub-task description.
        # History of other sub-tasks is NOT passed, drastically reducing context length.
        return SubAgent(
            task=task,
            global_context=self.state.global_rules,
            allowed_tools=["edit_file", "bash"]
        )
