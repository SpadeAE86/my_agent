import logging
from core.state import AgentState
from agent.main_agent import MainAgent

logging.basicConfig(level=logging.INFO)

class HierarchicalLoop:
    def __init__(self):
        self.state = AgentState(session_id="new-session")
        self.main_agent = MainAgent(self.state)

    def start(self, user_request: str):
        print(f"--- Starting Project: {user_request} ---")
        
        # 1. Main Agent Plans
        self.state.plan = self.main_agent.create_or_update_plan(user_request)
        
        # 2. Iteratively execute the Plan
        for task in self.state.plan.tasks:
            if task.status == "pending":
                task.status = "in_progress"
                
                # 3. Spawn SubAgent (Isolated memory, isolated tools)
                sub_agent = self.main_agent.spawn_subagent(task)
                
                # 4. Await SubAgent execution
                task.result = sub_agent.run()
                task.status = "completed"
                
                print(f"Task {task.id} done: {task.result}")

        print("--- All Tasks Completed! ---")

if __name__ == "__main__":
    loop = HierarchicalLoop()
    loop.start("Build a fast CRUD API for users")
