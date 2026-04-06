from core.state import TaskItem, AgentState
import logging

class SubAgent:
    def __init__(self, task: TaskItem, global_context: str, allowed_tools: list):
        self.task = task
        self.global_context = global_context
        self.allowed_tools = allowed_tools
        self.history = []
        
    def run(self) -> str:
        """
        Executes a localized ReAct loop isolated to this specific task.
        Once complete, returns a concise summary of the outcome.
        """
        logging.info(f"[SubAgent] Starting task: {self.task.description}")
        
        # Inject isolated prompt
        system_prompt = f"""
        You are an executor agent. 
        Global Rules: {self.global_context}
        Your specific task: {self.task.description}
        Only use the tools provided to finish this task. Do not deviate.
        """
        
        # Simulated React loop:
        # result = llm_provider.chat(system_prompt, + allowed tools)
        
        logging.info(f"[SubAgent] Task complete.")
        return "SubAgent Result: Task completed successfully. Modified 2 files."
