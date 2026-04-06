class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name: str, func):
        self.tools[name] = func

    def get_definitions(self, allowed_tools: list = None):
        # Return JSON schema definitions for the LLM
        return []

    def execute(self, name: str, kwargs: dict):
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")
        return self.tools[name](**kwargs)

# Global instances can be defined here
global_tools = ToolRegistry()
