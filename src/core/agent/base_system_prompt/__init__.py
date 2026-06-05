from core.agent.base_system_prompt.identity import get_section_identity
from core.agent.base_system_prompt.doing_tasks import get_section_doing_tasks
from core.agent.base_system_prompt.actions import get_section_actions
from core.agent.base_system_prompt.using_tools import get_section_using_tools
from core.agent.base_system_prompt.tone_style import get_section_tone_style
from core.agent.base_system_prompt.efficiency import get_section_efficiency

__all__ = [
    "get_section_identity",
    "get_section_doing_tasks",
    "get_section_actions",
    "get_section_using_tools",
    "get_section_tone_style",
    "get_section_efficiency",
]
