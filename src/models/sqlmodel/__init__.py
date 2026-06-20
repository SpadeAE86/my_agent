"""
SQLModel ORM models for MySQL persistence.

Importing this module should register all tables into SQLModel.metadata.
"""

from .prompt_template import PromptTemplate  # noqa: F401
from .http_request_trace import HttpRequestTrace  # noqa: F401
from .image_history import ImageHistoryCard  # noqa: F401
from .video_analysis import (  # noqa: F401
    VideoAnalysisHistory,
    VideoAnalysisSceneFrames,
    VideoAnalysisSceneSplitFramesCache,
    VideoAnalysisShotCard,
    VideoAnalysisShotCardSharedFields,
    VideoAnalysisShotCardV2,
    VideoAnalysisTokenJoinTemplate,
    VideoAnalysisVideoV2,
)
from .video_upload_cache import VideoSourceUploadCache  # noqa: F401
from .video_material_match import VideoMaterialMatchHistory  # noqa: F401
from .video_match import VideoMatchJob, VideoMatchShotRow  # noqa: F401
from .video_mix_compose import VideoMixComposeJob  # noqa: F401
from .workspace import Workspace, WorkspaceNode, WorkspaceEdge  # noqa: F401
from .collections import ThemeSpace, CollectionItem, TagLibraryItem  # noqa: F401
from .ticket_buyer import TicketBuyer  # noqa: F401
from .scheduler_job import SchedulerJob, SchedulerJobLog  # noqa: F401
from .agent_daily_message import AgentDailyMessage  # noqa: F401
from .volco_timbre import VolcoTimbre  # noqa: F401



