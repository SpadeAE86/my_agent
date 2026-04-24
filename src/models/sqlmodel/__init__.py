"""
SQLModel ORM models for MySQL persistence.

Importing this module should register all tables into SQLModel.metadata.
"""

from .prompt_template import PromptTemplate  # noqa: F401
from .image_history import ImageHistoryCard  # noqa: F401
from .video_analysis import VideoAnalysisHistory, VideoAnalysisShotCard  # noqa: F401

