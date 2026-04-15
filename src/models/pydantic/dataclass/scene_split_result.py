from pydantic import BaseModel
from pydantic.dataclasses import dataclass

@dataclass
class SceneSplitResult:
    scene_id: int
    frame_url_list: list
    start_time: float
    end_time: float
    duration_seconds: float