import os
import uuid
import tempfile
import shutil
import logging
from utils.obs_utils import upload_audio

log = logging.getLogger("clone_utils")

async def upload_original_audio(audio_bytes: bytes, audio_format: str, prefix: str = "ref") -> str:
    temp_dir = tempfile.mkdtemp(prefix=f"voice_{prefix}_")
    temp_file_path = os.path.join(temp_dir, f"audio.{audio_format}")
    try:
        with open(temp_file_path, "wb") as f:
            f.write(audio_bytes)
        project_id = f"original_{prefix}_{uuid.uuid4().hex[:12]}"
        url = await upload_audio(temp_file_path, project_id=project_id)
        return url
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
