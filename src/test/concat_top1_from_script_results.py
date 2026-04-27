"""
concat_top1_from_script_results.py

Read `script_tags_search_results.json`, take each segment's top1 `video_path`,
and concatenate them into a single MP4 for quick qualitative inspection.

Notes:
- Uses ffmpeg concat demuxer.
- If source clips differ in codecs/fps/resolution, we re-encode to H.264/AAC.

Run:
  python -m src.test.concat_top1_from_script_results
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List


RESULTS_JSON = Path(__file__).resolve().parent / "script_tags_search_results.json"
WORK_DIR = Path(__file__).resolve().parent / "workspace"
OUT_MP4 = Path(__file__).resolve().parent / "script_top1_concat.mp4"
FPS = 30


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_top1_paths(data: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for seg in (data.get("results") or []):
        if not isinstance(seg, dict):
            continue
        hits = seg.get("top_hits") or []
        if not isinstance(hits, list) or not hits:
            continue
        p = str((hits[0] or {}).get("video_path") or "").strip()
        if not p:
            continue
        out.append(p)
    return out


def _write_concat_list(paths: List[str], list_path: Path) -> None:
    # ffmpeg concat demuxer expects:
    # file 'C:\path\to\clip.mp4'
    lines = []
    for p in paths:
        # escape single quotes for ffmpeg list file
        pp = p.replace("'", r"'\''")
        lines.append(f"file '{pp}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _ffprobe_duration_seconds(path: str) -> float:
    # Returns container duration; may be unreliable for broken timestamps, but useful for logging.
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nk=1:nw=1",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="ignore").strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


def _normalize_clip(in_path: str, out_path: Path) -> None:
    """
    Normalize each segment to avoid concat timestamp/duration explosions:
    - reset PTS for audio/video to start at 0
    - enforce CFR (FPS) and yuv420p
    - re-encode to H.264/AAC
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-fflags",
        "+genpts",
        "-i",
        in_path,
        "-vf",
        f"fps={FPS},setpts=PTS-STARTPTS,format=yuv420p",
        "-af",
        "aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.check_call(cmd)


def main():
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(f"not found: {RESULTS_JSON}")

    data = _load_json(RESULTS_JSON)
    if not isinstance(data, dict):
        raise ValueError("results json must be an object")

    paths = _collect_top1_paths(data)
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise RuntimeError("no existing top1 video_path found to concat")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    normalized_dir = WORK_DIR / "normalized"
    normalized: List[str] = []
    for i, p in enumerate(paths):
        dur = _ffprobe_duration_seconds(p)
        print(f"[{i:02d}] src duration={dur:.3f}s path={p}")
        outp = normalized_dir / f"seg_{i:04d}.mp4"
        _normalize_clip(p, outp)
        ndur = _ffprobe_duration_seconds(str(outp))
        print(f"     normalized duration={ndur:.3f}s -> {outp}")
        normalized.append(str(outp))

    list_path = WORK_DIR / "concat_list_normalized.txt"
    _write_concat_list(normalized, list_path)

    # Re-encode to normalize.
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-fflags",
        "+genpts",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(OUT_MP4),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Wrote:", str(OUT_MP4))


if __name__ == "__main__":
    main()

