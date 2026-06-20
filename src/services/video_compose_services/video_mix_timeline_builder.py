from __future__ import annotations

from typing import Any, Dict, List, Optional

from models.pydantic.mix import (
    AudioConfig,
    Cap,
    CapConfig,
    CropConfig,
    MixedVideoRequest,
)
from models.sqlmodel.video_match import VideoMatchShotRow

VIDEO_TAIL_PAUSE_SEC = 0.4


def _best_video_path_from_hits(hits: Any) -> Optional[str]:
    if not isinstance(hits, list):
        return None
    for h in hits:
        if not isinstance(h, dict):
            continue
        for key in ("video_path", "video_url", "url", "obs_video_url"):
            u = str(h.get(key) or "").strip()
            if u:
                return u
    return None


def _scene_start_sec_for_selected_hit(row: VideoMatchShotRow, selected_url: Optional[str]) -> float:
    hits = row.match_top_hits_json
    if not isinstance(hits, list) or not hits:
        return 0.0
    
    target_hit = None
    if selected_url:
        sel_norm = selected_url.strip().lower()
        for h in hits:
            if not isinstance(h, dict):
                continue
            matched = False
            for key in ("video_path", "video_url", "url", "obs_video_url"):
                u = str(h.get(key) or "").strip().lower()
                if u == sel_norm:
                    matched = True
                    break
            if matched:
                target_hit = h
                break
                
    if not target_hit and hits:
        target_hit = hits[0]
        
    if not target_hit:
        return 0.0
        
    for key in ("start_time", "scene_start_sec", "scene_start"):
        v = target_hit.get(key)
        if v is not None:
            try:
                return max(0.0, float(v))
            except (TypeError, ValueError):
                pass
    src = target_hit.get("_source")
    if isinstance(src, dict):
        for key in ("start_time", "scene_start_sec"):
            v = src.get(key)
            if v is not None:
                try:
                    return max(0.0, float(v))
                except (TypeError, ValueError):
                    pass
    return 0.0


def _effective_top1_url(row: VideoMatchShotRow) -> Optional[str]:
    stored = str(row.top1_obs_url or "").strip() or None
    fallback = _best_video_path_from_hits(row.match_top_hits_json)
    return (stored or fallback or "").strip() or None


def collect_unique_source_obs_urls(shots: List[VideoMatchShotRow]) -> List[str]:
    ordered = sorted(shots, key=lambda s: s.shot_order)
    seen: set[str] = set()
    out: List[str] = []
    for row in ordered:
        u = _effective_top1_url(row)
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def format_srt_timestamp(seconds: float) -> str:
    """SRT 时间轴：HH:MM:SS,mmm"""
    ms_total = int(round(max(0.0, float(seconds)) * 1000))
    h, ms_rem = divmod(ms_total, 3600000)
    m, ms_rem = divmod(ms_rem, 60000)
    s, ms = divmod(ms_rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt_from_match_shots(shots: List[VideoMatchShotRow]) -> str:
    """
    与 ``build_mixed_video_request_from_shots`` 同一时间轴：每段口播对应一条字幕，
    起始时刻为累计视频时间轴上的 offset（与 audio_config.offset 一致）。
    """
    ordered = sorted(shots, key=lambda s: s.shot_order)
    lines: List[str] = []
    cursor = 0.0
    idx = 0
    for row in ordered:
        cap = (row.segment_text or "").strip()
        audio_dur = float(row.duration_sec or 0.0)
        if audio_dur <= 0:
            continue
        idx += 1
        start = cursor
        end = cursor + audio_dur
        lines.append(str(idx))
        lines.append(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}")
        safe = cap.replace("\r\n", "\n").replace("\r", "\n") if cap else " "
        lines.append(safe)
        lines.append("")
        vis_dur = audio_dur + VIDEO_TAIL_PAUSE_SEC
        cursor += vis_dur
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def build_mixed_video_request_from_shots(
    shots: List[VideoMatchShotRow],
    source_obs_to_high_res: Dict[str, str],
    *,
    biz_id: int,
    fps: int = 30,
    include_cap_config: bool = True,
) -> MixedVideoRequest:
    """
    按剪映式时间轴：每镜画面时长 = 口播时长 + ``VIDEO_TAIL_PAUSE_SEC``；音频在时间轴上早于画面结束 0.4s。
    视频轨使用 ``source_obs_to_high_res[原片 obs_url]`` 对应之高分辨率转码 URL。
    """
    if not (100 <= biz_id <= 999):
        raise ValueError("biz_id must be a three-digit integer in [100, 999]")

    ordered = sorted(shots, key=lambda s: s.shot_order)
    if not ordered:
        raise ValueError("no shots")

    obs_videos: List[str] = []
    obs_audios: List[str] = []
    crops: List[CropConfig] = []
    audio_cfgs: List[AudioConfig] = []
    captions: List[Cap] = []

    # 与 CapConfig 默认一致；混剪 Worker 会对每条 Cap 做 hex 校验，None 会触发 TypeError
    cap_theme = CapConfig()

    cursor = 0.0

    for row in ordered:
        top1 = _effective_top1_url(row)
        if not top1:
            raise ValueError(f"shot_order={row.shot_order} has no top1 video URL")

        high = (source_obs_to_high_res.get(top1) or "").strip() or top1
        audio_u = str(row.obs_audio_url or "").strip()
        if not audio_u:
            raise ValueError(f"shot_order={row.shot_order} missing obs_audio_url")

        audio_dur = float(row.duration_sec or 0.0)
        if audio_dur <= 0:
            raise ValueError(f"shot_order={row.shot_order} duration_sec must be positive")

        vis_dur = audio_dur + VIDEO_TAIL_PAUSE_SEC
        scene_start = _scene_start_sec_for_selected_hit(row, top1)
        crop_start = scene_start
        crop_end = scene_start + vis_dur

        obs_videos.append(high)
        obs_audios.append(audio_u)
        crops.append(CropConfig(start=crop_start, end=crop_end))
        audio_cfgs.append(AudioConfig(start=0.0, end=audio_dur, offset=cursor, volume=1.0))
        if include_cap_config:
            captions.append(
                Cap(
                    start=cursor,
                    end=cursor + audio_dur,
                    cap=row.segment_text or "",
                    font_size=cap_theme.font_size,
                    font_type=cap_theme.font_type,
                    color=cap_theme.cap_color,
                    outline_color=cap_theme.cap_outline_color,
                    outline_width=cap_theme.cap_outline_width,
                )
            )
        cursor += vis_dur

    cap_cfg: Optional[CapConfig] = None
    if include_cap_config:
        cap_cfg = CapConfig(
            caption_list=captions,
            font_size=cap_theme.font_size,
            font_type=cap_theme.font_type,
            cap_color=cap_theme.cap_color,
            cap_outline_color=cap_theme.cap_outline_color,
            cap_outline_width=cap_theme.cap_outline_width,
        )

    return MixedVideoRequest(
        biz_id=biz_id,
        obs_video_path_list=obs_videos,
        crop_config=crops,
        obs_audio_path_list=obs_audios,
        audio_config=audio_cfgs,
        cap_config=cap_cfg,
        fps=fps,
    )
