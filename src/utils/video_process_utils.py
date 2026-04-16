import os
from typing import List

import cv2
from scenedetect import detect, ContentDetector, open_video, SceneManager, VideoStreamCv2, VideoStream
from infra.logging.logger import logger as log
from models.pydantic.dataclass.scene_split_result import SceneSplitResult
from scenedetect.scene_manager import save_images

def save_scene_frames(frame, scene_id, frame_id, output_dir):
    # --- 保存为 WebP ---
    os.makedirs(output_dir, exist_ok=True)
    webp_filename = f"scene_{scene_id:03d}_frame_{frame_id:06d}.webp"
    webp_path = os.path.join(output_dir, webp_filename)
    # 使用 cv2 保存，质量参数设为 90 (默认75，100最高)
    # cv2.imread/imwrite 处理的是 BGR 格式，scenedetect 返回的也是 BGR，可以直接存
    import cv2
    cv2.imwrite(webp_path, frame, [cv2.IMWRITE_WEBP_QUALITY, 90])
    return webp_path

def get_video_scenes(video_path, frame_interval = 2, threshold=30.0, workspace_dir="./scene_detect_output") -> List[SceneSplitResult]:
    """
    检测视频场景并返回首尾帧及时间点

    :frame_interval: 帧间距
    :param video_path: 本地视频路径
    :param threshold: 灵敏度阈值 (默认30)
    :return: 包含场景信息的列表
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"未找到视频文件: {video_path}")

        # 创建输出目录
    os.makedirs(workspace_dir, exist_ok=True)

    # 1. 保持视频开启
    video: VideoStream = open_video(video_path, backend="opencv")
    fps = video.frame_rate
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    total_frames = video.duration.get_frames()
    total_time = video.duration.get_seconds()
    skip_frames = int(fps * frame_interval)  # 提前计算，避免每次循环都算

    # 2. 检测
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        log.info(f"未检测到场景切换，将整个视频作为一个场景处理")
        from scenedetect.frame_timecode  import FrameTimecode
        # 创建覆盖整个视频的虚拟场景
        start_time = FrameTimecode(fps=fps, timecode="00:00:00")
        end_time = FrameTimecode(fps=fps, timecode=total_frames)
        scene_list = [(start_time, end_time)]

    results = []
    # 遍历每个场景，独立计算抽帧数
    _, end_time = scene_list[-1]
    video = open_video(video_path, backend="opencv")
    total_frames = video.duration.get_frames()

    for i, scene in enumerate(scene_list):
        start_time, end_time = scene
        start_s = start_time.get_seconds()
        end_s = end_time.get_seconds()
        duration = end_s - start_s

        scene_no = i + 1

        start_f = start_time.get_frames()
        end_f = end_time.get_frames()
        target_f = start_f
        scene_frame_list = []

        # 安全守卫：如果视频指针已经超过了当前场景，直接跳过该场景
        if video.frame_number >= end_f:
            results.append(SceneSplitResult(
                scene_id=scene_no,
                frame_url_list=[],
                start_time=start_s,
                end_time=end_s,
                duration_seconds=duration
            ))
            continue

        #开始双指针，推进读取的帧位置，直到即将超过当前场景的结束帧
        while video.frame_number < end_f and video.frame_number < total_frames:
            # 将秒数转换为帧序号
            tmp_frame = video.frame_number
            is_target = (video.frame_number == target_f)
            frame = video.read(decode=is_target)

            if is_target:
                if frame is not None:
                    frame_path = save_scene_frames(frame, scene_no, tmp_frame, workspace_dir)
                    scene_frame_list.append(frame_path)
                target_f = min(end_f - 1, target_f + skip_frames)

        scene_result = SceneSplitResult(
            scene_id=scene_no,
            frame_url_list = scene_frame_list,
            start_time=start_s,
            end_time=end_s,
            duration_seconds=duration
        )
        results.append(scene_result)

    return results


# --- 使用示例 ---
if __name__ == "__main__":
    from pathlib import Path
    from yarl import URL
    file_path_list = [
        r"C:\Users\25065\Downloads\汽车\drive.mp4",
        r"C:\Users\25065\Downloads\汽车\inner.mp4",
        r"C:\Users\25065\Downloads\汽车\back.mp4",
        r"C:\Users\25065\Downloads\汽车\wheel.mp4",
        r"C:\Users\25065\Downloads\汽车\light.mp4",
        r"C:\Users\25065\Downloads\汽车\front.mp4"
    ]
    for video in file_path_list:

        file_path = Path(video)
        file_basename = file_path.stem

        # file_path = r"C:\Users\25065\Downloads\soft_H.264_1280x720_AAC_700.mp4"
        try:
            print(f"开始分析{video}...")
            scenes = get_video_scenes(str(file_path), workspace_dir=f"./car_detect_output/{file_basename}")

            log.info(f"检测到 {len(scenes)} 个场景：\n")

            for s in scenes:
                log.info(f"url list: {s.frame_url_list[0]} | start: {s.start_time} | end: {s.end_time} | duration: {s.duration_seconds}")

        except Exception as e:
            print(f"发生错误: {e}")