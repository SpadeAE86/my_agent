import os
from scenedetect import detect, ContentDetector, split_video_ffmpeg, open_video, SceneManager

from models.pydantic.dataclass.scene_split_result import SceneSplitResult
from scenedetect.scene_manager import save_images

def get_video_scenes(video_path, threshold=30.0, workspace_dir="./scene_detect_output"):
    """
    检测视频场景并返回首尾帧及时间点

    :param video_path: 本地视频路径
    :param threshold: 灵敏度阈值 (默认30)
    :return: 包含场景信息的列表
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"未找到视频文件: {video_path}")

    # 1. 保持视频开启
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30))

    # 2. 检测
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    # 3. 这里的 video 对象依然是活跃的，直接传给 save_images
    save_images(
        scene_list=scene_list,
        video=video,
        num_images=1,
        output_dir=workspace_dir,
        image_extension="webp"
    )
    results = []
    for i, scene in enumerate(scene_list):
        start_timecode, end_timecode = scene
        scene_no = i + 1  # SceneDetect 的编号通常从 1 开始

        # 2. 构建刚才保存的图片路径
        # 根据 save_images 的默认行为，第一张图会追加 -01
        image_filename = f"scene_{scene_no:03d}-01.webp"  # 03d 是为了匹配 SceneDetect 默认的补零规则
        local_path = os.path.join(workspace_dir, image_filename)

        scene_data = SceneSplitResult(
            scene_id=scene_no,
            frame_url_list=[local_path],  # 先存本地路径，后续上传后替换为 URL
            start_time=start_timecode.get_seconds(),
            end_time=end_timecode.get_seconds(),
            duration_seconds=end_timecode.get_seconds() - start_timecode.get_seconds()
        )
        results.append(scene_data)

    return results


# --- 使用示例 ---
if __name__ == "__main__":
    file_path = "test.mp4"
    try:
        scenes = get_video_scenes(file_path)

        print(f"检测到 {len(scenes)} 个场景：\n")
        print(f"{'场景':<5} | {'起始帧':<8} | {'结束帧':<8} | {'开始时间':<12} | {'结束时间'}")
        print("-" * 60)

        for s in scenes:
            print(f"{s['scene_id']:<5} | {s['start_frame']:<10} | {s['end_frame']:<10} | "
                  f"{s['start_time']:<12} | {s['end_time']}")

    except Exception as e:
        print(f"发生错误: {e}")