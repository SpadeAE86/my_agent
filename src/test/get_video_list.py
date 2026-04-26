import os

dir = r"C:\Users\admin\Downloads\LS6视频"

def get_video_list(dir):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv']
    video_files = []

    for root, dirs, files in os.walk(dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))

    return video_files

if __name__ == "__main__":
    videos = get_video_list(dir)
    for video in videos:
        print(video)