path = r"C:\Users\25065\Downloads\0.0%-zf0618-test-20260414-6258985-1.mp4"
from pymediainfo import MediaInfo

media_info = MediaInfo.parse(path)

for track in media_info.tracks:
    print(track.track_type, track.codec)