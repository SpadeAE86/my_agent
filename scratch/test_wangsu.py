import sys
import os
import requests
import time
from io import BytesIO

_IMGEDIT_URL = "https://aigateway.edgecloudapp.com/v1/7ab0a462e54c5b92a93c285e43dadcb6/gpt-imge-2-image-edit"
_IMGEDIT_KEY = "bd57778f1c8a4a8087dac525a56e5cd3"  # From .env
_NO_PROXY = {"http": None, "https": None}

def main():
    img_url = "https://freeuuu.obs.cn-east-3.myhuaweicloud.com/ai_picture/reference_image/1780912575508_l61ahbc81i.jpg"
    prompt = "generate a Japanese animation girl, who has beautiful brown hair and brown eyes like Yuki Asuna, add some Christmas element to her such as present or magic"
    size = "1440x2560"
    
    print("Downloading reference image...")
    r_img = requests.get(img_url, timeout=15, proxies=_NO_PROXY)
    print("Download completed.")
    
    files = [("image", ("reference.jpg", BytesIO(r_img.content), "image/jpeg"))]
    data = {
        "prompt": prompt,
        "aspect_ratio": "9:16",
        "quality": "medium",
        "size": size,
    }
    
    print(f"Sending POST request to {_IMGEDIT_URL}...")
    print(f"Payload: {data}")
    start = time.monotonic()
    try:
        # We will use a 30-second timeout here
        resp = requests.post(
            _IMGEDIT_URL,
            data=data,
            files=files,
            headers={"Authorization": f"Bearer {_IMGEDIT_KEY}"},
            timeout=30.0,
            proxies=_NO_PROXY,
        )
        print(f"Response received in {time.monotonic() - start:.2f}s!")
        print(f"Status Code: {resp.status_code}")
        print(f"Response Body: {resp.text[:500]}")
    except Exception as e:
        print(f"Request failed in {time.monotonic() - start:.2f}s with error: {e}")

if __name__ == "__main__":
    main()
