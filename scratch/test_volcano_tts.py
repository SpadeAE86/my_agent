import os
import sys
import asyncio
from dotenv import load_dotenv

# Add src to python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_path)

# Load env file
load_dotenv(os.path.join(src_path, ".env"))

from services.volcovoice_service import VolcoVoiceService

async def main():
    text = "你好，这是一个测试音频！"
    voice = "Vivi"
    print("Testing Volcano TTS generation...")
    print("VOLCANO_APP_ID:", os.getenv("VOLCANO_APP_ID"))
    print("VOLCANO_ACCESS_TOKEN:", os.getenv("VOLCANO_ACCESS_TOKEN")[:6] + "..." if os.getenv("VOLCANO_ACCESS_TOKEN") else None)
    
    try:
        url = await VolcoVoiceService.generate_voice(
            text=text,
            voice_character=voice,
            disable_segmentation=True
        )
        print("\n[SUCCESS] Audio generated and uploaded successfully!")
        print("Audio URL:", url)
    except Exception as e:
        print("\n[FAILED] Generation failed:", e)

if __name__ == "__main__":
    asyncio.run(main())
