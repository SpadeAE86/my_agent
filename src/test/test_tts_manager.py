import os
import sys
import asyncio

# Add src to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def main():
    from services.tts_services import tts_manager
    
    print("Initial default TTS engine:", tts_manager.default_engine_name)
    
    # 1. Test Volcano Engine (if default or set to volcano)
    print("\n--- Testing Volcano Engine ---")
    try:
        url_volcano = await tts_manager.generate_voice(
            text="你好，这是通过统一管理器调用的火山引擎声音。",
            voice_character="Vivi",
            engine_name="volcano"
        )
        print("Volcano generation success! URL:", url_volcano)
    except Exception as e:
        print("Volcano generation failed:", e)
        
    # 2. Test Qwen3 Custom Voice (Preset)
    print("\n--- Testing Qwen3 Custom Voice ---")
    try:
        url_qwen_custom = await tts_manager.generate_voice(
            text="你好，这是通过统一管理器调用的昆仑三预设音色。",
            voice_character="ryan",
            engine_name="qwen3"
        )
        print("Qwen3 Custom generation success! URL:", url_qwen_custom)
    except Exception as e:
        print("Qwen3 Custom generation failed:", e)

    # 3. Test Qwen3 Voice Design (Instruct)
    print("\n--- Testing Qwen3 Voice Design ---")
    try:
        url_qwen_design = await tts_manager.generate_voice(
            text="你好，这是通过文字描述由昆仑三引擎直接凭空设计生成的全新音色。",
            voice_character="custom_designed",
            engine_name="qwen3",
            instruct="一个声音非常甜美、说话速度很快的年轻女孩"
        )
        print("Qwen3 Design generation success! URL:", url_qwen_design)
    except Exception as e:
        print("Qwen3 Design generation failed:", e)

    # 4. Test Qwen3 Voice Clone (Reference Audio)
    print("\n--- Testing Qwen3 Voice Clone ---")
    ref_audio = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_tts\tts_1780262424778.wav"
    try:
        url_qwen_clone = await tts_manager.generate_voice(
            text="你好，这是通过提供参考音频文件由昆仑三进行克隆生成的克隆音色。",
            voice_character="cloned_speaker",
            engine_name="qwen3",
            ref_audio=ref_audio,
            x_vector_only_mode=True
        )
        print("Qwen3 Clone generation success! URL:", url_qwen_clone)
    except Exception as e:
        print("Qwen3 Clone generation failed:", e)

if __name__ == "__main__":
    asyncio.run(main())
