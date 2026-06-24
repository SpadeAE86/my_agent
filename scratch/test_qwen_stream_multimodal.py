import os
import sys
import json
import asyncio
from dotenv import load_dotenv

# Load env variables from src/.env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", ".env")
load_dotenv(env_path)

api_key = os.getenv("DASHSCOPE_API_KEY")
print(f"Using API Key: {api_key[:10]}...{api_key[-10:] if api_key else 'None'}")

async def try_streaming_call():
    import dashscope
    dashscope.api_key = api_key
    
    # Use our previously designed voice ID
    voice_id = "qwen-tts-vd-htmulti793-voice-20260621044630419-d9a5"
    model_name = "qwen3-tts-vd-2026-01-26"
    text = "你想要的是一个有脑子、有灵气、有审美、会主动生长出想法的搭档。"
    
    print(f"\n--- Trying MultiModalConversation.call with stream=True, model={model_name} ---")
    
    try:
        t0 = asyncio.get_event_loop().time()
        responses = await asyncio.to_thread(
            dashscope.MultiModalConversation.call,
            model=model_name,
            api_key=api_key,
            text=text,
            voice=voice_id,
            stream=True
        )
        
        chunk_count = 0
        audio_bytes_accumulated = 0
        
        # Iterate over the response stream
        # Wait, since MultiModalConversation.call is a synchronous generator when stream=True,
        # we iterate over it in an executor thread or standard loop
        def iterate_stream():
            nonlocal chunk_count, audio_bytes_accumulated
            for response in responses:
                chunk_count += 1
                status = response.status_code if hasattr(response, 'status_code') else 'Unknown'
                
                # Check response output structure
                output = response.get("output", {})
                audio_info = output.get("audio", {})
                audio_data = audio_info.get("data", "")
                audio_url = audio_info.get("url", "")
                
                if chunk_count <= 3 or audio_url:
                    print(f"Chunk #{chunk_count} - Status: {status}")
                    print(f"  Audio ID: {audio_info.get('id')}")
                    print(f"  Audio Data Length: {len(audio_data) if audio_data else 0}")
                    print(f"  Audio URL: {audio_url}")
                
                if audio_data:
                    import base64
                    raw_chunk = base64.b64decode(audio_data)
                    audio_bytes_accumulated += len(raw_chunk)
                    
            print(f"\nStream Finished. Total chunks: {chunk_count}")
            print(f"Total audio bytes accumulated: {audio_bytes_accumulated}")
            
        t_start = asyncio.get_event_loop().time()
        await asyncio.to_thread(iterate_stream)
        t_end = asyncio.get_event_loop().time()
        print(f"Total time taken for streaming loop: {t_end - t_start:.2f}s")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Streaming failed: {e}")

if __name__ == "__main__":
    asyncio.run(try_streaming_call())
