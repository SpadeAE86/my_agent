import os
import sys
import time
import random
import asyncio
import subprocess
import requests
from pathlib import Path

# Add project src directory to sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src_dir)

# Load env variables from src/.env
from dotenv import load_dotenv
load_dotenv(os.path.join(src_dir, ".env"))

# Configuration Options
LIMIT_TEST = False # Set to True to only process 11 voices for the concurrency test. Set to False to process all.
REGENERATE = False # Set to True to regenerate voices that already have a sample URL in the database.
TEXT_TO_SPEAK = "你好，很高兴认识你！我将成为你的AI伙伴，一起加油吧！"

async def check_tos_connectivity():
    from utils.tos_utils import get_tos_client, BUCKET_NAME
    import tos
    print(f"Checking TOS connectivity to bucket '{BUCKET_NAME}'...")
    try:
        client = get_tos_client()
        test_key = "audio/timbre_example/connectivity_test.txt"
        content = b"TOS Connectivity Test Successful"
        await asyncio.to_thread(
            client.put_object,
            bucket=BUCKET_NAME,
            key=test_key,
            content=content
        )
        print("TOS connectivity check PASSED! Test file uploaded successfully.")
        # Clean up the test file
        await asyncio.to_thread(
            client.delete_object,
            bucket=BUCKET_NAME,
            key=test_key
        )
        print("TOS test file cleaned up successfully.")
        return True
    except Exception as e:
        print(f"TOS connectivity check FAILED: {e}")
        return False

async def process_single_voice(row, text, regenerate):
    from services.tts_services import tts_manager
    from utils.tos_utils import upload_to_tos
    from models.sqlmodel.voice_timbre import VoiceTimbre
    from infra.storage.mysql_connector import mysql_connector
    from sqlmodel import select

    voice_character = row.voice_character
    provider = row.provider
    voice_id = row.id

    # Skip if already processed and regenerate is False
    if not regenerate and row.full_voice:
        print(f"  [SKIP] {voice_character} (full_voice already exists in DB: {row.full_voice})")
        return row.full_voice

    platform_name = provider
    model_name = "qwen3_online" if platform_name == "qwen" else "volcano_v3"

    # Determine the text to speak based on the voice language (Volcano English voices fail with Chinese text)
    text_to_use = text
    code_lower = row.voice_code.lower()
    if code_lower.startswith("en_") or "en_female" in code_lower or "en_male" in code_lower:
        text_to_use = "Hello, nice to meet you. I will be your partner. Let's do our best together!"
        print(f"  [INFO] {voice_character} is an English voice. Using English text.")

    print(f"  [START] {voice_character} ({provider}) -> synthesizing...")
    t_start = time.time()

    temp_mp3 = Path(f"./temp_{voice_character}_{voice_id}.mp3")
    temp_wav = Path(f"./{voice_character}.wav")

    try:
        # Synthesize voice
        engine_name = "qwen3_online" if provider == "qwen" else "volcano_v3"
        url = await tts_manager.generate_voice(
            text=text_to_use,
            voice_character=voice_character,
            engine_name=engine_name,
            instruct="用温暖亲切的口吻说话"
        )
        
        # Download the audio file
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(temp_mp3, "wb") as f:
            f.write(resp.content)

        # Transcode to WAV format
        cmd = [
            "ffmpeg", "-y", "-i", str(temp_mp3), str(temp_wav)
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Upload to TOS
        tos_prefix = f"audio/timbre_example/{platform_name}/{model_name}/"
        tos_url = await upload_to_tos(str(temp_wav), tos_prefix=tos_prefix)

        # Update database record
        async with mysql_connector.session_scope() as session:
            stmt = select(VoiceTimbre).where(VoiceTimbre.id == voice_id)
            res = await session.execute(stmt)
            db_row = res.scalar_one_or_none()
            if db_row:
                db_row.full_voice = tos_url
                session.add(db_row)
                await session.commit()

        duration = time.time() - t_start
        print(f"  [SUCCESS] {voice_character} ({provider}) -> Done in {duration:.2f}s! TOS URL: {tos_url}")
        return tos_url

    except Exception as e:
        print(f"  [FAILED] {voice_character} ({provider}) -> Error: {e}")
        raise e
    finally:
        # Clean up temp files
        try:
            if temp_mp3.exists():
                temp_mp3.unlink()
        except:
            pass
        try:
            if temp_wav.exists():
                temp_wav.unlink()
        except:
            pass

async def main():
    from infra.connector_loader import connector_loader
    from models.sqlmodel.voice_timbre import VoiceTimbre
    from infra.storage.mysql_connector import mysql_connector
    from sqlmodel import select

    print("Initializing connectors...")
    await connector_loader.startup()

    try:
        print("Checking TOS connectivity...")
        tos_ok = await check_tos_connectivity()
        if not tos_ok:
            print("TOS connectivity check failed. Exiting.")
            sys.exit(1)

        print("\nQuerying eligible online voices (only those supporting instruct)...")
        async with mysql_connector.session_scope() as session:
            stmt = select(VoiceTimbre).where(VoiceTimbre.is_enabled == 1)
            result = await session.execute(stmt)
            rows = result.scalars().all()

        eligible_voices = []
        for row in rows:
            actual_model_type = row.voice_model_type
            if not actual_model_type or actual_model_type == "default":
                if row.provider == "volcano":
                    actual_model_type = "small" if row.voice_character == "天才少女" else "big"
                else:
                    actual_model_type = "big"

            # Check instruct capability matching the models filter logic
            has_instruct = False
            if row.provider == "qwen":
                has_instruct = True
            elif row.provider == "volcano" and actual_model_type == "big":
                code_lower = row.voice_code.lower()
                if "uranus" in code_lower or "saturn" in code_lower:
                    has_instruct = True

            if has_instruct:
                eligible_voices.append(row)

        print(f"Total eligible voices found in DB: {len(eligible_voices)}")

        # For testing, mix Volcano and Qwen voices
        volcano_voices = [v for v in eligible_voices if v.provider == "volcano"]
        qwen_voices = [v for v in eligible_voices if v.provider == "qwen"]

        if LIMIT_TEST:
            test_voices = volcano_voices[:5] + qwen_voices[:6]
            print(f"LIMIT_TEST is True. Selected {len(test_voices)} voices (5 Volcano + 6 Qwen) for testing batch concurrency of 8.")
            eligible_voices = test_voices

        # Process in batches of 8
        batch_size = 8
        t_batch_start = time.time()
        for i in range(0, len(eligible_voices), batch_size):
            batch = eligible_voices[i:i+batch_size]
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Batch {i//batch_size + 1} with {len(batch)} voices:")
            for voice in batch:
                print(f"  - {voice.voice_character} ({voice.provider})")

            tasks = [process_single_voice(voice, TEXT_TO_SPEAK, REGENERATE) for voice in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Batch {i//batch_size + 1} complete. Status summary:")
            for voice, res in zip(batch, results):
                if isinstance(res, Exception):
                    print(f"  - {voice.voice_character}: [FAILED] {res}")
                else:
                    print(f"  - {voice.voice_character}: [SUCCESS] {res}")

        total_duration = time.time() - t_batch_start
        print(f"\nAll batches finished in {total_duration:.2f} seconds.")

    finally:
        print("Shutting down connectors...")
        await connector_loader.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
