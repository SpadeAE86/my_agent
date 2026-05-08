import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from services.image_generate_service import generate_image

async def test_gpt_image_generation():
    prompt = "A beautiful cyberpunk city at night with neon lights"
    model = "gpt-image-2"
    size = "1024x1024"
    
    print(f"Testing image generation with model: {model}")
    try:
        url = await generate_image(
            prompt=prompt,
            model=model,
            size=size
        )
        if url:
            print(f"Successfully generated image! URL: {url}")
        else:
            print("Failed to generate image (returned None)")
    except Exception as e:
        print(f"Error during image generation: {e}")

if __name__ == "__main__":
    asyncio.run(test_gpt_image_generation())
