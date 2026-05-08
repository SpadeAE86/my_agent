import os
from typing import Optional

from openai import AsyncOpenAI
import asyncio

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """你是一个人工智能助手，协助用户解答问题和提供信息。请根据用户的提问，尽可能准确和详细地回答。如果你不确定答案，可以说你不知道，但不要编造信息。"""
base = 1024

scale_map = {
    "1K": 1,
    "2K": 2,
}

_GPT_IMG_BASE = os.getenv("GPT_IMAGE_OPENAI_BASE_URL", "https://ai.comfly.chat/v1")
_GPT_IMG_KEY = os.getenv(
    "GPT_IMAGE_OPENAI_API_KEY",
    "sk-EZyThGS2JdkoxISCD7Dd64D625E94a8b9513D71aCfF6AcFc",
)

client = AsyncOpenAI(base_url=_GPT_IMG_BASE, api_key=_GPT_IMG_KEY, timeout=500)

def get_resolution(k_level: str, aspect_ratio: str = "1:1"):
    if k_level not in scale_map:
        raise ValueError(f"invalid resolution level: {k_level}")

    scale = scale_map[k_level]
    w, h = map(int, aspect_ratio.split(":"))

    # 归一化比例（保持比例）
    max_side = base * scale

    if w == h:
        return f"{max_side}x{max_side}"

    if w > h:
        width = max_side
        height = int(max_side * h / w)
    else:
        height = max_side
        width = int(max_side * w / h)

    return f"{width}x{height}"


async def call_gpt_image_2(prompt,
                           model="gpt-image-2",
                           size = "1440x2560",
                           reference_image_list: Optional[list[str]] = None
                           ):
    """
    Sends a chat message to the OpenAI API and returns the response message object.

    Args:
        model (str): The model to use for the chat.
        size (str): The size of the generated image, e.g., "1536x1024".
        reference_image_list (Optional[list[str]]): A list of reference image URLs.
    Returns:
        The raw response message object (with .content and .tool_calls).
    """

    extra_body = {
        "quality": "medium"
    }
    if reference_image_list:
        extra_body["image"] = reference_image_list

    images_response = await client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        response_format="url",
        extra_body=extra_body
    )

    image_url = images_response.data[0].url
    return image_url

if __name__ == "__main__":
    prompt = "generate a Japanese animation girl, who has beautiful pink hair and blue eyes, poster for me, add some Halloween element to her such as candy or magical style"
    result = asyncio.run(call_gpt_image_2(prompt, model="gpt-image-2"))
    print(f"receive: {result[:100]}")
