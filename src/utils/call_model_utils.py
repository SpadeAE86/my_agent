import os
from openai import OpenAI
from typing import Optional

from models.pydantic.model_output_schema.video_analysis_schema import SceneAnalysisResult
from models.pydantic.request import SEEDREAM_MODEL_MAP, SEEDTEXT_MODEL_MAP


ARK_API_KEY = "a3818169-d25e-49fd-8bf8-dea20197475c"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

    # 🌟 核心魔法：直接用 Pydantic 对象生成标准 JSON Schema

def call_doubao_vision(prompt, image_url_list, schema_json = None):
    if not ARK_API_KEY:
        print("错误：未在环境变量 FZ_API_KEY 中找到 API Key。")
        return

    client = OpenAI(
        base_url=BASE_URL,
        api_key=ARK_API_KEY,
    )

    prompt_text = """
    请分析这些视频片段里的画面，描述视频的内容，提取主体以及他对应的动作，以及内容的特点，封装成符合格式要求的json。
    """ if not prompt else prompt

    content_list = [
        {
            "type": "text",
            "text": prompt_text
        },
    ]

    for image_url in image_url_list:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })

    messages = [{"role": "user", "content": content_list}]
    response_format = None
    if schema_json:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "SceneAnalysisResult",  # 👈 Schema 名称必填
                "schema": schema_json
            }
        }
    try:
        print(f"message: {messages}, format: {response_format}")
        print("正在调用豆包 API 分析图片...")
        # 尝试使用 doubao_vision.py 中已有的模型名称，或者是常见的模型名
        # 因为在 doubao_vision.py 中写的是 "doubao-seed-1-6-vision-250815"
        # 我们用它去测试
        response = client.chat.completions.create(
            model="doubao-seed-1-6-vision-250815",
            messages=messages,
            response_format=response_format
        )

        result = response.choices[0].message.content
        print("====== 豆包 API 返回结果 ======")
        print(result)
        print("================================")
        return result
    except Exception as e:
        print(f"调用豆包 API 时发生错误: {e}")


def call_doubao_seedream(
    prompt: str,
    model: str = "Seedream 5.0",
    size: str = "2K"
) -> Optional[str]:
    """
    调用豆包 Seedream 模型生成图片
    
    Args:
        prompt: 图片生成提示词
        model: 用户可见的模型名称，通过 SEEDREAM_MODEL_MAP 映射到真实模型名
        size: 图片尺寸，支持 1K/2K/3K/4K 或宽x高格式如 "720x1280"
    
    Returns:
        生成图片的 URL，失败返回 None
    """
    if not ARK_API_KEY:
        print("错误：未找到 API Key。")
        return None

    real_model = SEEDREAM_MODEL_MAP.get(model, model)

    client = OpenAI(
        base_url=BASE_URL,
        api_key=ARK_API_KEY,
    )

    try:
        print(f"正在调用豆包 Seedream 生成图片...")
        print(f"模型: {model} -> {real_model}, 尺寸: {size}")
        print(f"提示词: {prompt}")
        
        images_response = client.images.generate(
            model=real_model,
            prompt=prompt,
            size=size,
            response_format="url",
            extra_body={
                "watermark": False,
            },
        )

        image_url = images_response.data[0].url
        print("====== 豆包 Seedream 返回结果 ======")
        print(f"图片 URL: {image_url}")
        print("====================================")
        return image_url
    except Exception as e:
        print(f"调用豆包 Seedream API 时发生错误: {e}")
        return None


def call_doubao_seedtext(
    prompt: str,
    model: str = "Seed 2.0 Pro",
    system_prompt: Optional[str] = None
) -> Optional[str]:
    """
    调用豆包 Seed 文本模型生成文本
    
    Args:
        prompt: 文本生成提示词
        model: 用户可见的模型名称，通过 SEEDTEXT_MODEL_MAP 映射到真实模型名
        system_prompt: 可选的系统提示词
    
    Returns:
        生成的文本，失败返回 None
    """
    if not ARK_API_KEY:
        print("错误：未找到 API Key。")
        return None

    real_model = SEEDTEXT_MODEL_MAP.get(model, model)

    client = OpenAI(
        base_url=BASE_URL,
        api_key=ARK_API_KEY,
    )

    try:
        print(f"正在调用豆包 Seed 文本模型...")
        print(f"模型: {model} -> {real_model}")
        if system_prompt:
            print(f"系统提示词: {system_prompt}")
        print(f"提示词: {prompt}")
        
        input_messages = []
        if system_prompt:
            input_messages.append({
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt
                    },
                ],
            })
        
        input_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt
                },
            ],
        })
        
        response = client.responses.create(
            model=real_model,
            input=input_messages
        )

        result = response.output_text
        print("====== 豆包 Seed 文本模型返回结果 ======")
        print(result)
        print("=========================================")
        return result
    except Exception as e:
        print(f"调用豆包 Seed 文本模型 API 时发生错误: {e}")
        return None


if __name__ == "__main__":
    SCHEMA_JSON = SceneAnalysisResult.model_json_schema()
    prompt_text = """
	你是一个专业的视频分镜分析师，同时你也了解用户在搜索视频时的习惯。
    请分析这些视频片段里的画面。
    【重要规则】
    1. 提取 object 时，请使用最通用的词汇，贴合日常口语表达。
    2. search_tags 字段极其重要，请发挥联想，写出用户搜什么词时应该看到这个视频。
    
    ### 1. 营销场景标签
    - **场景类型**：判断属于哪种营销场景
      可选：产品展示、使用场景、情感共鸣、品牌故事、教程演示、对比评测、生活方式展示
      
    - **目标受众**：这个画面最能打动哪类人群？
      示例：Z世代、精致妈妈、职场精英、银发族、健身达人、美食爱好者
      
    ### 2. 商业价值评估 (0-10分)
    - 产品展示清晰度：画面是否适合展示产品细节
    - 情感共鸣度：是否能引起观众情感共鸣
    - 画面美感度：构图、光线、色彩的专业程度
    - 通用适配性：是否容易与其他素材混剪
    """
    # image_url_list = [
    #     "https://freeuuu.obs.cn-east-3.myhuaweicloud.com/ai_picture/video_analysis/scene_004_frame_000093.webp",
    #     "https://freeuuu.obs.cn-east-3.myhuaweicloud.com/ai_picture/video_analysis/scene_004_frame_000141.webp",
    #     "https://freeuuu.obs.cn-east-3.myhuaweicloud.com/ai_picture/video_analysis/scene_004_frame_000189.webp",
    #     "https://freeuuu.obs.cn-east-3.myhuaweicloud.com/ai_picture/video_analysis/scene_004_frame_000221.webp"
    # ]
    # image_url_list = [
    #     "https://freeuuu.obs.cn-east-3.myhuaweicloud.com/ai_picture/car_video_analysis/front/scene_001_frame_000000.webp",
    #     "https://freeuuu.obs.cn-east-3.myhuaweicloud.com/ai_picture/car_video_analysis/front/scene_001_frame_000060.webp",
    #     "https://freeuuu.obs.cn-east-3.myhuaweicloud.com/ai_picture/car_video_analysis/front/scene_001_frame_000120.webp",
    #     "https://freeuuu.obs.cn-east-3.myhuaweicloud.com/ai_picture/car_video_analysis/front/scene_001_frame_000147.webp"
    # ]
    image_url_list = [
        "https://freeuuu.obs.cn-east-3.myhuaweicloud.com/ai_picture/car_video_analysis/inner/scene_001_frame_000000.webp",
        "https://freeuuu.obs.cn-east-3.myhuaweicloud.com/ai_picture/car_video_analysis/inner/scene_001_frame_000060.webp",
        "https://freeuuu.obs.cn-east-3.myhuaweicloud.com/ai_picture/car_video_analysis/inner/scene_001_frame_000116.webp"
    ]

    result = call_doubao_vision(prompt_text, image_url_list, SCHEMA_JSON)
