"""
Gemma 4 E4B-it 多模态推理脚本 (官方推荐方式)
- 使用 AutoModelForMultimodalLM 以正确加载视觉/音频编码器
- 使用 apply_chat_template(tokenize=True) 统一处理文本+图片+视频
"""
import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

# ============================================================
# 配置区
# ============================================================
MODEL_ID = "google/gemma-4-E4B-it"
QUANT_MODE = "int8"  # "none" | "int8" | "nf4"

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(TEST_DIR, "test_image.jpeg")
TEST_VIDEO = os.path.join(TEST_DIR, "test_video.mp4")


# 需要保持 BF16 精度的模块 (完整路径前缀，不是短名)
# llm_int8_skip_modules 内部用 re.match(pattern + ".", full_name) 做前缀匹配
# 所以必须用 "model.vision_tower" 而不是 "vision_tower"
_MODULES_TO_SKIP = [
    "model.vision_tower",           # 视觉编码器 (SigLIP)
    "model.multi_modal_projector",  # 多模态投影层
    "lm_head",                      # 语言模型输出头
]


def load_model(model_id: str, quant_mode: str = "none"):
    """加载多模态模型"""
    load_kwargs = {"device_map": "auto"}

    if quant_mode == "none":
        load_kwargs["torch_dtype"] = torch.bfloat16
        print("[INFO] 加载模式: BF16 全精度")
    elif quant_mode == "int8":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=_MODULES_TO_SKIP,
        )
        print("[INFO] 加载模式: INT8 量化 (视觉编码器/投影层/lm_head 保持 BF16)")
    elif quant_mode == "nf4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # NF4 没有类似的 skip 参数，需要设置 _keep_in_fp32_modules
        print("[INFO] 加载模式: NF4 量化 (视觉编码器保持 BF16)")

    processor = AutoProcessor.from_pretrained(model_id)

    # NF4 模式下，手动设置 _keep_in_fp32_modules (Google 漏设了这个)
    if quant_mode == "nf4":
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
        Gemma4ForConditionalGeneration._keep_in_fp32_modules = _MODULES_TO_SKIP

    model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)

    return processor, model


# ============================================================
# 1. 纯文本对话
# ============================================================
def chat_text(processor, model, user_prompt: str,
              system_prompt: str = "You are a helpful assistant.",
              max_new_tokens: int = 1024):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return processor.parse_response(response)


# ============================================================
# 2. 图片理解
# ============================================================
def chat_with_image(processor, model, image_path: str, user_prompt: str,
                    max_new_tokens: int = 1024):
    """用 PIL Image 对象传入，确保图片数据正确编码"""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    print(f"[IMAGE] 尺寸: {image.size}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    print(f"[IMAGE] token 数: {input_len}")

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return processor.parse_response(response)


# ============================================================
# 3. 视频理解 (手动抽帧，避免 torchvision/torchcodec 依赖问题)
# ============================================================
def chat_with_video(processor, model, video_path: str, user_prompt: str,
                    num_frames: int = 16, max_new_tokens: int = 1024):
    """
    用 OpenCV 手动抽帧，然后作为多张图片传入模型
    这样不依赖 torchvision.io.read_video 或 torchcodec
    """
    import cv2
    from PIL import Image
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    print(f"[VIDEO] 总帧数: {total_frames}, FPS: {fps:.1f}, 时长: {duration:.1f}s")
    print(f"[VIDEO] 均匀抽取 {num_frames} 帧")

    # 均匀抽帧
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    print(f"[VIDEO] 成功提取 {len(frames)} 帧, 尺寸: {frames[0].size if frames else 'N/A'}")

    # 每帧作为一张图片输入
    content = []
    for frame in frames:
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": user_prompt})

    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    print(f"[VIDEO] 总 token 数: {input_len}")

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return processor.parse_response(response)


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        print(f"[GPU] {gpu_name} ({vram_gb} GB VRAM)")

    print(f"[LOAD] 加载模型: {MODEL_ID}")
    processor, model = load_model(MODEL_ID, quant_mode=QUANT_MODE)
    print("[OK] 模型加载完成\n")

    # --- 测试1: 纯文本对话 ---
    print("=" * 60)
    print("[TEST 1] 纯文本对话")
    print("=" * 60)
    prompt = "用一句话解释什么是 KV Cache"
    print(f"[User] {prompt}")
    print("-" * 50)
    result = chat_text(processor, model, prompt)
    print(f"[Assistant] {result}\n")

    # --- 测试2: 图片理解 ---
    print("=" * 60)
    print("[TEST 2] 图片理解")
    print("=" * 60)
    result = chat_with_image(
        processor, model,
        image_path=TEST_IMAGE,
        user_prompt="描述一下这张图片里有什么",
    )
    print(f"[Assistant] {result}\n")

    # --- 测试3: 视频理解 ---
    print("=" * 60)
    print("[TEST 3] 视频理解")
    print("=" * 60)
    result = chat_with_video(
        processor, model,
        video_path=TEST_VIDEO,
        user_prompt="这个视频在讲什么？",
        num_frames=8,
    )
    print(f"[Assistant] {result}\n")