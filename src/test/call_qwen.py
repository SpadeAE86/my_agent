"""
Qwen3-VL-8B-Instruct 多模态推理脚本
- 使用 Qwen3VLForConditionalGeneration (Interleaved-MRoPE 3D时空编码)
- 支持文本/图片/视频
- INT8 量化 + 视觉编码器保持 BF16
"""
import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# ============================================================
# 配置区
# ============================================================
MODEL_ID = r"C:\Users\admin\.cache\modelscope\hub\models\Qwen\Qwen3-VL-8B-Instruct"
QUANT_MODE = "int8"  # "none" | "int8" | "nf4"

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(TEST_DIR, "test_image.jpeg")
TEST_VIDEO = os.path.join(TEST_DIR, "test_video.mp4")

# 需要保持 BF16 精度的模块 (完整路径前缀)
# Qwen3-VL 的视觉编码器模块名以 "model.visual" 开头
_MODULES_TO_SKIP = [
    "model.visual",       # 视觉编码器 (ViT + DeepStack)
    "model.merger",       # 视觉-语言投影层 (如果存在)
    "lm_head",            # 语言模型输出头
]


def load_model(model_id: str, quant_mode: str = "none"):
    """加载 Qwen3-VL 多模态模型"""
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
        print("[INFO] 加载模式: INT8 量化 (视觉编码器保持BF16)")
    elif quant_mode == "nf4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # NF4 没有 skip 参数，需要在类上设置
        Qwen3VLForConditionalGeneration._keep_in_fp32_modules = _MODULES_TO_SKIP
        print("[INFO] 加载模式: NF4 量化 (视觉编码器保持BF16)")

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
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
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # Qwen3-VL 的 decode 方式
    generated_ids = outputs[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response


# ============================================================
# 2. 图片理解
# ============================================================
def chat_with_image(processor, model, image_path: str, user_prompt: str,
                    max_new_tokens: int = 1024):
    """传入本地图片路径"""
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

    # 检查 image_grid_thw (T, H, W 时空网格)
    if "image_grid_thw" in inputs:
        print(f"[IMAGE] grid_thw: {inputs['image_grid_thw'].tolist()}")

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = outputs[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response


# ============================================================
# 3. 视频理解 (使用 OpenCV 手动抽帧，传入原生 video content)
# ============================================================
def chat_with_video(processor, model, video_path: str, user_prompt: str,
                    num_frames: int = 16, max_new_tokens: int = 1024):
    """
    用 OpenCV 抽帧后传给 processor。
    Qwen3-VL 的 video processor 会自动处理 temporal_patch_size=2 的时间配对,
    并生成 3D position ids (T, H, W)。
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

    # Qwen3-VL 支持 video 类型的 content
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
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
    print(f"[VIDEO] 总 token 数: {input_len}")

    # 检查 video_grid_thw (T, H, W 三维网格)
    if "video_grid_thw" in inputs:
        print(f"[VIDEO] grid_thw: {inputs['video_grid_thw'].tolist()}")
        thw = inputs["video_grid_thw"][0]
        print(f"[VIDEO] 时间维度T={thw[0].item()}, 高度H={thw[1].item()}, 宽度W={thw[2].item()}")

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = outputs[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response


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
    vram_used = torch.cuda.memory_allocated(0) / 1024**3
    print(f"[OK] 模型加载完成, VRAM: {vram_used:.1f} GB\n")

    # --- 测试1: 纯文本对话 ---
    print("=" * 60)
    print("[TEST 1] 纯文本对话")
    print("=" * 60)
    prompt = "用一句话解释什么是 3D RoPE（M-RoPE）"
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
        user_prompt="描述一下这张图片里有什么，尽量详细",
    )
    print(f"[Assistant] {result}\n")

    # --- 测试3: 视频理解 (Qwen3-VL 的强项: 3D时空编码) ---
    print("=" * 60)
    print("[TEST 3] 视频理解 (3D MRoPE 时空编码)")
    print("=" * 60)
    result = chat_with_video(
        processor, model,
        video_path=TEST_VIDEO,
        user_prompt="详细描述这个视频中发生了什么，包括角色的外观和动作变化",
        num_frames=8,
    )
    print(f"[Assistant] {result}\n")

    vram_peak = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"[VRAM] Peak: {vram_peak:.1f} GB")
