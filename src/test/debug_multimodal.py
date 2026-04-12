"""验证正确的 skip_modules 路径前缀"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
import bitsandbytes as bnb

MODEL_ID = "google/gemma-4-E4B-it"
TEST_FRAME = r"C:\AI\AiGithubProject\DIYProject\src\test\debug_frame.jpg"
TEST_IMAGE = r"C:\AI\AiGithubProject\DIYProject\src\test\test_image.jpeg"

print("[LOAD] INT8 with vision_tower skipped via correct prefix...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=[
            "model.vision_tower",
            "model.multi_modal_projector",
            "lm_head",
        ],
    ),
    device_map="auto",
)

# Verify vision is NOT quantized
q_count = 0
for name, module in model.named_modules():
    if "vision_tower" in name and isinstance(module, bnb.nn.Linear8bitLt):
        q_count += 1
print(f"[CHECK] Vision tower quantized modules: {q_count} (should be 0)")

# Check a vision module dtype
for name, param in model.named_parameters():
    if "vision_tower" in name and "weight" in name:
        print(f"[CHECK] {name}: dtype={param.dtype}")
        break

vram = torch.cuda.memory_allocated(0) / 1024**3
print(f"[VRAM] {vram:.1f} GB used")

# Test 1: video frame (anime character dancing)
img = Image.open(TEST_FRAME).convert("RGB")
print(f"\n[TEST 1] Video frame: {img.size}")
messages = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text", "text": "Describe what you see in this image in detail."},
]}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True, return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=False)
print(f"[RESULT 1] {processor.parse_response(response)}")

# Test 2: cherry blossom photo
img2 = Image.open(TEST_IMAGE).convert("RGB")
print(f"\n[TEST 2] Cherry blossom: {img2.size}")
messages2 = [{"role": "user", "content": [
    {"type": "image", "image": img2},
    {"type": "text", "text": "Describe this image in detail."},
]}]
inputs2 = processor.apply_chat_template(
    messages2, tokenize=True, return_dict=True, return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
outputs2 = model.generate(**inputs2, max_new_tokens=512)
response2 = processor.decode(outputs2[0][inputs2['input_ids'].shape[-1]:], skip_special_tokens=False)
print(f"[RESULT 2] {processor.parse_response(response2)}")

print(f"\n[VRAM] Peak: {torch.cuda.max_memory_allocated(0) / 1024**3:.1f} GB")
