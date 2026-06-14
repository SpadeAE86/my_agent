import asyncio
import sys
import os
import json
import numpy as np
from PIL import Image

# 将 src 目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer, util
from core.auto_cluster.router import CategoryRouter
from utils.call_model_utils import call_doubao_vision

# Qwen3-VL-Embedding-2B 本地路径
MODEL_PATH = r"C:\Users\admin\.cache\huggingface\hub\models--Qwen--Qwen3-VL-Embedding-2B\snapshots\9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
IMAGE_PATH = r"C:\Users\admin\Downloads\tmpjiqhduyp.png"
MOCK_TAGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_tags.json")

async def main():
    print("="*80)
    print("【二阶段打标测试】Qwen3-VL-Embedding-2B 本地视觉召回 + 豆包大模型引导打标")
    print("="*80)
    
    # 1. 验证输入文件存在
    if not os.path.exists(IMAGE_PATH):
        print(f"[错误] 未在下载目录中找到待打标图片: {IMAGE_PATH}")
        return
    if not os.path.exists(MOCK_TAGS_PATH):
        print(f"[错误] 未找到 mock_tags.json: {MOCK_TAGS_PATH}")
        return
        
    with open(MOCK_TAGS_PATH, "r", encoding="utf-8") as f:
        tags = json.load(f)
        
    print(f"1. 成功读取图片: {IMAGE_PATH}")
    print(f"2. 成功加载 {len(tags)} 个候选库标签: {tags}")
    
    # 2. 载入 Qwen3-VL-Embedding-2B 模型
    print("3. 正在从本地加载 Qwen3-VL-Embedding-2B 双模态向量模型...")
    model = SentenceTransformer(MODEL_PATH)
    print("   加载完成！")
    
    # 3. 提取图像和文本标签向量
    print("4. 正在提取图像特征向量 (本地推理，0 Token)...")
    img = Image.open(IMAGE_PATH)
    # 提取图像特征 (SentenceTransformer 会自动使用对应 vision_config 处理)
    img_emb = model.encode(img, convert_to_tensor=True)
    img_emb_np = img_emb.float().cpu().numpy()
    
    print("5. 正在提取候选标签特征向量 (本地推理，0 Token)...")
    # Qwen3-VL-Embedding 支持对文本直接编码
    tag_embs = model.encode(tags, convert_to_tensor=True)
    
    # 4. 计算余弦相似度并排序
    print("6. 开始计算图像与标签库的跨模态相似度...")
    sim_scores = util.cos_sim(img_emb.float(), tag_embs.float())[0].cpu().numpy()
    
    # 结合 CategoryRouter 进行分类归拢
    router = CategoryRouter()
    recalled_by_cat = {}
    
    for idx, tag in enumerate(tags):
        score = float(sim_scores[idx])
        category = router.route(tag)
        
        if category not in recalled_by_cat:
            recalled_by_cat[category] = []
        recalled_by_cat[category].append((tag, score))
        
    # 对每个类别的标签按相关度进行降序排序
    for cat in recalled_by_cat:
        recalled_by_cat[cat].sort(key=lambda x: x[1], reverse=True)
        
    # 5. 构筑打标分类及推荐词提示 (Guideline Info)
    print("\n================== 视觉召回推荐结果 ==================")
    guideline_lines = []
    
    # 按照相关度最高的类目进行输出
    sorted_categories = sorted(
        recalled_by_cat.items(),
        key=lambda x: x[1][0][1] if x[1] else -1.0,
        reverse=True
    )
    
    for cat_path, tag_scores in sorted_categories:
        # 提取 top 3 最相关的词作为 medoid 候选推荐
        top_tags = [f"'{t}' (相似度: {s:.3f})" for t, s in tag_scores[:3]]
        line = f"- 类目 {cat_path}: 推荐样例词包括: {', '.join(top_tags)}"
        guideline_lines.append(line)
        print(line)
    print("======================================================")
    
    guideline_info = "\n".join(guideline_lines)
    
    # 6. 调用豆包 Vision 进行二阶段打标（带推荐词风格吸引）
    print("\n7. 正在调用豆包 Vision 多模态模型进行二阶段精修打标...")
    # 我们模拟传入待分析图片的本地路径（实际系统需上传 OBS，由于是测试，我们可以直接用临时上传或模拟 URL，
    # 或者直接把本地图片传给支持本地的 Vision 接口，但由于 call_doubao_vision 接收 URL 列表，
    # 我们这里用一个真实的公共可访问图片 URL 或者本地转临时上传，或者如果用户有上传到 OBS 的 URL。
    # 慢着！call_doubao_vision 底层需要 OBS URL 列表。为了测试，我们可以先把这张 tmpjiqhduyp.png 自动上传到 OBS！）
    
    obs_url = None
    try:
        from utils.obs_utils import upload_to_obs
        print("   正在将本地测试图片上传至 OBS 临时存储以供大模型访问...")
        obs_url = await upload_to_obs(IMAGE_PATH, obs_prefix="ai_picture/video_analysis/test_qwen")
        print(f"   上传成功，OBS URL: {obs_url}")
    except Exception as e:
        print(f"[警告] 上传 OBS 失败: {e}。将无法调用外部 Vision API，仅能展示第一阶段召回。")
        return
        
    # 如果上传成功，调用豆包多模态 API
    # 采用默认智己汽车视频分镜的 Schema 或者是通用的打标 schema，这里我们采用自定义 prompt
    custom_prompt = (
        "你是一个专业的画面打标助手。\n"
        "请分析这张图片，描述画面的核心内容（比如人物外观、衣着、场景风格、画面氛围等），并输出 5-15 个打标词（search_tags）。\n"
        "请输出符合以下 JSON Schema 的格式：\n"
        "{\n"
        "  \"description\": \"画面描述\",\n"
        "  \"subject\": \"画面主体\",\n"
        "  \"search_tags\": [\"标签1\", \"标签2\", ...]\n"
        "}"
    )
    
    # 我们需要构造一个 Pydantic 的 Json Schema 来配合 call_doubao_vision
    schema_json = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "subject": {"type": "string"},
            "search_tags": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["description", "subject", "search_tags"]
    }
    
    result = await call_doubao_vision(
        prompt=custom_prompt,
        image_url_list=[obs_url],
        schema_json=schema_json,
        guideline_info=guideline_info # 传入我们的召回推荐词引导
    )
    
    print("\n====== 豆包多模态模型打标输出结果 ======")
    if result:
        print(result)
    else:
        print("(未获得有效打标返回)")
    print("=========================================")

if __name__ == "__main__":
    asyncio.run(main())
