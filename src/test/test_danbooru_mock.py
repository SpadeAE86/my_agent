import asyncio
import sys
import os
import numpy as np

# 将 src 目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.video_match_services.analysis_video import ensure_embedding_model_ready, get_embedding_model
from sentence_transformers import util
from sklearn.cluster import DBSCAN

async def run_danbooru_test():
    # 模拟常见的 Pixiv / Danbooru 标签测试集
    # 分为：中文单字、英文原始Tag (带下划线/括号)、英文清洗Tag (空格替换下划线、去除括号)
    test_data = [
        # 1. 角色组 SAO
        {"cn": "亚丝娜", "en_raw": "asuna_(sao)", "en_clean": "asuna", "cat": "Character_SAO"},
        {"cn": "桐人", "en_raw": "kirito_(sao)", "en_clean": "kirito", "cat": "Character_SAO"},
        {"cn": "闪光亚丝娜", "en_raw": "lightning_flash_asuna", "en_clean": "lightning flash asuna", "cat": "Character_SAO"},
        
        # 2. 角色组 Genshin
        {"cn": "雷电将军", "en_raw": "raiden_shogun", "en_clean": "raiden shogun", "cat": "Character_Genshin"},
        {"cn": "八重神子", "en_raw": "yae_miko", "en_clean": "yae miko", "cat": "Character_Genshin"},
        {"cn": "神里绫华", "en_raw": "kamisato_ayaka", "en_clean": "kamisato ayaka", "cat": "Character_Genshin"},
        
        # 3. 发型发色
        {"cn": "双马尾", "en_raw": "twintails", "en_clean": "twintails", "cat": "Hair"},
        {"cn": "单马尾", "en_raw": "ponytail", "en_clean": "ponytail", "cat": "Hair"},
        {"cn": "金发", "en_raw": "blonde_hair", "en_clean": "blonde hair", "cat": "Hair"},
        {"cn": "黑发", "en_raw": "black_hair", "en_clean": "black hair", "cat": "Hair"},
        
        # 4. 服装
        {"cn": "水手服", "en_raw": "sailor_fuku", "en_clean": "sailor suit", "cat": "Clothing"},
        {"cn": "百褶裙", "en_raw": "pleated_skirt", "en_clean": "pleated skirt", "cat": "Clothing"},
        {"cn": "过膝袜", "en_raw": "over-knee_socks", "en_clean": "over-knee socks", "cat": "Clothing"},
        {"cn": "裤袜", "en_raw": "pantyhose", "en_clean": "pantyhose", "cat": "Clothing"},
        
        # 5. 场景背景
        {"cn": "蓝天白云", "en_raw": "blue_sky", "en_clean": "blue sky", "cat": "Scenery"},
        {"cn": "落日余晖", "en_raw": "sunset", "en_clean": "sunset", "cat": "Scenery"},
        {"cn": "樱花树下", "en_raw": "cherry_blossoms", "en_clean": "cherry blossoms", "cat": "Scenery"},
        
        # 6. 噪音词
        {"cn": "机器学习", "en_raw": "machine_learning", "en_clean": "machine learning", "cat": "Noise"},
        {"cn": "投资理财", "en_raw": "personal_finance", "en_clean": "personal finance", "cat": "Noise"},
        {"cn": "汽车维修", "en_raw": "car_maintenance", "en_clean": "car maintenance", "cat": "Noise"}
    ]
    
    print("1. 正在初始化并加载 SentenceTransformer 向量化模型...")
    await ensure_embedding_model_ready()
    model = get_embedding_model()
    
    # 构造测试列表
    cn_tags = [item["cn"] for item in test_data]
    # 中文句子（带简单模板）
    cn_sentences = [f"这张插画的标签是{item['cn']}。" for item in test_data]
    en_raw_tags = [item["en_raw"] for item in test_data]
    en_clean_tags = [item["en_clean"] for item in test_data]
    
    print("2. 正在生成向量嵌入...")
    emb_cn = model.encode(cn_tags, convert_to_tensor=True)
    emb_cn_sent = model.encode(cn_sentences, convert_to_tensor=True)
    emb_en_raw = model.encode(en_raw_tags, convert_to_tensor=True)
    emb_en_clean = model.encode(en_clean_tags, convert_to_tensor=True)
    
    # 3. 评估指标：内部一致性 (Category Cohesion) 和 类别分离度 (Category Separation)
    # Cohesion: 同一类别的元素两两相似度均值
    # Separation: 该类别元素与其它所有类别元素的相似度均值 (越小越好，代表隔离度高)
    # Contrast: Cohesion - Separation (越大越好)
    
    def evaluate_embeddings(embeddings, name):
        sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        n = len(test_data)
        
        categories = sorted(list(set(item["cat"] for item in test_data if item["cat"] != "Noise")))
        
        print(f"\n================== 评估: {name} ==================")
        print(f"{'类别 (Category)':<20} | {'簇内凝聚力':<10} | {'簇外分离度':<10} | {'对比区分度 (Gap)':<10}")
        print("-" * 65)
        
        all_cohesion = []
        all_separation = []
        
        for cat in categories:
            # 找到当前类别的所有索引
            cat_indices = [i for i, item in enumerate(test_data) if item["cat"] == cat]
            other_indices = [i for i, item in enumerate(test_data) if item["cat"] != cat] # 包括噪音
            
            # 计算 Cohesion
            cohesion_vals = []
            for i in cat_indices:
                for j in cat_indices:
                    if i != j:
                        cohesion_vals.append(sim_matrix[i][j])
            cohesion = np.mean(cohesion_vals) if cohesion_vals else 1.0
            
            # 计算 Separation (到其它所有类别的相似度)
            separation_vals = []
            for i in cat_indices:
                for j in other_indices:
                    separation_vals.append(sim_matrix[i][j])
            separation = np.mean(separation_vals)
            
            contrast = cohesion - separation
            all_cohesion.append(cohesion)
            all_separation.append(separation)
            
            print(f"{cat:<20} | {cohesion:.4f}     | {separation:.4f}     | {contrast:.4f}")
            
        avg_cohesion = np.mean(all_cohesion)
        avg_separation = np.mean(all_separation)
        avg_contrast = avg_cohesion - avg_separation
        print("-" * 65)
        print(f"{'平均值 (Average)':<20} | {avg_cohesion:.4f}     | {avg_separation:.4f}     | {avg_contrast:.4f}")
        
        # 用 DBSCAN 进行一次聚类，看能不能把噪声区分出来
        # eps 设为 0.35 (即余弦相似度 >= 0.65)
        dbscan = DBSCAN(eps=0.35, min_samples=2, metric="cosine")
        labels = dbscan.fit_predict(embeddings.cpu().numpy())
        
        print(f"\nDBSCAN 聚类划分情况 (eps=0.35, min_samples=2):")
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(test_data[idx]["cn"])
        for label, members in sorted(clusters.items()):
            role_name = "噪声/未分类" if label == -1 else f"聚类簇 {label}"
            print(f"  * {role_name}: {members}")
            
    # 执行评估
    evaluate_embeddings(emb_cn, "中文单词标签 (CN Tags)")
    evaluate_embeddings(emb_cn_sent, "中文句子带模板 (CN Sentences with Template)")
    evaluate_embeddings(emb_en_raw, "英文原始标签 (EN Raw Danbooru Tags)")
    evaluate_embeddings(emb_en_clean, "英文清洗标签 (EN Cleaned Tags)")

if __name__ == "__main__":
    asyncio.run(run_danbooru_test())
