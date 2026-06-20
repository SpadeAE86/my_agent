import asyncio
import sys
import os
import numpy as np

# 将 src 目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.video_match_services.analysis_video import ensure_embedding_model_ready, get_embedding_model
from sentence_transformers import util

async def run_comparison():
    # 1. 准备三组测试数据
    # A. 中文单词标签
    tags_cn = [
        "亚丝娜", "闪光亚丝娜", "桐人", 
        "雷电将军", "八重神子", "神里绫华",
        "双马尾", "金发", "百褶裙", "过膝袜",
        "比特币挖矿", "深度学习"
    ]
    
    # B. 嵌入了标签的中文句子（带上下文）
    sentences_cn = [
        "图中的角色是亚丝娜。",
        "图中的角色是闪光亚丝娜。",
        "图中的角色是桐人。",
        "图中的角色是雷电将军。",
        "图中的角色是八重神子。",
        "图中的角色是神里绫华。",
        "女主角梳着双马尾的发型。",
        "女主角有着一头金色的头发。",
        "她身上穿着一件学院风百褶裙。",
        "她腿上穿着黑色的过膝袜。",
        "比特币挖矿需要消耗大量的电力和计算资源。",
        "深度学习是基于人工神经网络的机器学习方法。"
    ]
    
    # C. 英文/Danbooru 标签 (通常是 Pixiv 和 Danbooru 的标准格式)
    tags_en = [
        "asuna_(sao)", "asuna_(yuuki)", "kirito_(sao)",
        "raiden_shogun", "yae_miko", "kamisato_ayaka",
        "twintails", "blonde_hair", "pleated_skirt", "over-knee_socks",
        "bitcoin_mining", "deep_learning"
    ]
    
    print("1. 正在初始化并加载 SentenceTransformer 向量化模型...")
    await ensure_embedding_model_ready()
    model = get_embedding_model()
    
    print("2. 正在生成三组数据的向量嵌入...")
    emb_tags_cn = model.encode(tags_cn, convert_to_tensor=True)
    emb_sents_cn = model.encode(sentences_cn, convert_to_tensor=True)
    emb_tags_en = model.encode(tags_en, convert_to_tensor=True)
    
    # 计算余弦相似度矩阵
    sim_tags_cn = util.cos_sim(emb_tags_cn, emb_tags_cn).cpu().numpy()
    sim_sents_cn = util.cos_sim(emb_sents_cn, emb_sents_cn).cpu().numpy()
    sim_tags_en = util.cos_sim(emb_tags_en, emb_tags_en).cpu().numpy()
    
    # 3. 定义一些典型对比关系
    # 格式: (名称, 索引A, 索引B)
    test_pairs = [
        ("【同人/同名变体】亚丝娜 vs 闪光亚丝娜", 0, 1),
        ("【同作角色关联】亚丝娜 vs 桐人", 0, 2),
        ("【同作角色关联】雷电将军 vs 八重神子", 3, 4),
        ("【同作角色关联】雷电将军 vs 神里绫华", 3, 5),
        ("【跨作角色关联】亚丝娜 vs 雷电将军", 0, 3),
        ("【属性与属性】  双马尾 vs 金发", 6, 7),
        ("【属性与属性】  百褶裙 vs 过膝袜", 8, 9),
        ("【角色与属性】  亚丝娜 vs 双马尾", 0, 6),
        ("【角色与属性】  神里绫华 vs 百褶裙", 5, 8),
        ("(" + "【噪音与角色】  比特币 vs 亚丝娜".strip() + ")", 10, 0),
        ("(" + "【噪音与属性】  深度学习 vs 双马尾".strip() + ")", 11, 6),
        ("(" + "【噪音与噪音】  比特币 vs 深度学习".strip() + ")", 10, 11)
    ]
    
    print("\n" + "="*80)
    print(f"{'对比关系 (Pairs)':<35} | {'中文单词':<8} | {'中文句子':<8} | {'英文标签':<8}")
    print("-"*80)
    
    for name, a, b in test_pairs:
        s_tag_cn = sim_tags_cn[a][b]
        s_sent_cn = sim_sents_cn[a][b]
        s_tag_en = sim_tags_en[a][b]
        print(f"{name:<35} | {s_tag_cn:.4f}   | {s_sent_cn:.4f}   | {s_tag_en:.4f}")
        
    print("="*80)
    
    # 4. 计算对比度 (Contrast Ratio / Discriminating Power)
    # 对比度定义为：正相关对 (平均) - 负相关/噪音对 (平均)
    # 正相关对: 同人变体、同作角色、属性关联
    pos_pairs = [
        (0, 1), (0, 2), (3, 4), (3, 5), (6, 7), (8, 9)
    ]
    # 负相关/噪音对: 跨作角色、噪音与角色/属性
    neg_pairs = [
        (0, 3), (10, 0), (11, 6)
    ]
    
    def get_avg_sim(matrix, pairs):
        return np.mean([matrix[a][b] for a, b in pairs])
        
    print("\n【 聚类/区分能力指标评估 】")
    for name, matrix in [("中文单词标签 (CN Tags)", sim_tags_cn), 
                         ("中文句子上下文 (CN Sentences)", sim_sents_cn), 
                         ("英文标签 (EN Tags / Danbooru)", sim_tags_en)]:
        pos_avg = get_avg_sim(matrix, pos_pairs)
        neg_avg = get_avg_sim(matrix, neg_pairs)
        contrast = pos_avg - neg_avg
        print(f"- {name}:")
        print(f"  * 正相关对平均相似度 (Pos Avg): {pos_avg:.4f}")
        print(f"  * 噪声/不相关对平均相似度 (Neg Avg): {neg_avg:.4f}")
        print(f"  * 语义区分度/对比度 (Contrast = Pos - Neg): {contrast:.4f}")

if __name__ == "__main__":
    asyncio.run(run_comparison())
