import asyncio
import sys
import os
import numpy as np

# 将 src 目录添加到 Python 路径中，以加载项目内的服务与模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.analysis_video import ensure_embedding_model_ready, get_embedding_model
from sklearn.cluster import DBSCAN
from sentence_transformers import util

async def run_anime_dbscan():
    tags = [
        # 组 1: 刀剑神域 (Sword Art Online) 关联标签
        "亚丝娜", "桐人", "阐释者", "闪光亚丝娜", "刀剑神域",
        
        # 组 2: 原神 (Genshin Impact) 关联标签
        "雷电将军", "八重神子", "神里绫华", "提瓦特大陆", "原神",
        
        # 组 3: 萌属性/外貌与服饰特征 (Anime Girl Appearance & Attire)
        "双马尾", "金发", "白毛", "百褶裙", "过膝袜", "水手服",
        
        # 组 4: 场景与画面氛围 (Scenery & Background)
        "樱花树下", "落日余晖", "教室课桌", "晴朗天空",
        
        # 孤立噪声点 (Outliers / Unrelated)
        "比特币挖矿", "深度学习聚类"
    ]
    
    print("1. 正在初始化并加载 SentenceTransformer 向量化模型...")
    await ensure_embedding_model_ready()
    model = get_embedding_model()
    
    print("2. 正在生成高维语义向量 (Embeddings)...")
    embeddings = model.encode(tags, convert_to_tensor=True)
    embeddings_np = embeddings.cpu().numpy()
    
    print("3. 正在运行 DBSCAN 聚类 (eps=0.35, min_samples=2, metric='cosine')...")
    # eps=0.35 意味着余弦距离 <= 0.35，即余弦相似度 >= 0.65
    clusterer = DBSCAN(eps=0.35, min_samples=2, metric="cosine")
    clusterer.fit(embeddings_np)
    labels = clusterer.labels_
    
    # 计算余弦相似度矩阵
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    
    # 整理聚类群组
    clusters = {}
    noise_indices = []
    for idx, label in enumerate(labels):
        if label == -1:
            noise_indices.append(idx)
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
            
    print("\n================== DBSCAN 标签聚类结果 ==================")
    
    # 输出主题簇
    for label, member_indices in sorted(clusters.items()):
        # 寻找该簇的中心代表词 (Medoid)
        best_medoid_idx = -1
        best_avg_sim = -1.0
        if len(member_indices) == 1:
            best_medoid_idx = member_indices[0]
            best_avg_sim = 1.0
        else:
            for idx_a in member_indices:
                total_sim = 0.0
                for idx_b in member_indices:
                    if idx_a != idx_b:
                        total_sim += sim_matrix[idx_a][idx_b]
                avg_sim = total_sim / (len(member_indices) - 1)
                if avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim
                    best_medoid_idx = idx_a
                    
        medoid_tag = tags[best_medoid_idx]
        print(f"\n【 簇 ID: {label} 】(成员数: {len(member_indices)}) -> 语义中心词: \"{medoid_tag}\"")
        for idx in member_indices:
            tag_name = tags[idx]
            sim_to_medoid = sim_matrix[idx][best_medoid_idx]
            self_sims = [sim_matrix[idx][other_idx] for other_idx in member_indices if idx != other_idx]
            avg_internal_sim = np.mean(self_sims) if self_sims else 1.0
            
            suffix = " (中心词)" if idx == best_medoid_idx else ""
            print(f"  - \"{tag_name:<12}\" | 到中心相似度: {sim_to_medoid:.4f} | 簇内平均相似度: {avg_internal_sim:.4f}{suffix}")

    # 输出噪声点
    if noise_indices:
        print(f"\n【 噪声/离群点 】(数量: {len(noise_indices)}) - 无法并入任何核心二次元分类:")
        for idx in noise_indices:
            tag_name = tags[idx]
            # 找到它在整个词表中除自己外最相似的词
            all_sims = [(other_idx, sim_matrix[idx][other_idx]) for other_idx in range(len(tags)) if idx != other_idx]
            all_sims.sort(key=lambda x: x[1], reverse=True)
            print(f"  - \"{tag_name:<12}\" | 最亲近的词: \"{tags[all_sims[0][0]]}\" (相似度: {all_sims[0][1]:.4f})")

    print("\n===================================================================")

if __name__ == "__main__":
    asyncio.run(run_anime_dbscan())
