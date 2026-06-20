import asyncio
import sys
import os
import numpy as np

# 将 src 目录添加到 Python 路径中，以加载项目内的服务与模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.video_match_services.analysis_video import ensure_embedding_model_ready, get_embedding_model
from sklearn.cluster import HDBSCAN
from sentence_transformers import util

async def run_anime_demo():
    # 准备动漫/插画场景常见的 Pixiv & Danbooru 风格标签
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
    
    print("3. 正在运行 HDBSCAN 聚类 (min_cluster_size=3, metric='cosine')...")
    clusterer = HDBSCAN(min_cluster_size=3, min_samples=1, metric="cosine")
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
            
    print("\n================== Pixiv & Danbooru 标签聚类结果 ==================")
    
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
            
        # 打印簇内最相似的前两对
        if len(member_indices) >= 2:
            pairs = []
            for i in range(len(member_indices)):
                for j in range(i+1, len(member_indices)):
                    idx_a = member_indices[i]
                    idx_b = member_indices[j]
                    pairs.append((idx_a, idx_b, sim_matrix[idx_a][idx_b]))
            pairs.sort(key=lambda x: x[2], reverse=True)
            print("  最强关联:")
            for idx_a, idx_b, sim in pairs[:2]:
                print(f"    * \"{tags[idx_a]}\" <=> \"{tags[idx_b]}\" (相似度: {sim:.4f})")

    # 输出噪声点
    if noise_indices:
        print(f"\n【 噪声/离群点 】(数量: {len(noise_indices)}) - 无法并入任何核心动漫分类:")
        # 寻找各个簇的代表词
        cluster_medoids = {}
        for l, m_indices in clusters.items():
            best_medoid_idx = -1
            best_avg_sim = -1.0
            for idx_a in m_indices:
                total_sim = 0.0
                for idx_b in m_indices:
                    if idx_a != idx_b:
                        total_sim += sim_matrix[idx_a][idx_b]
                avg_sim = total_sim / (len(m_indices) - 1) if len(m_indices) > 1 else 1.0
                if avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim
                    best_medoid_idx = idx_a
            cluster_medoids[l] = best_medoid_idx
            
        for idx in noise_indices:
            tag_name = tags[idx]
            print(f"  - \"{tag_name}\"")
            for label, medoid_idx in sorted(cluster_medoids.items()):
                sim_to_medoid = sim_matrix[idx][medoid_idx]
                print(f"    -> 到 簇 {label} 中心 (\"{tags[medoid_idx]}\") 相似度: {sim_to_medoid:.4f}")
            # 全表最相似
            all_sims = [(other_idx, sim_matrix[idx][other_idx]) for other_idx in range(len(tags)) if idx != other_idx]
            all_sims.sort(key=lambda x: x[1], reverse=True)
            print(f"    -> 整个词表中最亲近的词: \"{tags[all_sims[0][0]]}\" (相似度: {all_sims[0][1]:.4f})")

    print("\n===================================================================")

if __name__ == "__main__":
    asyncio.run(run_anime_demo())
