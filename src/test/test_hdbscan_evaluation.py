import asyncio
import sys
import os
import numpy as np

# 将 src 目录添加到 Python 路径中，以加载项目内的服务与模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.video_match_services.analysis_video import ensure_embedding_model_ready, get_embedding_model
from sklearn.cluster import HDBSCAN
from sentence_transformers import util

async def run_evaluation():
    tags = [
        # 类别 A: 车身/外观风格
        "炫酷车身外观", "流线型车身设计", "红色动感车漆", "熏黑轮毂外观", "时尚车身线条",
        # 类别 B: 智能座舱/屏幕/内饰
        "超大前排中控屏", "超宽智慧双联屏", "智能副驾大屏", "豪华真皮座椅", "质感麂皮内饰",
        # 类别 C: 智能驾驶/自动泊车
        "一键自动泊车", "全自动代客泊车", "城市智能辅助驾驶", "高速路段NOA", "超声波雷达感知",
        # 类别 D: 动力/充电/电池
        "800V超快充", "15分钟快速充电", "超长续航电池包", "双电机强劲动力",
        # 噪声点 (Outliers / Noise)
        "车里有只小猫", "香氛系统散发茉莉香气"
    ]
    
    print("1. 正在加载项目向量化模型 (SentenceTransformer)...")
    await ensure_embedding_model_ready()
    model = get_embedding_model()
    
    print("2. 正在生成向量表征...")
    embeddings = model.encode(tags, convert_to_tensor=True)
    
    print("3. 正在运行 HDBSCAN 聚类...")
    # 转化为 numpy 用于 sklearn
    embeddings_np = embeddings.cpu().numpy()
    clusterer = HDBSCAN(min_cluster_size=3, min_samples=1, metric="cosine")
    clusterer.fit(embeddings_np)
    labels = clusterer.labels_
    
    # 4. 计算余弦相似度矩阵 (Cosine Similarity Matrix)
    # util.cos_sim 会计算成对的余弦相似度，值在 -1 到 1 之间（越接近 1 越相似）
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    
    # 5. 整理聚类群组
    clusters = {}
    noise_indices = []
    for idx, label in enumerate(labels):
        if label == -1:
            noise_indices.append(idx)
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
            
    print("\n================== HDBSCAN 聚类可解释性评估 ==================")
    
    # 评估每个簇
    for label, member_indices in sorted(clusters.items()):
        print(f"\n------------------------------------------------------------")
        print(f"【 簇 ID: {label} 】(样本数: {len(member_indices)})")
        print(f"------------------------------------------------------------")
        
        # 寻找该簇的中心代表词 (Medoid)
        # Centroid/Medoid 是与同簇内其他所有成员平均余弦相似度最高的词
        best_medoid_idx = -1
        best_avg_sim = -1.0
        
        # 如果簇内只有一个元素（理论上由于 min_cluster_size=3 不会发生，但做个兼容）
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
        print(f"★ 簇中心代表词 (Medoid): \"{medoid_tag}\" (与其他成员平均余弦相似度: {best_avg_sim:.4f})")
        print("成员详情与到代表词的相似度:")
        for idx in member_indices:
            tag_name = tags[idx]
            sim_to_medoid = sim_matrix[idx][best_medoid_idx]
            # 计算该词与簇内其他所有词的平均相似度
            self_sims = [sim_matrix[idx][other_idx] for other_idx in member_indices if idx != other_idx]
            avg_internal_sim = np.mean(self_sims) if self_sims else 1.0
            
            suffix = " (中心词本身)" if idx == best_medoid_idx else ""
            print(f"  - \"{tag_name:<12}\" | 与中心词相似度: {sim_to_medoid:.4f} | 簇内平均相似度: {avg_internal_sim:.4f}{suffix}")
            
        # 打印簇内相似度较高的 Top Pair，揭示为什么它们被分为一类
        if len(member_indices) >= 2:
            print("\n  簇内最相似的 Top 对关联:")
            pairs = []
            for i in range(len(member_indices)):
                for j in range(i+1, len(member_indices)):
                    idx_a = member_indices[i]
                    idx_b = member_indices[j]
                    pairs.append((idx_a, idx_b, sim_matrix[idx_a][idx_b]))
            pairs.sort(key=lambda x: x[2], reverse=True)
            for idx_a, idx_b, sim in pairs[:3]:
                print(f"    - \"{tags[idx_a]}\" <=> \"{tags[idx_b]}\" | 相似度: {sim:.4f}")

    # 分析噪声/离群点
    if noise_indices:
        print(f"\n------------------------------------------------------------")
        print(f"【 噪声/离群点分析 】(样本数: {len(noise_indices)})")
        print(f"------------------------------------------------------------")
        print("解释为什么这些词被判定为离群噪声（显示它们到各个聚类中心词的最大余弦相似度）：")
        
        # 寻找各个簇的代表词索引
        cluster_medoids = {}
        for label, member_indices in clusters.items():
            best_medoid_idx = -1
            best_avg_sim = -1.0
            for idx_a in member_indices:
                total_sim = 0.0
                for idx_b in member_indices:
                    if idx_a != idx_b:
                        total_sim += sim_matrix[idx_a][idx_b]
                avg_sim = total_sim / (len(member_indices) - 1) if len(member_indices) > 1 else 1.0
                if avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim
                    best_medoid_idx = idx_a
            cluster_medoids[label] = best_medoid_idx
            
        for idx in noise_indices:
            tag_name = tags[idx]
            print(f"  - 离群词: \"{tag_name}\"")
            for label, medoid_idx in sorted(cluster_medoids.items()):
                sim_to_medoid = sim_matrix[idx][medoid_idx]
                print(f"    -> 到 簇 {label} 中心词 (\"{tags[medoid_idx]}\") 的余弦相似度: {sim_to_medoid:.4f}")
            
            # 找到它在整个词表中除自己外最相似的词
            all_sims_except_self = [(other_idx, sim_matrix[idx][other_idx]) for other_idx in range(len(tags)) if idx != other_idx]
            all_sims_except_self.sort(key=lambda x: x[1], reverse=True)
            most_similar_idx, max_sim = all_sims_except_self[0]
            print(f"    -> 整个词表中与它最接近的词: \"{tags[most_similar_idx]}\" | 相似度: {max_sim:.4f}")
            print(f"    [结论]: 该词与其他已知簇中心词的相似度均较低（低于密度聚类的连通阈值），因此被判定为噪声。")

    print("\n==============================================================")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
