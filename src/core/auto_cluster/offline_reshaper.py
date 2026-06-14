import numpy as np
from typing import List, Dict, Any, Tuple

class OfflineReshaper:
    """
    负责对累积的历史标签向量进行离线批量聚类和拓扑重构（Correction of semantic drift）。
    默认采用密度敏感的 HDBSCAN 算法，并能自动处理噪声离群点。
    """
    def __init__(self, min_cluster_size: int = 3, min_samples: int = 1):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def reshape_category(self, vectors: np.ndarray, tags: List[str]) -> Tuple[List[int], Dict[int, List[str]]]:
        """
        对单一维度下的标签进行重构。
        返回：
            - labels: 每个标签对应的新聚类 ID（-1 为噪声）
            - clusters: 整理后的聚类簇字典 { cluster_id: [tag1, tag2, ...] }
        """
        if len(vectors) < self.min_cluster_size:
            # 数据量不足以形成簇，直接返回全部为噪声
            return [-1] * len(tags), {-1: list(tags)}

        use_fallback = (len(vectors) < 15)
        labels = []
        if not use_fallback:
            try:
                from sklearn.cluster import HDBSCAN
                clusterer = HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric="cosine"
                )
                clusterer.fit(vectors)
                labels = list(clusterer.labels_)
                if all(l == -1 for l in labels):
                    use_fallback = True
            except (ImportError, AttributeError, ValueError) as e:
                use_fallback = True

        if use_fallback:
            from sklearn.cluster import DBSCAN
            # eps = 0.35 相当于相似度 0.65
            clusterer = DBSCAN(eps=0.35, min_samples=2, metric="cosine")
            clusterer.fit(vectors)
            labels = list(clusterer.labels_)

        # 整理聚类群组
        clusters = {}
        for idx, label in enumerate(labels):
            tag_name = tags[idx]
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(tag_name)

        return labels, clusters

    def calculate_medoids(self, vectors: np.ndarray, labels: List[int], tags: List[str]) -> Dict[int, str]:
        """
        计算每个聚类簇的语义中心词 (Medoid)。
        Medoid 是与同簇内其他所有成员平均余弦相似度最高的词。
        """
        from sentence_transformers import util
        import torch

        # 转为 PyTorch 张量以利用 util.cos_sim 加速
        embeddings_tensor = torch.tensor(vectors)
        sim_matrix = util.cos_sim(embeddings_tensor, embeddings_tensor).numpy()

        # 分类整理索引
        cluster_indices = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            if label not in cluster_indices:
                cluster_indices[label] = []
            cluster_indices[label].append(idx)

        medoids = {}
        for label, indices in cluster_indices.items():
            if len(indices) == 1:
                medoids[label] = tags[indices[0]]
                continue

            best_medoid_idx = indices[0]
            best_avg_sim = -1.0
            
            for idx_a in indices:
                total_sim = 0.0
                for idx_b in indices:
                    if idx_a != idx_b:
                        total_sim += sim_matrix[idx_a][idx_b]
                avg_sim = total_sim / (len(indices) - 1)
                if avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim
                    best_medoid_idx = idx_a
            
            medoids[label] = tags[best_medoid_idx]

        return medoids
