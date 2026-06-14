import os
import pickle
import numpy as np
from typing import Dict, Any, Tuple, List

# 尝试导入 river，若没有则启动 Fallback 模式
RIVER_AVAILABLE = False
try:
    from river import cluster
    RIVER_AVAILABLE = True
except ImportError:
    pass

class FallbackMicroCluster:
    """
    自研纯 NumPy 实现的轻量级微簇（Micro-cluster），用以无 River 环境时的等价 Fallback。
    """
    def __init__(self, center: np.ndarray, weight: float = 1.0):
        self.center = np.copy(center)
        self.weight = weight
        self.member_count = 1
        self.variance_val = 0.0

    def update(self, x: np.ndarray):
        self.member_count += 1
        # 增量计算均值/中心
        self.center = (self.center * (self.member_count - 1) + x) / self.member_count
        self.weight += 1.0

    def decay(self, factor: float):
        self.weight *= np.power(2.0, -factor)

    def variance(self) -> float:
        return self.variance_val


class FallbackDenStream:
    """
    用纯 NumPy 模拟的 DenStream 流式密度聚类算法。
    """
    def __init__(self, decay_factor: float = 0.01, beta: float = 0.6, mu: float = 2.0, epsilon: float = 0.7746):
        self.decay_factor = decay_factor
        self.beta = beta
        self.mu = mu
        self.epsilon = epsilon
        
        # p_micro_clusters: 核心/稳定微簇 (CMC)
        self.p_micro_clusters: List[FallbackMicroCluster] = []
        # o_micro_clusters: 离群/萌芽微簇 (OMC)
        self.o_micro_clusters: List[FallbackMicroCluster] = []
        # 全局唯一簇计数器，分配虚拟 ID
        self.cluster_counter = 0
        # 记录微簇 ID 映射
        self.p_cluster_ids: Dict[FallbackMicroCluster, int] = {}

    def learn_one(self, x: np.ndarray) -> int:
        # 1. 衰减历史微簇权重
        for mc in self.p_micro_clusters + self.o_micro_clusters:
            mc.decay(self.decay_factor)
            
        # 2. 尝试合并到最近的核心微簇 (CMC)
        best_cmc = None
        min_cmc_dist = float('inf')
        for mc in self.p_micro_clusters:
            dist = np.linalg.norm(mc.center - x)
            if dist < min_cmc_dist:
                min_cmc_dist = dist
                best_cmc = mc
                
        if best_cmc and min_cmc_dist <= self.epsilon:
            best_cmc.update(x)
            self._prune_and_promote()
            return self.p_cluster_ids[best_cmc]

        # 3. 尝试合并到最近的离群微簇 (OMC)
        best_omc = None
        min_omc_dist = float('inf')
        for mc in self.o_micro_clusters:
            dist = np.linalg.norm(mc.center - x)
            if dist < min_omc_dist:
                min_omc_dist = dist
                best_omc = mc
                
        if best_omc and min_omc_dist <= self.epsilon:
            best_omc.update(x)
            self._prune_and_promote()
            # 刚更新的 OMC 如果晋升了，会有新的核心 ID，否则返回临时的 -1 (噪声)
            if best_omc in self.p_cluster_ids:
                return self.p_cluster_ids[best_omc]
            return -1

        # 4. 无法并入任何核心或离群，创建新的离群微簇
        new_mc = FallbackMicroCluster(x, weight=1.0)
        self.o_micro_clusters.append(new_mc)
        self._prune_and_promote()
        return -1

    def predict_one(self, x: np.ndarray) -> int:
        # 预测只看已稳定的核心微簇 (CMC)
        best_cmc = None
        min_cmc_dist = float('inf')
        for mc in self.p_micro_clusters:
            dist = np.linalg.norm(mc.center - x)
            if dist < min_cmc_dist:
                min_cmc_dist = dist
                best_cmc = mc
        if best_cmc and min_cmc_dist <= self.epsilon:
            return self.p_cluster_ids[best_cmc]
        return -1

    def _prune_and_promote(self):
        # 检查离群微簇晋升与清退
        still_omc = []
        for mc in self.o_micro_clusters:
            if mc.weight >= self.mu:
                # 晋升为稳定核心微簇
                self.p_micro_clusters.append(mc)
                self.p_cluster_ids[mc] = self.cluster_counter
                self.cluster_counter += 1
            elif mc.weight >= self.beta * self.mu:
                # 依然维持离群状态
                still_omc.append(mc)
            # 权重过低（小于 beta * mu）的 OMC 会被自然过滤清退（Pruned）
        self.o_micro_clusters = still_omc

        # 检查核心微簇的清退（若核心微簇权重衰减到低于 mu，降级回 OMC）
        still_cmc = []
        for mc in self.p_micro_clusters:
            if mc.weight < self.mu:
                self.o_micro_clusters.append(mc)
                if mc in self.p_cluster_ids:
                    del self.p_cluster_ids[mc]
            else:
                still_cmc.append(mc)
        self.p_micro_clusters = still_cmc


class OnlineClusterer:
    """
    负责维护每个分类维度空间的增量/流式聚类器。
    若安装了 river，则底层使用真实的 DenStream，否则无缝切入自研 Fallback 模式。
    """
    def __init__(self, sim_threshold: float = 0.70, decay_factor: float = 0.01):
        self.sim_threshold = sim_threshold
        self.decay_factor = decay_factor
        # 根据余弦相似度计算欧氏半径：d = sqrt(2 * (1 - s))
        self.epsilon = np.sqrt(2 * (1.0 - sim_threshold))
        
        # 字典映射：category_name -> Clusterer (DenStream / FallbackDenStream)
        self.clusterers: Dict[str, Any] = {}
        # 已分配的概念名缓存，用于前端友好输出
        self.concept_names: Dict[str, Dict[int, str]] = {}

    def _get_or_create_clusterer(self, category: str) -> Any:
        if category not in self.clusterers:
            if RIVER_AVAILABLE:
                # 采用真正的 River 库流式聚类
                from river import cluster
                self.clusterers[category] = cluster.DenStream(
                    decaying_factor=self.decay_factor,
                    beta=0.6,
                    mu=2.0,
                    epsilon=self.epsilon,
                    n_samples_init=2
                )
            else:
                # 采用自研 Fallback 模式
                self.clusterers[category] = FallbackDenStream(
                    decay_factor=self.decay_factor,
                    beta=0.6,
                    mu=2.0,
                    epsilon=self.epsilon
                )
        return self.clusterers[category]

    def learn_tag(self, tag: str, category: str, vector: np.ndarray) -> Tuple[int, bool]:
        """
        让流式聚类器学习新向量并输出所属簇ID与是否为稳定/核心微簇。
        """
        clusterer = self._get_or_create_clusterer(category)
        
        # River 需要字典格式，Fallback 需要 np.ndarray 格式
        if RIVER_AVAILABLE:
            x_dict = {i: float(val) for i, val in enumerate(vector)}
            clusterer.learn_one(x_dict)
            cluster_id = clusterer.predict_one(x_dict)
        else:
            cluster_id = clusterer.learn_one(vector)
            
        # 判断该 cluster_id 是否属于核心微簇
        is_stable = (cluster_id >= 0)
        return cluster_id, is_stable

    def predict_tag(self, tag: str, category: str, vector: np.ndarray) -> int:
        """
        只预测不更新。
        """
        clusterer = self._get_or_create_clusterer(category)
        if RIVER_AVAILABLE:
            x_dict = {i: float(val) for i, val in enumerate(vector)}
            return clusterer.predict_one(x_dict)
        else:
            return clusterer.predict_one(vector)

    def get_micro_clusters(self, category: str) -> Tuple[List[Any], List[Any]]:
        """
        获取当前维度下的稳定（核心）与临时（离群）微簇快照。
        """
        clusterer = self._get_or_create_clusterer(category)
        return list(clusterer.p_micro_clusters), list(clusterer.o_micro_clusters)

    def save_state(self, filepath: str):
        """
        保存整个在线聚类器的状态到本地 Pickle 文件，支持热启动。
        """
        with open(filepath, "wb") as f:
            pickle.dump({
                "clusterers": self.clusterers,
                "concept_names": self.concept_names,
                "sim_threshold": self.sim_threshold,
                "decay_factor": self.decay_factor,
                "epsilon": self.epsilon
            }, f)

    def load_state(self, filepath: str):
        """
        从 Pickle 文件中恢复状态。
        """
        if not os.path.exists(filepath):
            return
        with open(filepath, "rb") as f:
            state = pickle.load(f)
            self.clusterers = state.get("clusterers", {})
            self.concept_names = state.get("concept_names", {})
            self.sim_threshold = state.get("sim_threshold", 0.70)
            self.decay_factor = state.get("decay_factor", 0.01)
            self.epsilon = state.get("epsilon", 0.7746)
