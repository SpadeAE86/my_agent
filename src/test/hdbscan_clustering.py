import asyncio
import sys
import os

# 将 src 目录添加到 Python 路径中，以加载项目内的服务与模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.analysis_video import ensure_embedding_model_ready, get_embedding_model
from sklearn.cluster import HDBSCAN

async def run_demo():
    # 1. 准备演示用的文本标签输入 (包含四个类别和两个孤立噪声标签)
    tags = [
        # 类别 A: 车身/外观风格
        "炫酷车身外观", "流线型车身设计", "红色动感车漆", "熏黑轮毂外观", "时尚车身线条",
        # 类别 B: 智能座舱/屏幕/内饰
        "超大前排中控屏", "超宽智慧双联屏", "智能副驾大屏", "豪华真皮座椅", "质感麂皮内饰",
        # 类别 C: 智能驾驶/自动泊车
        "一键自动泊车", "全自动代客泊车", "城市智能辅助驾驶", "高速路段NOA", "超声波雷达感知",
        # 类别 D: 动力/充电/电池
        "800V超快充", "15分钟快速充电", "超长续航电池包", "双电机强劲动力",
        # 噪声点 (Outliers / Noise): 特立独行、不属于上面任何群组的孤立词
        "车里有只小猫", "香氛系统散发茉莉香气"
    ]
    
    print("1. 正在初始化并加载项目向量化模型 (SentenceTransformer)...")
    await ensure_embedding_model_ready()
    model = get_embedding_model()
    
    print("2. 正在为文本标签生成高维向量嵌入 (Embeddings)...")
    # model.encode 能够直接将文本列表转换为 numpy 特征矩阵
    embeddings = model.encode(tags)
    print(f"   生成成功，特征矩阵形状为: {embeddings.shape} (共 {embeddings.shape[0]} 个标签，每个维度 {embeddings.shape[1]})")
    
    print("3. 正在调用 scikit-learn 的 HDBSCAN 进行密度聚类...")
    # 初始化 HDBSCAN 聚类器
    # - min_cluster_size: 形成一个簇所需要的最小样本数。演示中设为 3。
    # - min_samples: 确定密度计算的邻域保守性。
    # - metric: 余弦距离（'cosine'）非常适合评估文本向量的方向相似性。
    clusterer = HDBSCAN(min_cluster_size=3, min_samples=1, metric="cosine")
    clusterer.fit(embeddings)
    
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_  # 样本在所属簇内的成员置信度 (0.0 到 1.0)
    
    print("\n================== HDBSCAN 聚类结果 ==================")
    clusters = {}
    noise = []
    
    for idx, (tag, label, prob) in enumerate(zip(tags, labels, probabilities)):
        if label == -1:
            noise.append((tag, prob))
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((tag, prob))
            
    # 输出聚类完成的主题簇
    for label, items in sorted(clusters.items()):
        print(f"\n[簇 ID: {label}] (样本数: {len(items)})")
        for item, prob in items:
            print(f"  - {item:<15} (聚类置信度: {prob:.4f})")
            
    # 输出未能归入任何簇的孤立词（噪声）
    if noise:
        print(f"\n[噪声/离群点] (样本数: {len(noise)}) - 判定为孤立词，未强行并入任何大类:")
        for item, prob in noise:
            print(f"  - {item:<15} (置信度: {prob:.4f})")
    print("\n=======================================================")

if __name__ == "__main__":
    asyncio.run(run_demo())
