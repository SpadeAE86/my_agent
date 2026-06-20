import asyncio
import sys
import os
import pickle
import numpy as np

# 将 src 目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.video_match_services.analysis_video import ensure_embedding_model_ready, get_embedding_model


async def main():
    print("="*80)
    print("流式增量打标概念演化聚类 MVP 演示准备")
    print("="*80)
    print("说明：")
    print("1. 运行此脚本需要提前安装 river 库: pip install river")
    print("2. 本脚本演示了如何将 SentenceTransformer 的高维向量进行 L2 归一化，")
    print("   使其能够在只支持欧氏距离的 River 库中等价进行余弦距离（Cosine Distance）的流式聚类。")
    print("3. 并展示了如何利用 DenStream 的核心微簇 (CMC) 和离群微簇 (OMC) 映射为 '稳定概念' 与 '临时/萌芽概念'。")
    print("="*80)
    
    try:
        from river import cluster
    except ImportError:
        print("\n[提示] 检测到当前 Python 环境未安装 river 库。")
        print("请在终端执行以下命令进行安装：")
        print("  C:\\Users\\admin\\anaconda3\\envs\\py312\\python.exe -m pip install river")
        print("\n下面将输出本设计的设计逻辑与核心伪代码实现：")
        show_design_info()
        return

    # 1. 初始化模型
    print("\n1. 正在初始化向量模型...")
    await ensure_embedding_model_ready()
    model = get_embedding_model()

    # 2. 初始化 DenStream 聚类器
    # epsilon 控制微簇的半径。由于我们对向量做了 L2 归一化，欧式距离 d 和余弦相似度 s 的关系为: d = sqrt(2 * (1 - s))
    # 假设我们希望余弦相似度 >= 0.7 的标签归为一类，对应的欧氏距离阈值为 d = sqrt(2 * (1 - 0.7)) = sqrt(0.6) ≈ 0.77
    # 因此，我们设置 epsilon = 0.77
    epsilon = np.sqrt(2 * (1.0 - 0.7))
    print(f"2. 初始化 DenStream (epsilon={epsilon:.4f}, 等价余弦相似度阈值=0.70)")
    
    # decay_factor (衰减因子): 每次 learn_one 时，历史点的权重都会乘以 2^(-decay_factor)
    # beta: 决定临时微簇转换为稳定微簇的权重阈值
    # mu: 决定微簇的最小权重
    denstream = cluster.DenStream(
        decay_factor=0.01,
        beta=0.2,
        mu=1.0,
        epsilon=epsilon
    )

    # 模拟流式标签序列（模拟用户和AI在不同时间打下的标签）
    tags_stream = [
        # 第一阶段：开始出现 SAO 相关标签
        ("亚丝娜", "Character_SAO"),
        ("桐人", "Character_SAO"),
        ("闪光亚丝娜", "Character_SAO"),
        
        # 第二阶段：乱入一个噪声标签
        ("比特币挖矿", "Noise"),
        
        # 第三阶段：开始出现 Genshin 相关标签
        ("雷电将军", "Character_Genshin"),
        ("八重神子", "Character_Genshin"),
        ("神里绫华", "Character_Genshin"),
        
        # 第四阶段：SAO 标签再次高频出现，强化 SAO 概念
        ("亚丝娜", "Character_SAO"),
        ("桐人", "Character_SAO"),
        ("阐释者", "Character_SAO"),
        
        # 第五阶段：出现另一个噪声标签，而之前的“比特币挖矿”因为没有再次出现，其权重应该正在衰减
        ("汽车维修", "Noise"),
    ]

    print("\n3. 开始流式输入标签数据并更新聚类场：")
    for step, (tag, category) in enumerate(tags_stream):
        # A. 提取高维语义向量
        emb = model.encode(tag, convert_to_tensor=True)
        emb_np = emb.cpu().numpy()
        
        # B. 核心步骤：L2 归一化，使欧氏距离等价于余弦距离
        norm = np.linalg.norm(emb_np)
        if norm > 0:
            emb_normalized = emb_np / norm
        else:
            emb_normalized = emb_np
            
        # C. 转换为 River 接受的字典格式
        x = {i: float(val) for i, val in enumerate(emb_normalized)}
        
        # D. 流式学习与预测
        # learn_one 会更新微簇的中心、权重（Mass）、方差（Variance）以及衰减旧微簇
        denstream.learn_one(x)
        cluster_id = denstream.predict_one(x)
        
        print(f"  [Step {step:02d}] 标签: \"{tag:<8}\" ({category:<17}) -> 归入簇 ID: {cluster_id}")

    # 4. 打印当前聚类器的状态（稳定概念 vs 临时概念）
    print("\n4. 评估 DenStream 内部概念场状态：")
    
    # 核心微簇 (Core Micro-Clusters, CMCs) -> 对应稳定概念 (Stable Concepts)
    print(f"\n【 稳定概念 (Core Micro-clusters) 】数量: {len(denstream.p_micro_clusters)}")
    for i, cmc in enumerate(denstream.p_micro_clusters):
        center = cmc.center
        weight = cmc.weight
        variance = cmc.variance
        print(f"  * CMC #{i}: 权重(Mass)={weight:.4f}, 半径/方差={variance():.4f}")
        
    # 离群微簇 (Outlier Micro-clusters, OMCs) -> 对应临时/萌芽概念 (Provisional Concepts)
    print(f"\n【 临时/萌芽概念 (Outlier Micro-clusters) 】数量: {len(denstream.o_micro_clusters)}")
    for i, omc in enumerate(denstream.o_micro_clusters):
        center = omc.center
        weight = omc.weight
        variance = omc.variance
        print(f"  * OMC #{i}: 权重(Mass)={weight:.4f}, 半径/方差={variance():.4f}")

    # 5. 模型持久化演示
    print("\n5. 演示模型持久化：")
    model_path = "river_denstream_state.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(denstream, f)
    print(f"  * 已将当前流式聚类器的全部状态（含所有微簇、时间步和权重）保存至: {model_path}")
    
    # 载入测试
    with open(model_path, "rb") as f:
        loaded_denstream = pickle.load(f)
    print(f"  * 成功载入模型，载入后稳定概念数: {len(loaded_denstream.p_micro_clusters)}")
    
    # 清理测试文件
    if os.path.exists(model_path):
        os.remove(model_path)


def show_design_info():
    print("""
================================================================================
流式增量打标概念自组织演化设计方案 (Detailed MVP Steps)
================================================================================

步骤 1: 安装依赖
--------------------------------------------------
确保你的 Python 环境中安装了 river:
pip install river

步骤 2: 向量提取与归一化 (L2 Normalization)
--------------------------------------------------
因为 River 的算法（如 DenStream, DBSTREAM）在底层计算距离时默认采用欧氏距离 (Euclidean Distance)。
为了使用我们效果最好的“余弦相似度”，我们需要在将 Embedding 喂给 River 前进行 L2 归一化：
    v_norm = v / ||v||_2
归一化后，欧氏距离与余弦相似度的关系满足：
    d_euclidean = sqrt(2 * (1 - sim_cosine))
如果我们设定余弦相似度阈值为 s，那么对应的 River epsilon 参数值应设为:
    epsilon = sqrt(2 * (1 - s))

步骤 3: 初始化 River 流式聚类模型
--------------------------------------------------
from river import cluster
import numpy as np

# 设定余弦相似度过滤门槛为 0.70，则 epsilon ≈ 0.7746
eps = np.sqrt(2 * (1.0 - 0.70))

# DenStream 参数配置说明：
# - decay_factor: 衰减速度。值越大，旧标签遗忘得越快（若某个标签长期不出现，其微簇质量会迅速缩减直到死亡）。
# - mu: 维持一个核心微簇(CMC)所需的最小权重。
# - beta: 控制临时微簇(OMC)转化为稳定微簇的门槛系数。OMC 权重达到 beta * mu 即可存活，达到 mu 变为 CMC。
denstream = cluster.DenStream(
    decay_factor=0.01, 
    beta=0.2, 
    mu=1.0, 
    epsilon=eps
)

步骤 4: 更新与查询模型 (Learn & Predict)
--------------------------------------------------
每次在系统中产生新标签（AI反推或用户手动添加）时：
1. 提取词向量 (使用 paraphrase-multilingual-MiniLM-L12-v2)。
2. 对向量做 L2 归一化，转换为 dict 格式。
3. 调用 denstream.learn_one(x_dict) 增量更新聚类引力场。
4. 调用 cluster_id = denstream.predict_one(x_dict) 获取归属概念场。

步骤 5: 系统状态映射与状态归档
--------------------------------------------------
DenStream 会自动在内存里维护两个微簇列表：
* denstream.p_micro_clusters (核心微簇) -> 映射为系统中的 "稳定概念 (Stable Concepts)"
* denstream.o_micro_clusters (离群微簇) -> 映射为系统中的 "萌芽概念 (Provisional Concepts)"

系统可以定期读取这两个列表的中心向量 (center)、权重 (weight) 以及半径 (variance) 存入关系型数据库（如 SQLite/PostgreSQL），
用以呈现在前端的“概念看板”上，方便用户直观查看哪些标签正在汇聚成新概念。
""")

if __name__ == "__main__":
    asyncio.run(main())
