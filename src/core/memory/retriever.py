# core/memory/retriever.py — 记忆检索 (RAG)
# 职责:
#   1. 基于关键词的快速检索 (diskcache 索引)
#   2. 基于语义相似度的向量检索 (可选, 后续接入 embedding)
#   3. 混合排序: 结合时间衰减 + 相关性评分
#   4. 返回 top-k 最相关的记忆片段供 prompt 注入
