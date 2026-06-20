import numpy as np
from typing import List
from services.video_match_services.analysis_video import ensure_embedding_model_ready, get_embedding_model

class TagEmbedder:
    """
    负责将标签进行特定空间的引导词包装（Contextual Prompt Frame）并提取高维语义向量。
    最后进行 L2 归一化，使得下游欧氏距离等价于余弦距离。
    """
    def __init__(self):
        # 针对不同类别的上下文引导模板 (Contextual Prompt Frames)
        # 用自注意力机制在预训练通用模型中实现“解耦投影”
        self.templates = {
            "Character": "A character name: {tag}",
            "Hair": "A hairstyle description: {tag}",
            "Clothing": "A clothing or outfit description: {tag}",
            "Scenery": "A scene or background description: {tag}",
            "General": "A tag description: {tag}"
        }

    async def get_embedding(self, tag: str, category: str) -> np.ndarray:
        """
        获取单个标签在指定空间维度下的归一化向量。
        """
        # 1. 确保 SentenceTransformer 已就绪
        await ensure_embedding_model_ready()
        model = get_embedding_model()
        
        # 2. 匹配模板包装
        if category == "raw" or not category:
            prompted_text = tag
        else:
            fallback = self.templates.get("General", "A tag description: {tag}")
            template = self.templates.get(category, fallback)
            prompted_text = template.format(tag=tag)
        
        # 3. 提取特征向量
        embedding_tensor = model.encode(prompted_text, convert_to_tensor=True)
        embedding_np = embedding_tensor.cpu().numpy()
        
        # 4. L2 归一化 (核心步骤：使得欧氏距离计算结果等同于余弦距离)
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_normalized = embedding_np / norm
        else:
            embedding_normalized = embedding_np
            
        return embedding_normalized

    async def get_embeddings_batch(self, tags: List[str], categories: List[str]) -> np.ndarray:
        """
        批量获取标签的归一化向量。
        """
        await ensure_embedding_model_ready()
        model = get_embedding_model()
        
        prompted_texts = []
        for tag, category in zip(tags, categories):
            if category == "raw" or not category:
                prompted_texts.append(tag)
            else:
                fallback = self.templates.get("General", "A tag description: {tag}")
                template = self.templates.get(category, fallback)
                prompted_texts.append(template.format(tag=tag))
            
        embeddings_tensor = model.encode(prompted_texts, convert_to_tensor=True)
        embeddings_np = embeddings_tensor.cpu().numpy()
        
        # 批量 L2 归一化
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        # 避免除以 0
        norms = np.where(norms == 0, 1.0, norms)
        embeddings_normalized = embeddings_np / norms
        
        return embeddings_normalized


