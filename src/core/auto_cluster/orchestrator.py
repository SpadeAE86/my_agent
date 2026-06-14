import numpy as np
import json
import os
import pickle
from typing import Dict, Any, List, Tuple

from .router import CategoryRouter
from .embedder import TagEmbedder
from .online_clusterer import OnlineClusterer
from .offline_reshaper import OfflineReshaper
from .llm_arbitrator import LLMArbitrator

class AutoClusterOrchestrator:
    """
    自组织语义空间系统总编排器（升级版：支持层次化本体生长与路径枚举法）。
    串联 Routing, Embedding, Online clustering, Offline reshaping, 以及 LLM Arbitration。
    """
    def __init__(self, sim_threshold: float = 0.70, decay_factor: float = 0.01):
        self.router = CategoryRouter()
        self.embedder = TagEmbedder()
        self.online_clusterer = OnlineClusterer(sim_threshold=sim_threshold, decay_factor=decay_factor)
        self.offline_reshaper = OfflineReshaper()
        self.llm_arbitrator = LLMArbitrator()
        
        # 知识图谱本体分类树 (Taxonomy Tree) - 默认顶级类目
        self.taxonomy: Dict[str, Any] = {
            "Character": {},
            "Hair": {},
            "Clothing": {},
            "Scenery": {},
            "General": {}
        }
        
        # 内存历史记录，采用【路径枚举法 (Path Enumeration)】支持无限级的树形分裂
        # 键是分类路径，如 "General" 或 "Weapon/Cold_Weapon"
        self.history: Dict[str, List[Dict[str, Any]]] = {
            "Character": [],
            "Hair": [],
            "Clothing": [],
            "Scenery": [],
            "General": []
        }
        
        # 存储稳定概念的名字映射 { category_path: { cluster_id: concept_name } }
        self.stable_concept_names: Dict[str, Dict[int, str]] = {
            "Character": {},
            "Hair": {},
            "Clothing": {},
            "Scenery": {},
            "General": {}
        }

        # 存储被弃用/变更的类目路径重定向映射 { old_path: new_path }
        self.category_redirects: Dict[str, str] = {}

        # 存储逻辑软链接映射 { category_path: List[tag] }
        self.logical_soft_links: Dict[str, List[str]] = {}

    def resolve_category_path(self, path: str) -> str:
        """
        递归地解析类目路径的重定向，直到指向最新有效的活动路径。
        """
        visited = set()
        while path in self.category_redirects:
            if path in visited:
                break # 防止死循环
            visited.add(path)
            path = self.category_redirects[path]
        return path

    async def process_tag(self, tag: str) -> Dict[str, Any]:
        """
        流式处理一个新产生的标签。
        """
        # 1. Routing 路由分配，并进行重定向解析得到当前最新的活动路径
        raw_path = self.router.route(tag)
        category_path = self.resolve_category_path(raw_path)
        
        # 2. Embedding + L2 归一化（如果是新生成的子空间，会使用专属模板）
        if category_path not in self.embedder.templates:
            # 自动追溯父级模板作为兜底
            parent = category_path.split("/")[0]
            self.embedder.templates[category_path] = self.embedder.templates.get(parent, self.embedder.templates["General"])
            
        vector = await self.embedder.get_embedding(tag, category_path)
        
        # 3. Online Absorb 流式吸纳
        cluster_id, is_stable = self.online_clusterer.learn_tag(tag, category_path, vector)
        
        # 4. 写入历史记录
        if category_path not in self.history:
            self.history[category_path] = []
            self.stable_concept_names[category_path] = {}
            
        exists = any(item["tag"] == tag for item in self.history[category_path])
        if not exists:
            self.history[category_path].append({
                "tag": tag,
                "vector": vector,
                "cluster_id": cluster_id
            })
            
        if is_stable and cluster_id not in self.stable_concept_names[category_path]:
            self.stable_concept_names[category_path][cluster_id] = f"{tag}相关概念"
            
        return {
            "tag": tag,
            "category_path": category_path,
            "cluster_id": cluster_id,
            "is_stable": is_stable,
            "concept_name": self.stable_concept_names[category_path].get(cluster_id, "临时/萌芽概念")
        }

    def _insert_taxonomy_node(self, tree: Dict[str, Any], path_parts: List[str]):
        """
        递归向本体树中插入多级子类目节点
        """
        if not path_parts:
            return
        node = path_parts[0]
        if node not in tree:
            tree[node] = {}
        self._insert_taxonomy_node(tree[node], path_parts[1:])

    async def run_offline_reshape(self, category_path: str) -> Dict[str, Any]:
        """
        对指定维度路径下的所有历史标签执行离线重构（Offline Reshaping）。
        """
        items = self.history.get(category_path, [])
        if not items:
            return {"status": "empty", "category_path": category_path}
            
        tags = [item["tag"] for item in items]
        vectors = np.array([item["vector"] for item in items])
        
        labels, clusters = self.offline_reshaper.reshape_category(vectors, tags)
        
        for idx, label in enumerate(labels):
            items[idx]["cluster_id"] = label
            
        medoids = self.offline_reshaper.calculate_medoids(vectors, labels, tags)
        
        new_concept_names = {}
        for label, member_tags in clusters.items():
            if label == -1:
                continue
            medoid = medoids.get(label)
            concept_name = await self.llm_arbitrator.name_cluster(member_tags, medoid=medoid)
            new_concept_names[label] = concept_name
            
        # 合并冲突检查
        active_labels = sorted(list(new_concept_names.keys()))
        merged_map = {lbl: lbl for lbl in active_labels}
        
        for i in range(len(active_labels)):
            for j in range(i + 1, len(active_labels)):
                lbl_a = active_labels[i]
                lbl_b = active_labels[j]
                
                target_a = merged_map[lbl_a]
                target_b = merged_map[lbl_b]
                if target_a == target_b:
                    continue
                    
                should_merge = await self.llm_arbitrator.decide_merge(clusters[lbl_a], clusters[lbl_b])
                if should_merge:
                    for k in merged_map:
                        if merged_map[k] == target_b:
                            merged_map[k] = target_a
                            
        final_concept_names = {}
        final_clusters = {}
        for orig_label, target_label in merged_map.items():
            if target_label not in final_clusters:
                final_clusters[target_label] = []
            final_clusters[target_label].extend(clusters[orig_label])
            final_concept_names[target_label] = new_concept_names[target_label]

        self.stable_concept_names[category_path] = final_concept_names
        
        return {
            "status": "success",
            "category_path": category_path,
            "total_tags": len(tags),
            "original_clusters_count": len(active_labels),
            "final_concepts": [
                {
                    "concept_id": cid,
                    "concept_name": final_concept_names[cid],
                    "medoid": medoids.get(cid, "未知"),
                    "tags": tags_list
                }
                for cid, tags_list in final_clusters.items()
            ],
            "noise_tags": clusters.get(-1, [])
        }

    async def evolve_general_ontology(self) -> List[Dict[str, Any]]:
        """
        【本体系统级生长进化（第六步 - LLM 仲裁）】
        对 General 空间中的历史累积数据运行离线聚类，发现高密度类簇后，
        结合当前的本体树进行 LLM 研判，动态生长出新的顶级类目（水平生长）或子类目（垂直生长）。
        """
        items = self.history.get("General", [])
        if len(items) < 3:
            return []
            
        tags = [item["tag"] for item in items]
        vectors = np.array([item["vector"] for item in items])
        
        # 1. 运行临时聚类发现凝聚簇
        labels, clusters = self.offline_reshaper.reshape_category(vectors, tags)
        
        growth_reports = []
        
        # 2. 评估每一个聚集出来的簇
        for label, member_tags in clusters.items():
            if label == -1:
                continue # 忽略噪声点
                
            # 调用 LLM 仲裁本体如何生长
            growth_decision = await self.llm_arbitrator.assess_ontology_growth(self.taxonomy, member_tags)
            decision = growth_decision.get("decision", "REJECT")
            
            if decision == "REJECT":
                continue
                
            new_node = growth_decision.get("new_node_name")
            alias = growth_decision.get("alias", new_node)
            parent_path = growth_decision.get("parent_path", "")
            template = growth_decision.get("template", "A tag description: {tag}")
            
            # 3. 构造路径 (Path Enumeration)
            if decision == "VERTICAL" and parent_path:
                full_path = f"{parent_path}/{new_node}"
            else:
                full_path = new_node
                
            # 4. 更新本体树 (Taxonomy Tree)
            path_parts = full_path.split("/")
            self._insert_taxonomy_node(self.taxonomy, path_parts)
            
            # 5. 更新 Embedder 模板
            self.embedder.templates[full_path] = template
            
            # 6. 更新 Router 路由匹配
            for tag in member_tags:
                self.router.fast_map[tag] = full_path
                
            # 7. 动态初始化新空间的存储
            if full_path not in self.history:
                self.history[full_path] = []
                self.stable_concept_names[full_path] = {}
                
            # 8. 重构迁移历史数据 (Data migration & Re-embedding)
            # 从 General 列表中踢除这批标签
            self.history["General"] = [item for item in self.history["General"] if item["tag"] not in member_tags]
            
            # 用新空间的专属模板重新计算 Embedding 向量，并移入新空间
            for tag in member_tags:
                vector = await self.embedder.get_embedding(tag, full_path)
                self.history[full_path].append({
                    "tag": tag,
                    "vector": vector,
                    "cluster_id": 0 # 重置为新空间的初始簇
                })
                
            growth_reports.append({
                "decision": decision,
                "full_path": full_path,
                "alias": alias,
                "template": template,
                "tags_migrated": member_tags
            })
            
        return growth_reports

    def save_state(self, filepath: str):
        """
        持久化当前总编排器及其所有子组件的状态（含路由表、向量映射、微簇状态及历史记录）。
        """
        with open(filepath, "wb") as f:
            pickle.dump({
                "history": self.history,
                "taxonomy": self.taxonomy,
                "stable_concept_names": self.stable_concept_names,
                "router_fast_map": self.router.fast_map,
                "embedder_templates": self.embedder.templates,
                "online_clusterer": self.online_clusterer,
                "category_redirects": self.category_redirects,
                "logical_soft_links": self.logical_soft_links
            }, f)

    def load_state(self, filepath: str) -> bool:
        """
        从本地 Pickle 文件恢复上一次运行的状态，支持增量热启动。
        """
        if not os.path.exists(filepath):
            return False
        with open(filepath, "rb") as f:
            state = pickle.load(f)
            self.history = state["history"]
            self.taxonomy = state["taxonomy"]
            self.stable_concept_names = state["stable_concept_names"]
            self.router.fast_map = state["router_fast_map"]
            self.embedder.templates = state["embedder_templates"]
            self.online_clusterer = state["online_clusterer"]
            self.category_redirects = state.get("category_redirects", {})
            self.logical_soft_links = state.get("logical_soft_links", {})
        return True

    def _delete_taxonomy_node(self, tree: Dict[str, Any], path_parts: List[str]) -> bool:
        """
        递归地从分类本体树中删除节点。
        """
        if not path_parts:
            return False
        node = path_parts[0]
        if node in tree:
            if len(path_parts) == 1:
                del tree[node]
                return True
            else:
                return self._delete_taxonomy_node(tree[node], path_parts[1:])
        return False

    async def _migrate_category_data(self, old_path: str, new_path: str):
        """
        内部辅助方法：迁移旧分类的历史、映射、模板，并在模板改变时重新进行向量计算。
        """
        # 1. 迁移历史数据
        old_history = self.history.pop(old_path, [])
        if new_path not in self.history:
            self.history[new_path] = []
            
        # 合并历史，防止重复 tag
        existing_tags = {item["tag"] for item in self.history[new_path]}
        tags_to_add = []
        for item in old_history:
            if item["tag"] not in existing_tags:
                self.history[new_path].append(item)
                tags_to_add.append(item["tag"])
                
        # 2. 迁移概念名称
        old_concepts = self.stable_concept_names.pop(old_path, {})
        if new_path not in self.stable_concept_names:
            self.stable_concept_names[new_path] = {}
        self.stable_concept_names[new_path].update(old_concepts)
        
        # 3. 迁移 Embedding 模板并做空间投影刷新
        old_template = self.embedder.templates.pop(old_path, None)
        if new_path not in self.embedder.templates:
            if old_template:
                self.embedder.templates[new_path] = old_template
            else:
                parent = new_path.split("/")[0]
                self.embedder.templates[new_path] = self.embedder.templates.get(parent, self.embedder.templates["General"])
                
        # 4. 重新向量化以匹配新子空间语义投影 (Dynamic Re-embedding)
        if tags_to_add:
            new_vectors = await self.embedder.get_embeddings_batch(
                tags_to_add, [new_path] * len(tags_to_add)
            )
            # 更新已添加项的向量
            vector_map = dict(zip(tags_to_add, new_vectors))
            for item in self.history[new_path]:
                if item["tag"] in vector_map:
                    item["vector"] = vector_map[item["tag"]]
                    
        # 5. 更新路由映射 fast_map
        for tag, path in list(self.router.fast_map.items()):
            if path == old_path:
                self.router.fast_map[tag] = new_path

    async def restructure_taxonomy(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        原子化执行本体树结构重构操作，迁移历史标签、路由及模板，并重新向量化。
        """
        executed_actions = []
        for action_info in actions:
            action = action_info.get("action")
            reason = action_info.get("reason", "")
            
            if action == "SHIFT_DOWN":
                source = action_info.get("source_path")
                target_parent = action_info.get("target_parent_path")
                if not source or not target_parent:
                    continue
                
                source_basename = source.split("/")[-1]
                new_path = f"{target_parent}/{source_basename}"
                
                # 从原位置删除，插入到新位置
                deleted = self._delete_taxonomy_node(self.taxonomy, source.split("/"))
                if not deleted:
                    continue
                self._insert_taxonomy_node(self.taxonomy, new_path.split("/"))
                
                # 建立重定向并迁移数据
                self.category_redirects[source] = new_path
                await self._migrate_category_data(source, new_path)
                
                executed_actions.append({
                    "action": "SHIFT_DOWN",
                    "source_path": source,
                    "new_path": new_path,
                    "reason": reason
                })
                
            elif action == "SHIFT_UP":
                source = action_info.get("source_path")
                target_parent = action_info.get("target_parent_path", "")
                if not source:
                    continue
                
                source_basename = source.split("/")[-1]
                if target_parent:
                    new_path = f"{target_parent}/{source_basename}"
                else:
                    new_path = source_basename
                    
                deleted = self._delete_taxonomy_node(self.taxonomy, source.split("/"))
                if not deleted:
                    continue
                self._insert_taxonomy_node(self.taxonomy, new_path.split("/"))
                
                self.category_redirects[source] = new_path
                await self._migrate_category_data(source, new_path)
                
                executed_actions.append({
                    "action": "SHIFT_UP",
                    "source_path": source,
                    "new_path": new_path,
                    "reason": reason
                })
                
            elif action == "CREATE_PARENT":
                sources = action_info.get("source_paths", [])
                new_parent = action_info.get("new_parent_name")
                alias = action_info.get("alias", new_parent)
                if not sources or not new_parent:
                    continue
                
                # 插入父节点
                self._insert_taxonomy_node(self.taxonomy, new_parent.split("/"))
                
                moved_sources = []
                for src in sources:
                    src_basename = src.split("/")[-1]
                    new_path = f"{new_parent}/{src_basename}"
                    
                    deleted = self._delete_taxonomy_node(self.taxonomy, src.split("/"))
                    if deleted:
                        self._insert_taxonomy_node(self.taxonomy, new_path.split("/"))
                        self.category_redirects[src] = new_path
                        await self._migrate_category_data(src, new_path)
                        moved_sources.append((src, new_path))
                        
                if moved_sources:
                    executed_actions.append({
                        "action": "CREATE_PARENT",
                        "new_parent": new_parent,
                        "alias": alias,
                        "moved_sources": moved_sources,
                        "reason": reason
                    })
                    
            elif action == "MERGE_CATEGORIES":
                source = action_info.get("source_path")
                target = action_info.get("target_path")
                if not source or not target:
                    continue
                
                deleted = self._delete_taxonomy_node(self.taxonomy, source.split("/"))
                if not deleted:
                    continue
                    
                self.category_redirects[source] = target
                await self._migrate_category_data(source, target)
                
                executed_actions.append({
                    "action": "MERGE_CATEGORIES",
                    "source_path": source,
                    "target_path": target,
                    "reason": reason
                })
                
        return executed_actions

    async def evolve_taxonomy_structure(self) -> List[Dict[str, Any]]:
        """
        触发整套分类结构的智能研判与优化。
        提取当前活动类目及其典型内容，让 LLM 做重构判决，最后执行物理迁移。
        """
        # 1. 获取所有活动叶子节点路径
        def get_leaf_paths(tree, prefix="") -> List[str]:
            paths = []
            for node, subtree in tree.items():
                path = f"{prefix}/{node}" if prefix else node
                if not subtree:
                    paths.append(path)
                else:
                    paths.extend(get_leaf_paths(subtree, path))
            return paths
            
        active_paths = get_leaf_paths(self.taxonomy)
        if "General" not in active_paths:
            active_paths.append("General")
            
        # 2. 收集每个类目下的典型标签样本
        category_summaries = {}
        for path in active_paths:
            items = self.history.get(path, [])
            # 最多取 12 个标签作为样本
            category_summaries[path] = [it["tag"] for it in items[:12]]
            
        # 3. 调用 LLM 进行研判
        actions = await self.llm_arbitrator.analyze_taxonomy_restructure(self.taxonomy, category_summaries)
        if not actions:
            return []
            
        # 4. 执行重构操作
        executed = await self.restructure_taxonomy(actions)
        return executed

    async def discover_soft_links(self, similarity_threshold: float = 0.45) -> Dict[str, List[str]]:
        """
        利用向量相似度检索，自动发掘跨类目的逻辑“软链接”。
        例如：使用类目名称 "Hair" 向量去计算所有其他类目标签的相似度，
        对关联度高但物理不在该类目的标签建立软链接，提升用户找词体验。
        采用 raw 模式提取向量，避免模板带来的偏置。
        """
        def get_leaf_paths(tree, prefix="") -> List[str]:
            paths = []
            for node, subtree in tree.items():
                path = f"{prefix}/{node}" if prefix else node
                if not subtree:
                    paths.append(path)
                else:
                    paths.extend(get_leaf_paths(subtree, path))
            return paths
            
        active_paths = get_leaf_paths(self.taxonomy)
        if "General" not in active_paths:
            active_paths.append("General")
            
        new_soft_links = {}
        for path in active_paths:
            new_soft_links[path] = []
            # 提取类目名字的最后一个分段
            cat_name = path.split("/")[-1]
            
            # 为了提高语义匹配率，对预设的关键类目名称进行扩展
            translation_map = {
                "Hair": "发型发色双马尾长发短发", 
                "Clothing": "服饰制服鞋裙水手服百褶裙袜子", 
                "Weapon": "冷兵器武器装备刀剑阐释者太刀", 
                "Scenery": "场景背景风景樱花落日蓝天"
            }
            query_text = translation_map.get(cat_name, cat_name)
            
            # 编码该类目的查询语义向量 (raw 模式)
            query_vector = await self.embedder.get_embedding(query_text, "raw")
            
            # 计算与其他类目所有标签的相似度
            for other_path in active_paths:
                if other_path == path:
                    continue
                other_items = self.history.get(other_path, [])
                if not other_items:
                    continue
                    
                other_tags = [it["tag"] for it in other_items]
                # 重新以 raw 模式计算候选标签向量
                other_vectors = await self.embedder.get_embeddings_batch(other_tags, ["raw"] * len(other_tags))
                
                # 计算余弦相似度 (由于向量已做 L2 归一化，点积即为余弦值)
                dots = np.dot(other_vectors, query_vector)
                
                for idx, score in enumerate(dots):
                    if score >= similarity_threshold:
                        tag = other_tags[idx]
                        new_soft_links[path].append(tag)
                        
        self.logical_soft_links = new_soft_links
        return new_soft_links



