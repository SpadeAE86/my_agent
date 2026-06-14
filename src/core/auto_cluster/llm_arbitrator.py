import os
import json
from typing import List, Optional, Dict, Any
from utils.call_model_utils import call_doubao_seedtext

class LLMArbitrator:
    """
    负责利用大语言模型（LLM）进行高层次的语义裁决。
    包括：命名聚类簇、判断两组标签是否应当合并、评估新概念在知识图谱本体树中的生长方向（垂直 vs 水平）。
    支持在无 API Key 或调用失败时的本地 Heuristic 兜底算法。
    """
    def __init__(self):
        # 检查系统环境变量，判断 API Key 是否可用
        self.api_available = bool(os.getenv("ARK_API_KEY"))

    async def name_cluster(self, tags: List[str], medoid: Optional[str] = None) -> str:
        """
        利用 LLM 对给定的聚类簇成员进行语义归纳，提炼出一个 2-6 字的精炼中文概念分类名称。
        """
        if not tags:
            return "未命名概念"
            
        sample_tags = tags[:15]
        
        if self.api_available:
            system_prompt = (
                "你是一个专业的二次元插画标签分析助手。\n"
                "根据给定的这一组标签，提炼它们的语义交集或核心主题，起一个极其精炼的中文概念分类名称（2-6个字，例如'刀剑神域'、'金发发型'、'学院风服饰'、'自然风景'）。\n"
                "注意：只输出提炼出的中文名称，禁止包含任何标点符号、解释、备注、前缀或后缀。"
            )
            prompt = f"标签列表: {', '.join(sample_tags)}"
            if medoid:
                prompt += f"\n中心词提示: {medoid}"
                
            try:
                result = await call_doubao_seedtext(
                    prompt=prompt,
                    model="Seed 2.0 Pro",
                    system_prompt=system_prompt,
                    thinking=False
                )
                if result:
                    clean_res = result.strip().strip('"').strip("'")
                    if 0 < len(clean_res) <= 15:
                        return clean_res
            except Exception:
                pass

        base_name = medoid if medoid else tags[0]
        return f"{base_name}相关概念"

    async def decide_merge(self, cluster_a_tags: List[str], cluster_b_tags: List[str]) -> bool:
        """
        利用 LLM 判断两组标签语义是否完全一致，需要合并。
        """
        if not cluster_a_tags or not cluster_b_tags:
            return False
            
        sample_a = cluster_a_tags[:8]
        sample_b = cluster_b_tags[:8]
        
        if self.api_available:
            system_prompt = (
                "你是一个专业的标签数据库重构裁判。\n"
                "请判断给定的 A 组和 B 组标签是否指向同一个具体的动漫人物、同一作品、或者同一具体的服饰发型等概念（例如 A:['雷电将军', '影'] 与 B:['雷神'] 指向同一角色，应当合并；而 A:['双马尾'] 与 B:['水手服'] 不是同一个东西，应当保持独立）。\n"
                "回答规范：如果认为应当合并，请输出 'MERGE'；如果应当保持独立，请输出 'KEEP'。只输出这一个单词，不要带任何解释。"
            )
            prompt = f"A组标签: {', '.join(sample_a)}\nB组标签: {', '.join(sample_b)}"
            
            try:
                result = await call_doubao_seedtext(
                    prompt=prompt,
                    model="Seed 2.0 Pro",
                    system_prompt=system_prompt,
                    thinking=False
                )
                if result:
                    clean_res = result.strip().upper()
                    if "MERGE" in clean_res:
                        return True
                    if "KEEP" in clean_res:
                        return False
            except Exception:
                pass

        intersection = set(cluster_a_tags) & set(cluster_b_tags)
        if len(intersection) >= 2:
            return True
        return False

    async def assess_ontology_growth(self, current_taxonomy: Dict[str, Any], target_tags: List[str]) -> Dict[str, Any]:
        """
        利用大模型评估新发掘的标签簇在现有分类本体树（Taxonomy Schema）中的生长方向。
        返回决策结构体：
        {
            "decision": "HORIZONTAL" | "VERTICAL" | "REJECT",
            "parent_path": str,  # 如 "Weapon" 或 "Weapon/Cold_Weapon"
            "new_node_name": str, # 新节点英文 Key，如 "Cold_Weapon"
            "alias": str,         # 中文别名，如 "冷兵器"
            "template": str       # 专用 Embedding 提示词模板
        }
        """
        sample_tags = target_tags[:10]
        
        if self.api_available:
            system_prompt = (
                "你是一个专门负责知识图谱与打标系统本体演化（Ontology Evolution）的专家。\n"
                "你的职责是分析一个新发掘出来的标签类簇，并对照当前系统已有的类别树拓扑，决定其在拓扑中的最佳生长位置与方向。\n"
                "【生长规则说明】：\n"
                "1. HORIZONTAL（水平生长）：该类簇是一个与现有所有顶级分类都无关的全新领域，需创建新的顶级分类。\n"
                "2. VERTICAL（垂直生长）：该类簇应该归属于现有的某个顶级或多级分类之下，成为其子节点。\n"
                "3. REJECT（不生长/保持原样）：词簇不够内聚，或不需要做本体演化。\n"
                "\n"
                "必须严格按照以下 JSON 格式输出，不要有任何 Markdown 包裹块，不要输出解释文字：\n"
                "{\n"
                "  \"decision\": \"HORIZONTAL\" | \"VERTICAL\" | \"REJECT\",\n"
                "  \"parent_path\": \"如果是VERTICAL，请给出父节点路径，例如 'Weapon' 或 'Weapon/Cold_Weapon'；如果是HORIZONTAL或REJECT则填空字符串\",\n"
                "  \"new_node_name\": \"新节点的英文Key，采用驼峰或下划线单数名词（例如 'Cold_Weapon'）\",\n"
                "  \"alias\": \"新节点的中文名称（例如 '冷兵器'）\",\n"
                "  \"template\": \"该子空间专属的 Embedding 引导词模板，必须包含 {tag}（例如 'A cold weapon, blade or sword: {tag}'）\"\n"
                "}"
            )
            
            prompt = (
                f"【当前本体分类树】:\n{json.dumps(current_taxonomy, ensure_ascii=False, indent=2)}\n\n"
                f"【新发现的标签簇】:\n{sample_tags}"
            )
            
            try:
                result = await call_doubao_seedtext(
                    prompt=prompt,
                    model="Seed 2.0 Pro",
                    system_prompt=system_prompt,
                    thinking=False
                )
                if result:
                    # 尝试解析 JSON
                    clean_res = result.strip()
                    # 过滤可能被 markdown ```json ``` 包裹的情况
                    if clean_res.startswith("```"):
                        lines = clean_res.split("\n")
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines[-1].startswith("```"):
                            lines = lines[:-1]
                        clean_res = "\n".join(lines).strip()
                        
                    data = json.loads(clean_res)
                    if "decision" in data and "new_node_name" in data:
                        return data
            except Exception as e:
                # 记录日志或直接滑入 Heuristic
                pass

        # ----------------------------------------------------------------------
        # Local Heuristic 兜底机制 (算法离线运行或模型报错时保证基础路由)
        # ----------------------------------------------------------------------
        # 简单规则匹配：如果包含武器词汇且当前存在 Weapon，则垂直生长
        has_weapon = any(t in ["太刀", "阐释者", "刀", "剑", "枪"] for t in sample_tags)
        if has_weapon and "Weapon" in current_taxonomy:
            return {
                "decision": "VERTICAL",
                "parent_path": "Weapon",
                "new_node_name": "Cold_Weapon",
                "alias": "冷兵器",
                "template": "A cold weapon, blade or sword: {tag}"
            }
        
        # 兜底默认为 REJECT
        return {
            "decision": "REJECT",
            "parent_path": "",
            "new_node_name": "",
            "alias": "",
            "template": "A tag description: {tag}"
        }

    async def analyze_taxonomy_restructure(
        self,
        current_taxonomy: Dict[str, Any],
        category_summaries: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        利用 LLM 定期研判并梳理已有的本体树结构。
        判断是否存在需要进行 SHIFT_DOWN (降级)、SHIFT_UP (升级)、CREATE_PARENT (归拢父节点)、MERGE_CATEGORIES (合并类目) 的结构调整。
        """
        if self.api_available:
            system_prompt = (
                "你是一个资深的标签数据库结构优化专家（Taxonomy Architect）。\n"
                "根据给定的【当前分类树】以及【各个分类路径中的代表性标签】，评估是否需要对分类树进行结构梳理与优化。\n"
                "【优化动作类型说明】：\n"
                "1. SHIFT_DOWN: 降级。把一个分类作为子分类移动到另一个分类下。例如，将 'Hair' (包含发型/发色) 降级移动到 'Character' 下，变为 'Character/Hair'。\n"
                "2. SHIFT_UP: 升级。把一个深层子分类提到更高层级或根节点。例如，把 'Weapon/Firearm' 升级为根节点的 'Firearm'。\n"
                "3. CREATE_PARENT: 归拢建立父节点。如果两个分类 (如 'Clothing', 'Hair') 属于同一性质，可新建一个父节点 (如 'Appearance' 别名 '外观')，然后将它们都移动到该父节点下。\n"
                "4. MERGE_CATEGORIES: 合并。如果两个分类具有高度相似的标签且语义重复，合并它们。例如，把 'Outfit' 合并到 'Clothing' 中。\n"
                "\n"
                "你只能对当前分类树中已有的路径提出优化，严禁虚构不存在的节点作为源路径。\n"
                "必须严格按照以下 JSON 列表格式输出，不要有任何 Markdown 包裹块，不要输出解释文字：\n"
                "[\n"
                "  {\n"
                "    \"action\": \"SHIFT_DOWN\" | \"SHIFT_UP\" | \"CREATE_PARENT\" | \"MERGE_CATEGORIES\",\n"
                "    \"source_path\": \"被调整的源路径，例如 'Hair' 或 'Weapon/Firearm'\",\n"
                "    \"target_parent_path\": \"SHIFT_DOWN或CREATE_PARENT时的父节点路径（如 'Character' 或 'Appearance'）；SHIFT_UP时移到根节点则填空字符串''；MERGE_CATEGORIES时不填\",\n"
                "    \"source_paths\": \"仅在CREATE_PARENT时使用，为一个字符串列表，表示需要被归拢到新父节点底下的所有子节点原路径，如 ['Clothing', 'Hair']\",\n"
                "    \"new_parent_name\": \"仅在CREATE_PARENT时使用，表示新父节点的英文Key（如 'Appearance'）\",\n"
                "    \"alias\": \"新类目的中文别名（如 '外观'）\",\n"
                "    \"target_path\": \"仅在MERGE_CATEGORIES时使用，表示合并的目标路径（如 'Clothing'）\",\n"
                "    \"reason\": \"调整原因的简短说明\"\n"
                "  }\n"
                "]"
            )
            
            prompt = (
                f"【当前分类树】:\n{json.dumps(current_taxonomy, ensure_ascii=False, indent=2)}\n\n"
                f"【各分类下的典型标签】:\n{json.dumps(category_summaries, ensure_ascii=False, indent=2)}"
            )
            
            try:
                result = await call_doubao_seedtext(
                    prompt=prompt,
                    model="Seed 2.0 Pro",
                    system_prompt=system_prompt,
                    thinking=False
                )
                if result:
                    clean_res = result.strip()
                    if clean_res.startswith("```"):
                        lines = clean_res.split("\n")
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines[-1].startswith("```"):
                            lines = lines[:-1]
                        clean_res = "\n".join(lines).strip()
                    actions = json.loads(clean_res)
                    if isinstance(actions, list):
                        return actions
            except Exception:
                pass

        # ----------------------------------------------------------------------
        # Heuristic 兜底机制 (离线测试用)
        # ----------------------------------------------------------------------
        # Heuristic 1: 如果同时存在 "Character" 和 "Hair"，且 "Hair" 在根节点，则推荐 SHIFT_DOWN 把 Hair 归于 Character 之下
        if "Character" in current_taxonomy and "Hair" in current_taxonomy:
            return [
                {
                    "action": "SHIFT_DOWN",
                    "source_path": "Hair",
                    "target_parent_path": "Character",
                    "reason": "发型发色标签应归属于人物特征的子目下"
                }
            ]
            
        return []

