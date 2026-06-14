import asyncio
import sys
import os
import json
import pytest
# 将 src 目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.auto_cluster import AutoClusterOrchestrator

# ============================================================
# 测试控制参数
# ============================================================
# REINDEX = True: 每次运行清空缓存，重新进行完整的打标和重构（常用于全新验证）
# REINDEX = False: 基于上一次运行保存的状态（auto_cluster_state.pkl）进行增量学习和追加
REINDEX = True 

@pytest.mark.asyncio
async def test_auto_cluster_pipeline():
    print("="*80)
    print("【测试】自组织语义空间系统 core/auto_cluster 全流程验证 (Elasticsearch 版)")
    print(f"模式: {'[REINDEX = True] (清空重建模式)' if REINDEX else '[REINDEX = False] (增量追加入库模式)'}")
    print("="*80)
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    tags_file = os.path.join(test_dir, "mock_tags.json")
    state_file = os.path.join(test_dir, "auto_cluster_state.pkl")
    
    # 1. 读取独立的测试标签列表文件
    if not os.path.exists(tags_file):
        print(f"[错误] 未找到测试标签文件: {tags_file}")
        return
        
    with open(tags_file, "r", encoding="utf-8") as f:
        tags_stream = json.load(f)
        
    print(f"成功载入 {len(tags_stream)} 个测试标签，来源: mock_tags.json")
    
    # 2. 初始化 Orchestrator 并加载状态
    orchestrator = AutoClusterOrchestrator(sim_threshold=0.65, decay_factor=0.005)
    
    # 根据 REINDEX 决定是否从文件热启动
    loaded = False
    if not REINDEX:
        if os.path.exists(state_file):
            print(f"正在从本地恢复上一次的状态: {state_file}")
            loaded = orchestrator.load_state(state_file)
            if loaded:
                print("状态恢复成功！已读入已有本体拓扑和微簇。")
        else:
            print("未找到历史状态文件，将自动开启全新初始化。")
            
    if not loaded:
        # 全新初始化时，我们在本体树中预置 "Weapon/Firearm" 分类以供垂直生长测试
        print("全新初始化：清空历史，注入初始本体树拓扑。")
        if os.path.exists(state_file):
            try:
                os.remove(state_file)
            except Exception:
                pass
        orchestrator.taxonomy["Weapon"] = {"Firearm": {}}
        
    print("\n[当前本体树拓扑状态]:")
    print(json.dumps(orchestrator.taxonomy, ensure_ascii=False, indent=2))
    
    # 3. 开始流式处理标签
    print("\n--- 1. 开始流式处理标签 (Routing + Embedding + L2 Norm + Online Cluster) ---")
    for step, tag in enumerate(tags_stream):
        res = await orchestrator.process_tag(tag)
        print(f"[Step {step:02d}] 标签: \"{res['tag']:<8}\" | 路由路径: {res['category_path']:<15} | "
              f"簇ID: {res['cluster_id']:<2} | 是否稳定: {str(res['is_stable']):<5} | "
              f"分配概念: {res['concept_name']}")
              
    # 4. 触发本体生长演化
    print("\n--- 2. 触发 General 空间本体生长演化 (LLM 裁判生长方向) ---")
    growth_reports = await orchestrator.evolve_general_ontology()
    
    if not growth_reports:
        print("  (未发生本体维度生长或被 LLM 拒绝)")
    else:
        for idx, report in enumerate(growth_reports):
            print(f"  * 发现生长点 #{idx:02d}:")
            print(f"    - 生长决策: {report['decision']}")
            print(f"    - 生长本体路径 (Path): \"{report['full_path']}\" (别名: {report['alias']})")
            print(f"    - 专属向量化模板: \"{report['template']}\"")
            print(f"    - 迁移/收敛的历史标签: {report['tags_migrated']}")
            
        print("\n[生长进化后的新本体树状态]:")
        print(json.dumps(orchestrator.taxonomy, ensure_ascii=False, indent=2))

    # 4.5. 触发本体结构优化重构 (SHIFT_DOWN / CREATE_PARENT / SHIFT_UP / MERGE_CATEGORIES)
    print("\n--- 2.5. 触发本体结构演化梳理与优化 (Restructuring & Optimization) ---")
    restructure_reports = await orchestrator.evolve_taxonomy_structure()
    if not restructure_reports:
        print("  (未发生本体结构重构)")
    else:
        for idx, report in enumerate(restructure_reports):
            print(f"  * 结构重构动作 #{idx:02d}:")
            print(f"    - 类型: {report['action']}")
            if report['action'] == "CREATE_PARENT":
                print(f"    - 新建父节点: {report['new_parent']} (别名: {report['alias']})")
                print(f"    - 迁移子分类: {report['moved_sources']}")
            else:
                print(f"    - 目标路径: {report.get('new_path') or report.get('target_path')}")
                print(f"    - 原始路径: {report['source_path']}")
            print(f"    - 原因: {report['reason']}")
            
        print("\n[结构重构后的最终本体树拓扑]:")
        print(json.dumps(orchestrator.taxonomy, ensure_ascii=False, indent=2))
        
        # 验证重定向是否生效
        print("\n--- 验证重定向与动态重新向量化 ---")
        test_new_tag = "长发"
        res_tag = await orchestrator.process_tag(test_new_tag)
        print(f"  * 测试新标签: \"{test_new_tag}\" -> 路由结果路径: {res_tag['category_path']}")
        
        # 检查迁移后历史标签的数据状态（是否正确更新并重算向量）
        def get_leaf_paths_temp(tree, current_prefix="") -> list:
            paths = []
            for node, subtree in tree.items():
                path = f"{current_prefix}/{node}" if current_prefix else node
                if not subtree:
                    paths.append(path)
                else:
                    paths.extend(get_leaf_paths_temp(subtree, path))
            return paths
        active_leafs = get_leaf_paths_temp(orchestrator.taxonomy)
        for leaf in active_leafs:
            history_items = orchestrator.history.get(leaf, [])
            if history_items:
                print(f"  * 分类 \"{leaf}\" 拥有历史标签数量: {len(history_items)}，首个标签: \"{history_items[0]['tag']}\"，向量维度: {len(history_items[0]['vector'])}")
        
        # 验证逻辑软链接自动发掘 (Soft link discovery)
        print("\n--- 2.6. 触发跨类目的逻辑“软链接”自动发掘 (Logical Soft Links Discovery) ---")
        links = await orchestrator.discover_soft_links(similarity_threshold=0.45)

        print("  * 成功发掘出以下类目软链接:")
        for path, soft_tags in links.items():
            if soft_tags:
                print(f"    - 类目 \"{path}\" 的软链接词: {soft_tags}")
                
    # 5. 运行离线重构 (Offline Reshaping)
    print("\n--- 3. 触发所有空间类目的离线重构 (Offline Reshaping + Medoids) ---")
    
    def get_leaf_paths(tree, current_prefix="") -> list:
        paths = []
        for node, subtree in tree.items():
            path = f"{current_prefix}/{node}" if current_prefix else node
            if not subtree:
                paths.append(path)
            else:
                paths.extend(get_leaf_paths(subtree, path))
        return paths
        
    all_paths = get_leaf_paths(orchestrator.taxonomy)
    all_paths.append("General")
    all_paths = sorted(list(set(all_paths)))
    
    for path in all_paths:
        print(f"\n==================== 重构类别空间路径: {path} ====================")
        res_reshape = await orchestrator.run_offline_reshape(path)
        
        if res_reshape["status"] == "empty":
            print("  (该空间暂无标签数据)")
            continue
            
        print(f"标签总数: {res_reshape['total_tags']} | 初始簇数量: {res_reshape['original_clusters_count']}")
        
        print("\n【 最终沉淀的核心概念 (Stable Concepts) 】:")
        if not res_reshape["final_concepts"]:
            print("  (无稳定核心概念)")
        for concept in res_reshape["final_concepts"]:
            print(f"  * 概念 ID: {concept['concept_id']}")
            print(f"    - 概念名称: \"{concept['concept_name']}\"")
            print(f"    - 语义中心词 (Medoid): \"{concept['medoid']}\"")
            print(f"    - 包含的标签: {concept['tags']}")
            print("      (以后可批量同步/写入 Elasticsearch 向量字段与元数据字段)")
            
        print("\n【 识别出的孤立噪声 (Noise / Outliers) 】:")
        if not res_reshape["noise_tags"]:
            print("  (无噪声标签)")
        else:
            print(f"  * 噪声词列表: {res_reshape['noise_tags']}")
            
    # 6. 保存状态以供下一次增量运行
    orchestrator.save_state(state_file)
    print(f"\n[状态存盘] 成功将当前系统状态保存至: {state_file}")
    
    print("\n" + "="*80)
    print("流式增量聚类与离线重构系统 core/auto_cluster 测试结束。 (Elasticsearch 版)")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_auto_cluster_pipeline())
