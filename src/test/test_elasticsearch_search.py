# -*- coding: utf-8 -*-
import asyncio
import os
import sys

# Ensure we can import from src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Set env to local
os.environ["env"] = "local"

from config.config import MY_CONFIG
MY_CONFIG["search_provider"] = "elasticsearch"

from infra.storage.elasticsearch_connector import elasticsearch_connector
from infra.storage.elasticsearch.search_coordinator import search_cards_es, match_script_tags_segments_es
from routers.video_analysis.search import VideoAnalysisSearchRequest


async def main():
    print("Initializing Elasticsearch connector...")
    await elasticsearch_connector.ensure_init()
    
    # Check if index exists
    client = await elasticsearch_connector.get_client()
    exists = await client.indices.exists(index="car_interior_analysis_v2")
    print(f"Index 'car_interior_analysis_v2' exists: {exists}")
    
    if not exists:
        print("Error: Index does not exist. Please run build_elasticsearch_index_v2 first.")
        await elasticsearch_connector.close()
        return

    # Let's perform a basic keyword search
    print("\n--- Testing Keyword Search (Precise) ---")
    req = VideoAnalysisSearchRequest(
        tokens=[{"text": "静态展示", "join": "AND", "source_field": "movement"}],
        fuzzy=False,
        size=5,
        workspace="v2"
    )
    
    hits, search_mode = await search_cards_es(
        query_text="智己LS6",
        req=req,
        term_filters=[{"term": {"movement": "静态展示"}}],
        should_boosts=[],
        must_not_multi_matches=[],
        size=5
    )
    print(f"Search mode: {search_mode}")
    print(f"Returned hits: {len(hits)}")
    for i, h in enumerate(hits):
        print(f"  [{i+1}] ID={h.get('_id')} Score={h.get('_score')}")

    # Let's perform a fuzzy RRF search
    print("\n--- Testing Fuzzy RRF Search ---")
    req_fuzzy = VideoAnalysisSearchRequest(
        tokens=[{"text": "智己LS6", "join": "AND", "source_field": "car_model"}],
        fuzzy=True,
        use_rrf=True,
        size=5,
        workspace="v2"
    )
    
    hits_f, search_mode_f = await search_cards_es(
        query_text="屏幕 特写 画面",
        req=req_fuzzy,
        term_filters=[{"term": {"car_model": "智己LS6"}}],
        should_boosts=[],
        must_not_multi_matches=[],
        size=5
    )
    print(f"Search mode: {search_mode_f}")
    print(f"Returned hits: {len(hits_f)}")
    for i, h in enumerate(hits_f):
        print(f"  [{i+1}] ID={h.get('_id')} Score={h.get('_score')}")

    # Let's perform a segment matching query
    print("\n--- Testing Segment Matching Coordinator ---")
    segments = [
        {
            "id": 1,
            "segment_text": "智己LS6的中控屏展示",
            "duration": 5.0,
            "car_model": "智己LS6",
            "movement": "静态展示"
        }
    ]
    
    match_results = await match_script_tags_segments_es(
        segments,
        top_k=3,
        mode="field_aligned_hybrid",
        shot_cards_version="v2"
    )
    
    print(f"Segment match results: {len(match_results)}")
    for r in match_results:
        print(f"Segment ID: {r.get('segment_id')}")
        print(f"Query text: {r.get('query_text')}")
        print(f"Top hits count: {len(r.get('top_hits', []))}")
        for i, hit in enumerate(r.get("top_hits", [])):
            print(f"  Hit [{i+1}] ID: {hit.get('_id')} Score: {hit.get('_score')} Path: {hit.get('video_path')}")
        print(f"Filled hits count: {len(r.get('filled_hits', []))}")
        for i, hit in enumerate(r.get("filled_hits", [])):
            print(f"  Filled [{i+1}] ID: {hit.get('_id')} Role: {hit.get('role')} Path: {hit.get('video_path')}")

    await elasticsearch_connector.close()


if __name__ == "__main__":
    asyncio.run(main())
