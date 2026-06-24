import asyncio


async def main():
    # Ensure we use the same connector lifecycle as FastAPI
    from infra.connector_loader import connector_loader
    from infra.storage.opensearch_connector import opensearch_connector

    await connector_loader.startup()
    try:
        client = await opensearch_connector.get_client()
        index = "car_interior_analysis"

        exists = await client.indices.exists(index=index)
        print(f"index_exists: {exists} ({index})")
        if not exists:
            return

        # Count docs
        resp = await client.count(index=index, body={"query": {"match_all": {}}})
        print(f"doc_count: {resp.get('count')}")

        # Show a sample doc (if any)
        if (resp.get("count") or 0) > 0:
            sample = await client.search(
                index=index,
                body={"size": 1, "query": {"match_all": {}}, "_source": {"exclude": ["*vector*"]}},
            )
            hits = (sample.get("hits") or {}).get("hits") or []
            if hits:
                print("sample_doc_id:", hits[0].get("_id"))
                print("sample_doc_source:", hits[0].get("_source"))
    finally:
        await connector_loader.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

