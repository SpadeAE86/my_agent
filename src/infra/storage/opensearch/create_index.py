from typing import Dict, Any, Optional, Type
from models.pydantic.opensearch_index.base_index import (
    BaseIndex,
    get_index_name,
    build_field_types_from_markers,
)
from infra.storage.opensearch_connector import opensearch_connector
from infra.logging.logger import logger as log


class IndexManager:
    def __init__(self):
        self.connector = opensearch_connector
    
    async def get_client(self):
        await self.connector.ensure_init()
        return await self.connector.get_client()
    
    async def get_index_config(self, index_name: str) -> Optional[Dict[str, Any]]:
        client = await self.get_client()
        try:
            response = await client.indices.get(index=index_name)
            if index_name in response:
                return response[index_name]
            return None
        except Exception as e:
            log.error(f"Failed to get index config for {index_name}: {e}")
            return None
    
    async def index_exists(self, index_name: str) -> bool:
        client = await self.get_client()
        try:
            return await client.indices.exists(index=index_name)
        except Exception as e:
            log.error(f"Failed to check index existence for {index_name}: {e}")
            return False
    
    async def create_index(
        self,
        model_class: Type[BaseIndex],
        field_types: Optional[Dict[str, Dict[str, Any]]] = None,
        settings: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> bool:
        index_name = get_index_name(model_class)
        client = await self.get_client()
        
        # If caller didn't provide field_types, try to derive from Annotated markers.
        if field_types is None:
            field_types = build_field_types_from_markers(model_class)
            if not field_types:
                raise ValueError(
                    f"No field_types provided and no markers found on model {model_class.__name__}. "
                    f"Please pass field_types explicitly or annotate fields with markers."
                )
        
        try:
            exists = await self.index_exists(index_name)
            
            if exists:
                if overwrite:
                    log.info(f"Index {index_name} exists, overwriting...")
                    await self.delete_index(index_name)
                else:
                    log.info(f"Index {index_name} already exists, skipping creation")
                    return False
            
            # Resolve settings in priority order:
            # 1) explicit `settings` arg
            # 2) model_class.Meta.settings override
            # 3) global defaults
            meta_settings = getattr(getattr(model_class, "Meta", None), "settings", None)
            resolved_settings = settings or meta_settings or {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }

            index_body = {
                "settings": resolved_settings,
                "mappings": {"properties": {}},
            }
            
            for field_name in model_class.model_fields.keys():
                if field_name in field_types:
                    index_body["mappings"]["properties"][field_name] = field_types[field_name]
            
            await client.indices.create(index=index_name, body=index_body)
            log.info(f"Created index: {index_name}")
            return True
            
        except Exception as e:
            log.error(f"Failed to create index {index_name}: {e}")
            raise
    
    async def delete_index(self, index_name: str) -> bool:
        client = await self.get_client()
        try:
            await client.indices.delete(index=index_name)
            log.info(f"Deleted index: {index_name}")
            return True
        except Exception as e:
            log.error(f"Failed to delete index {index_name}: {e}")
            raise
    
    async def update_index_settings(self, index_name: str, settings: Dict[str, Any]) -> bool:
        client = await self.get_client()
        try:
            await client.indices.put_settings(index=index_name, body=settings)
            log.info(f"Updated settings for index: {index_name}")
            return True
        except Exception as e:
            log.error(f"Failed to update settings for index {index_name}: {e}")
            raise
    
    async def update_index_mapping(self, index_name: str, properties: Dict[str, Any]) -> bool:
        client = await self.get_client()
        try:
            await client.indices.put_mapping(index=index_name, body={"properties": properties})
            log.info(f"Updated mapping for index: {index_name}")
            return True
        except Exception as e:
            log.error(f"Failed to update mapping for index {index_name}: {e}")
            raise
    
    async def list_indices(self) -> list:
        client = await self.get_client()
        try:
            response = await client.cat.indices(format="json")
            return [idx["index"] for idx in response]
        except Exception as e:
            log.error(f"Failed to list indices: {e}")
            raise
    
    async def get_index_stats(self, index_name: str) -> Optional[Dict[str, Any]]:
        client = await self.get_client()
        try:
            response = await client.indices.stats(index=index_name)
            if index_name in response["indices"]:
                return response["indices"][index_name]
            return None
        except Exception as e:
            log.error(f"Failed to get stats for index {index_name}: {e}")
            return None


index_manager = IndexManager()

