from typing import Dict, Any, List, Optional, Type
from models.pydantic.opensearch_index.base_index import (
    BaseIndex,
    get_vector_fields,
    get_text_fields,
    get_searchable_fields,
    get_index_name
)
from sentence_transformers import SentenceTransformer
from infra.storage.opensearch.create_index import index_manager


class QueryBuilder:
    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        self.embedding_model = embedding_model or SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    async def get_index_field_types(self, model_class: Type[BaseIndex]) -> Optional[Dict[str, Any]]:
        index_name = get_index_name(model_class)
        index_config = await index_manager.get_index_config(index_name)
        
        if index_config and 'mappings' in index_config:
            return index_config['mappings'].get('properties', {})
        
        return None
    
    def build_hybrid_search(
        self,
        model_class: Type[BaseIndex],
        query: str,
        size: int = 10,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        search_fields: Optional[List[str]] = None,
        vector_field: Optional[str] = None,
        field_boosts: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        vector_fields = get_vector_fields(model_class)
        if not vector_fields:
            raise ValueError(f"No vector fields found in model {model_class.__name__}")
        
        if vector_field is None:
            vector_field = vector_fields[0]
        elif vector_field not in vector_fields:
            raise ValueError(f"Vector field '{vector_field}' not found. Available: {vector_fields}")
        
        if search_fields is None:
            search_fields = get_searchable_fields(model_class)
        
        query_vector = self._generate_embedding(query)
        
        queries = []
        
        for field_name in search_fields:
            boost = field_boosts.get(field_name, 1.0) if field_boosts else 1.0
            queries.append({
                "match": {
                    field_name: {
                        "query": query,
                        "boost": bm25_weight * boost
                    }
                }
            })
        
        queries.append({
            "knn": {
                vector_field: {
                    "vector": query_vector,
                    "k": size
                },
                "boost": vector_weight
            }
        })
        
        search_body = {
            "size": size,
            "query": {
                "hybrid": {
                    "queries": queries
                }
            },
            "_source": {
                "exclude": vector_fields
            }
        }
        
        return search_body
    
    def build_dynamic_hybrid_search(
        self,
        model_class: Type[BaseIndex],
        query: str,
        size: int = 10,
        bm25_factor: float = 0.5,
        vector_factor: float = 0.5,
        field_weight_overrides: Optional[Dict[str, float]] = None,
        vector_weight_overrides: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Constructs a hybrid search query dynamically based on Pydantic field metadata (json_schema_extra).
        """
        vector_fields = get_vector_fields(model_class)
        if not vector_fields:
            raise ValueError(f"No vector fields found in model {model_class.__name__}")
            
        search_fields = get_searchable_fields(model_class)
        
        field_weight_overrides = field_weight_overrides or {}
        vector_weight_overrides = vector_weight_overrides or {}
        
        # 1. Resolve Text Field Weights
        weighted_text_fields = []
        for field_name in search_fields:
            field_info = model_class.model_fields.get(field_name)
            
            # Priority: Override -> json_schema_extra -> Default (1.0)
            weight = 1.0
            if field_name in field_weight_overrides:
                weight = field_weight_overrides[field_name]
            elif field_info and field_info.json_schema_extra and "search_weight" in field_info.json_schema_extra:
                weight = float(field_info.json_schema_extra["search_weight"])
                
            weighted_text_fields.append(f"{field_name}^{weight}")
            
        # 2. Resolve Vector Field Weights
        query_vector = self._generate_embedding(query)
        vector_queries = []
        
        for field_name in vector_fields:
            field_info = model_class.model_fields.get(field_name)
            
            # Priority: Override -> json_schema_extra -> Default (1.0)
            weight = 1.0
            if field_name in vector_weight_overrides:
                weight = vector_weight_overrides[field_name]
            elif field_info and field_info.json_schema_extra and "vector_weight" in field_info.json_schema_extra:
                weight = float(field_info.json_schema_extra["vector_weight"])
                
            vector_queries.append({
                "knn": {
                    field_name: {
                        "vector": query_vector,
                        "k": size
                    },
                    "boost": weight * vector_factor
                }
            })
            
        # 3. Build Query
        queries = []
        if weighted_text_fields:
            queries.append({
                "multi_match": {
                    "query": query,
                    "fields": weighted_text_fields,
                    "type": "best_fields",
                    "boost": bm25_factor
                }
            })
            
        queries.extend(vector_queries)
        
        search_body = {
            "size": size,
            "query": {
                "hybrid": {
                    "queries": queries
                }
            },
            "_source": {
                "exclude": vector_fields
            }
        }
        
        return search_body
    
    def build_semantic_search(
        self,
        model_class: Type[BaseIndex],
        query: str,
        size: int = 10,
        vector_field: Optional[str] = None
    ) -> Dict[str, Any]:
        vector_fields = get_vector_fields(model_class)
        if not vector_fields:
            raise ValueError(f"No vector fields found in model {model_class.__name__}")
        
        if vector_field is None:
            vector_field = vector_fields[0]
        elif vector_field not in vector_fields:
            raise ValueError(f"Vector field '{vector_field}' not found. Available: {vector_fields}")
        
        query_vector = self._generate_embedding(query)
        
        search_body = {
            "size": size,
            "query": {
                "knn": {
                    vector_field: {
                        "vector": query_vector,
                        "k": size
                    }
                }
            },
            "_source": {
                "exclude": vector_fields
            }
        }
        
        return search_body
    
    def build_keyword_search(
        self,
        model_class: Type[BaseIndex],
        query: str,
        size: int = 10,
        search_fields: Optional[List[str]] = None,
        field_boosts: Optional[Dict[str, float]] = None,
        query_type: str = "best_fields"
    ) -> Dict[str, Any]:
        vector_fields = get_vector_fields(model_class)
        
        if search_fields is None:
            search_fields = get_searchable_fields(model_class)
        
        if field_boosts:
            fields = []
            for field_name in search_fields:
                boost = field_boosts.get(field_name, 1.0)
                fields.append(f"{field_name}^{boost}")
        else:
            fields = search_fields
        
        search_body = {
            "size": size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "type": query_type
                }
            },
            "_source": {
                "exclude": vector_fields
            }
        }
        
        return search_body
    
    def build_filter_search(
        self,
        model_class: Type[BaseIndex],
        query: str,
        filters: Dict[str, Any],
        size: int = 10,
        search_fields: Optional[List[str]] = None,
        field_boosts: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        vector_fields = get_vector_fields(model_class)
        
        if search_fields is None:
            search_fields = get_searchable_fields(model_class)
        
        if field_boosts:
            fields = []
            for field_name in search_fields:
                boost = field_boosts.get(field_name, 1.0)
                fields.append(f"{field_name}^{boost}")
        else:
            fields = search_fields
        
        search_body = {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": fields,
                                "type": "best_fields"
                            }
                        }
                    ],
                    "filter": []
                }
            },
            "_source": {
                "exclude": vector_fields
            }
        }
        
        for field, filter_config in filters.items():
            if isinstance(filter_config, dict):
                if 'gte' in filter_config or 'lte' in filter_config or 'gt' in filter_config or 'lt' in filter_config:
                    search_body["query"]["bool"]["filter"].append({
                        "range": {field: filter_config}
                    })
                else:
                    search_body["query"]["bool"]["filter"].append({
                        "term": {field: filter_config}
                    })
            elif isinstance(filter_config, list):
                search_body["query"]["bool"]["filter"].append({
                    "terms": {field: filter_config}
                })
            else:
                search_body["query"]["bool"]["filter"].append({
                    "term": {field: filter_config}
                })
        
        return search_body
    
    def build_multi_vector_search(
        self,
        model_class: Type[BaseIndex],
        query: str,
        size: int = 10,
        vector_fields: Optional[List[str]] = None,
        vector_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        all_vector_fields = get_vector_fields(model_class)
        if not all_vector_fields:
            raise ValueError(f"No vector fields found in model {model_class.__name__}")
        
        vector_fields = vector_fields or all_vector_fields
        query_vector = self._generate_embedding(query)
        
        queries = []
        
        for field_name in vector_fields:
            weight = vector_weights.get(field_name, 1.0) if vector_weights else 1.0
            queries.append({
                "knn": {
                    field_name: {
                        "vector": query_vector,
                        "k": size
                    },
                    "boost": weight
                }
            })
        
        search_body = {
            "size": size,
            "query": {
                "hybrid": {
                    "queries": queries
                }
            },
            "_source": {
                "exclude": all_vector_fields
            }
        }
        
        return search_body
    
    def build_rrf_search(
        self,
        model_class: Type[BaseIndex],
        query: str,
        size: int = 10,
        search_fields: Optional[List[str]] = None,
        vector_field: Optional[str] = None,
        k: int = 60
    ) -> Dict[str, Any]:
        vector_fields = get_vector_fields(model_class)
        if not vector_fields:
            raise ValueError(f"No vector fields found in model {model_class.__name__}")
        
        if vector_field is None:
            vector_field = vector_fields[0]
        elif vector_field not in vector_fields:
            raise ValueError(f"Vector field '{vector_field}' not found. Available: {vector_fields}")
        
        if search_fields is None:
            search_fields = get_searchable_fields(model_class)
        
        query_vector = self._generate_embedding(query)
        
        search_body = {
            "size": size,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": search_fields,
                                "type": "best_fields"
                            }
                        },
                        {
                            "knn": {
                                vector_field: {
                                    "vector": query_vector,
                                    "k": size
                                }
                            }
                        }
                    ]
                }
            },
            "rrf": {
                "rank_constant": k,
                "window_size": size
            },
            "_source": {
                "exclude": vector_fields
            }
        }
        
        return search_body
    
    def _generate_embedding(self, text: str) -> List[float]:
        if not text:
            return [0.0] * 384
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def update_embedding_model(self, model: SentenceTransformer):
        self.embedding_model = model


query_builder = QueryBuilder()
