from typing import Dict, Any, List, Optional, Type, Tuple, Any as AnyType
from models.elasticsearch_index.base_index import (
    BaseIndex,
    get_vector_fields,
    get_searchable_fields,
    get_field_weights,
    get_vector_weights,
    get_index_name
)

_UNKNOWN = "未知"


def _text_for_bm25(v: Any) -> str:
    if isinstance(v, list):
        return " ".join([s for x in v for s in [str(x).strip()] if s and s != _UNKNOWN])
    if isinstance(v, str):
        t = v.strip()
        return "" if t == _UNKNOWN else t
    return ""


_FIELD_ALIGN_TEXT_MAP: List[Tuple[str, str, float]] = [
    ("marketing_phrases",       "marketing_phrases",       1.5),
    ("function_selling_points", "function_selling_points", 1.3),
    ("design_selling_points",   "design_selling_points",   1.3),
    ("description",             "description",             1.2),
    ("scene_location",          "scene_location",          1.1),
    ("scenario_a",              "scenario_a",              1.1),
    ("scenario_b",              "scenario_b",              1.0),
    ("segment_text",            "description",             1.0),
    ("subject",                 "subject",                 1.0),
    ("object",                  "object",                  0.9),
    ("marketing_tags",          "marketing_phrases",       0.8),
]

_FIELD_ALIGN_BROADCAST: List[Tuple[str, float]] = [
    ("extra_tags", 0.7),
]

_BROADCAST_INDEX_TEXT_FIELDS: List[str] = [
    "marketing_phrases",
    "function_selling_points",
    "design_selling_points",
    "description",
    "scene_location",
    "scenario_a",
    "scenario_b",
    "subject",
    "object",
]


class QueryBuilder:
    def __init__(self, embedding_model: Optional[AnyType] = None):
        self.embedding_model = embedding_model
    
    def _generate_embedding(self, text: str) -> List[float]:
        if not text:
            return [0.0] * 384
        if self.embedding_model is None:
            from services.analysis_video import get_embedding_model
            self.embedding_model = get_embedding_model()
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def update_embedding_model(self, model: AnyType):
        self.embedding_model = model

    def build_es_field_aligned_bm25_query(
        self,
        seg: Dict[str, Any],
        *,
        size: int,
    ) -> Dict[str, Any]:
        """
        Build a field-aligned BM25 query body for ES.
        """
        should_clauses: List[Dict[str, Any]] = []

        # --- Phase 1: per-field aligned match ---
        for seg_field, index_field, boost in _FIELD_ALIGN_TEXT_MAP:
            text = _text_for_bm25(seg.get(seg_field))
            if not text:
                continue
            should_clauses.append(
                {"match": {index_field: {"query": text, "boost": float(boost)}}}
            )

        # --- Phase 2: broadcast catch-all fields ---
        for seg_field, boost in _FIELD_ALIGN_BROADCAST:
            text = _text_for_bm25(seg.get(seg_field))
            if not text:
                continue
            should_clauses.append(
                {
                    "multi_match": {
                        "query": text,
                        "fields": _BROADCAST_INDEX_TEXT_FIELDS,
                        "type": "best_fields",
                        "boost": float(boost),
                    }
                }
            )

        query: Dict[str, Any] = (
            {"bool": {"should": should_clauses, "minimum_should_match": 1}}
            if should_clauses
            else {"match_all": {}}
        )

        return query

    def build_es_keyword_query(
        self,
        model_class: Type[BaseIndex],
        query: str,
        size: int = 10,
        search_fields: Optional[List[str]] = None,
        field_boosts: Optional[Dict[str, float]] = None,
        filters: Optional[List[Dict[str, Any]]] = None,
        should_boosts: Optional[List[Dict[str, Any]]] = None,
        must_nots: Optional[List[Dict[str, Any]]] = None,
        explain: bool = False,
    ) -> Dict[str, Any]:
        """
        Build a standard Elasticsearch BM25 keyword query.
        """
        vector_fields = get_vector_fields(model_class)
        if search_fields is None:
            search_fields = get_searchable_fields(model_class)
        
        fields = []
        for field_name in search_fields:
            boost = 1.0
            if field_boosts and field_name in field_boosts:
                boost = field_boosts[field_name]
            fields.append(f"{field_name}^{boost}")
            
        multi_match_clause = {
            "multi_match": {
                "query": query,
                "fields": fields,
                "type": "best_fields",
                "_name": "bm25_text_match"
            }
        }
        
        bool_query: Dict[str, Any] = {
            "must": [multi_match_clause]
        }
        
        if filters:
            bool_query["filter"] = filters
        if should_boosts:
            bool_query["should"] = should_boosts
            bool_query["minimum_should_match"] = 0
        if must_nots:
            bool_query["must_not"] = must_nots
            
        body: Dict[str, Any] = {
            "size": size,
            "query": {
                "bool": bool_query
            },
            "_source": {
                "excludes": vector_fields
            }
        }
        if explain:
            body["explain"] = True
            
        return body

    def build_es_rrf_query(
        self,
        model_class: Type[BaseIndex],
        query: str,
        size: int = 10,
        search_fields: Optional[List[str]] = None,
        vector_fields: Optional[List[str]] = None,
        query_vector: Optional[List[float]] = None,
        field_weight_overrides: Optional[Dict[str, float]] = None,
        vector_weight_overrides: Optional[Dict[str, float]] = None,
        filters: Optional[List[Dict[str, Any]]] = None,
        should_boosts: Optional[List[Dict[str, Any]]] = None,
        must_nots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build an Elasticsearch RRF retriever query.
        """
        all_vector_fields = get_vector_fields(model_class)
        all_search_fields = get_searchable_fields(model_class)
        
        search_fields = search_fields or all_search_fields
        vector_fields = vector_fields or all_vector_fields
        vector_fields = [vf for vf in vector_fields if vf in all_vector_fields]
        
        # 1. Standard text retriever
        text_weights = get_field_weights(model_class).copy()
        if field_weight_overrides:
            text_weights.update(field_weight_overrides)
            
        weighted_text_fields = []
        for field_name in search_fields:
            weight = text_weights.get(field_name, 1.0)
            weighted_text_fields.append(f"{field_name}^{weight}")
            
        text_multi_match = {
            "multi_match": {
                "query": query,
                "fields": weighted_text_fields,
                "type": "best_fields",
                "_name": "bm25_text_match"
            }
        }
        
        standard_bool: Dict[str, Any] = {"must": [text_multi_match]}
        if filters:
            standard_bool["filter"] = filters
        if should_boosts:
            standard_bool["should"] = should_boosts
            standard_bool["minimum_should_match"] = 0
        if must_nots:
            standard_bool["must_not"] = must_nots
            
        retrievers = [
            {
                "standard": {
                    "query": {
                        "bool": standard_bool
                    }
                }
            }
        ]
        
        # 2. KNN retrievers
        qv = query_vector if query_vector is not None else self._generate_embedding(query)
        vector_weights = get_vector_weights(model_class).copy()
        if vector_weight_overrides:
            vector_weights.update(vector_weight_overrides)
            
        for field_name in vector_fields:
            boost = float(vector_weights.get(field_name, 1.0))
            
            knn_clause: Dict[str, Any] = {
                "field": field_name,
                "query_vector": qv,
                "k": size,
                "num_candidates": max(size * 5, 100),
                "boost": boost
            }
            
            if filters or must_nots or should_boosts:
                knn_filter_bool: Dict[str, Any] = {}
                if filters:
                    knn_filter_bool["filter"] = filters
                if must_nots:
                    knn_filter_bool["must_not"] = must_nots
                if should_boosts:
                    knn_filter_bool["should"] = should_boosts
                    knn_filter_bool["minimum_should_match"] = 0
                knn_clause["filter"] = {"bool": knn_filter_bool}
                
            retrievers.append({
                "knn": knn_clause
            })
            
        body = {
            "size": size,
            "retriever": {
                "rrf": {
                    "retrievers": retrievers,
                    "rank_constant": 60,
                    "window_size": max(size * 3, 100)
                }
            },
            "_source": {
                "excludes": all_vector_fields
            }
        }
        
        return body

    def build_es_field_aligned_hybrid_query(
        self,
        seg: Dict[str, Any],
        size: int,
        vector_fields: List[str],
        query_vector: List[float],
        filters: Optional[List[Dict[str, Any]]] = None,
        should_boosts: Optional[List[Dict[str, Any]]] = None,
        must_nots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build an ES RRF query using field-aligned BM25 for the standard retriever.
        """
        all_vector_fields = get_vector_fields(CarInteriorAnalysisV2)
        
        # 1. Aligned BM25 query
        bm25_query = self.build_es_field_aligned_bm25_query(seg, size=size)
        
        standard_bool: Dict[str, Any] = {"must": [bm25_query]}
        if filters:
            standard_bool["filter"] = filters
        if should_boosts:
            standard_bool["should"] = should_boosts
            standard_bool["minimum_should_match"] = 0
        if must_nots:
            standard_bool["must_not"] = must_nots
            
        retrievers = [
            {
                "standard": {
                    "query": {
                        "bool": standard_bool
                    }
                }
            }
        ]
        
        # 2. KNN retrievers
        vector_weights = get_vector_weights(CarInteriorAnalysisV2).copy()
        
        for field_name in vector_fields:
            if field_name not in all_vector_fields:
                continue
            boost = float(vector_weights.get(field_name, 1.0))
            
            knn_clause: Dict[str, Any] = {
                "field": field_name,
                "query_vector": query_vector,
                "k": size,
                "num_candidates": max(size * 5, 100),
                "boost": boost
            }
            
            if filters or must_nots or should_boosts:
                knn_filter_bool: Dict[str, Any] = {}
                if filters:
                    knn_filter_bool["filter"] = filters
                if must_nots:
                    knn_filter_bool["must_not"] = must_nots
                if should_boosts:
                    knn_filter_bool["should"] = should_boosts
                    knn_filter_bool["minimum_should_match"] = 0
                knn_clause["filter"] = {"bool": knn_filter_bool}
                
            retrievers.append({
                "knn": knn_clause
            })
            
        body = {
            "size": size,
            "retriever": {
                "rrf": {
                    "retrievers": retrievers,
                    "rank_constant": 60,
                    "window_size": max(size * 3, 100)
                }
            },
            "_source": {
                "excludes": all_vector_fields
            }
        }
        
        return body


try:
    from models.elasticsearch_index.car_interior_analysis_v2 import CarInteriorAnalysisV2
    query_builder = QueryBuilder()
except Exception:
    query_builder = None
